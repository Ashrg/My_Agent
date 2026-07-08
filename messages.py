import json
import re
import time
from datetime import datetime
from pathlib import Path
from string import Template
from typing import List, Dict, Any, Optional

from agent_config import AgentConfig


#  MemoryStore：管理记忆文件的读写

class MemoryStore:
    """三层记忆文件管理。
    memory/
    ├── MEMORY.md       长期记忆
    ├── YYYY-MM-DD.md   每日情景记忆
    └── history.jsonl   对话日志 纯记录
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.memory_file = memory_dir / "MEMORY.md"
        self.history_file = memory_dir / "history.jsonl"

    # 初始化 

    def ensure_files(self):
        """首次运行时创建空的记忆文件。"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            self.memory_file.write_text("# 长期记忆\n\n", encoding="utf-8")
        if not self.history_file.exists():
            self.history_file.touch()

    # 长期记忆

    def read_memory(self) -> str:
        """读取 MEMORY.md 全文。"""
        self.ensure_files()
        return self.memory_file.read_text(encoding="utf-8")

    def write_memory(self, text: str):
        """覆盖写入 MEMORY.md。"""
        self.ensure_files()
        self.memory_file.write_text(text.strip() + "\n", encoding="utf-8")

    # 每日情景记忆 

    def _today_episode_file(self) -> Path:
        return self.memory_dir / f"{datetime.now():%Y-%m-%d}.md"

    def read_today_episode(self) -> str:
        """读取今日情景记忆全文。"""
        self.ensure_files()
        path = self._today_episode_file()
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def append_episode(self, text: str):
        """追加一段情景记忆。"""
        self.ensure_files()
        with self._today_episode_file().open("a", encoding="utf-8") as f:
            f.write("\n" + text.strip() + "\n")

    # 对话日志

    def append_history(self, message: dict):
        """追加一条消息到对话日志。"""
        self.ensure_files()
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "role": message.get("role"),
            "content": _json_safe(message.get("content")),
        }
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 会话索引

    def _session_index_file(self) -> Path:
        return self.memory_dir / "session_index.jsonl"

    def load_session_index(self) -> List[Dict]:
        """读取会话索引，返回列表。"""
        path = self._session_index_file()
        if not path.exists():
            return []
        entries = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        entries.reverse()  # 最新在前
        return entries

    def append_session_index(self, entry: Dict, max_entries: int = 100):
        """追加一条索引，超过上限时保留最新的。"""
        self.ensure_files()
        path = self._session_index_file()

        # 读取现有条目
        entries = []
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        # 追加并裁剪
        entries.append(entry)
        if len(entries) > max_entries:
            entries = entries[-max_entries:]

        # 重写文件
        with path.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


#  ConversationManager：消息管理 + 压缩

class ConversationManager:
    """对话管理类。
    记忆由 agent.py 在构建 system prompt 时从 MemoryStore 读文件注入。
    """

    # 从模板文件加载压缩 prompt
    _COMPACT_TEMPLATE_PATH = Path(__file__).parent / "templates" / "compact_prompt.md"
    _COMPACT_TEMPLATE = Template(
        _COMPACT_TEMPLATE_PATH.read_text(encoding="utf-8")
        if _COMPACT_TEMPLATE_PATH.exists()
        else "${old_conversation}\n${current_memory}\n${today_episode}"
    )

    def __init__(self, config: AgentConfig, client=None):
        self.config = config
        self.client = client
        self.messages: List[Dict[str, Any]] = []
        self.interaction_count = 0

        # 记忆存储
        self.store = MemoryStore(Path(self.config.memory_dir))
        self.store.ensure_files()

    # 消息操作

    def add_message(self, role: str, content: str) -> None:
        """添加消息到对话历史，同时写入日志。"""
        message = {"role": role, "content": content}
        if self.config.show_system_messages:
            print(f"[对话管理] 添加消息: {role} - {content}")
        self.messages.append(message)
        self.store.append_history(message)

    def add_system_message(self, user_question: str, system_prompt: str) -> None:
        """添加系统提示和用户问题。不再插入记忆 user 消息。"""
        self.add_message("system", system_prompt)
        self.add_user_question(user_question)

    def add_user_question(self, user_question: str) -> None:
        """添加用户问题。"""
        self.add_message("user", f"question: {user_question}")

    def add_observation(self, observation: str) -> None:
        """添加工具执行观察结果。"""
        obs_msg = json.dumps({"observation": observation}, ensure_ascii=False)
        self.add_message("user", obs_msg)

    def add_error_observation(self, error_msg: str) -> None:
        """添加错误观察结果。"""
        error_obs = json.dumps({"observation": f"错误: {error_msg}"}, ensure_ascii=False)
        self.add_message("user", error_obs)

    def increment_interaction(self) -> int:
        """增加交互计数，每次 final_answer 后调用。"""
        self.interaction_count += 1
        return self.interaction_count

    #  压缩触发 

    def should_refresh_prompt(self) -> bool:
        """检查是否需要刷新系统提示。"""
        return self.interaction_count % self.config.refresh_prompt_interval == 0

    def _should_compact(self) -> bool:
        """消息数超过阈值时触发压缩。"""
        non_system = [m for m in self.messages if m.get("role") != "system"]
        return len(non_system) > self.config.compact_after_messages

    #  压缩执行 调用大模型

    def refresh_context_with_prompt(self, user_question: str, system_prompt: str) -> None:
        """刷新上下文并添加新的系统提示。"""
        self.compact_context(system_prompt)
        self.add_user_question(user_question)

    def compact_context(self, system_prompt: str) -> None:
        """压缩上下文：先生成会话索引，再 LLM 压缩。"""
        if self._should_compact():
            if self.config.auto_index_session: #压缩前创建一次索引
                self._index_current_session()  
            self._compact_via_llm()

        # 重建 messages：只保留 system prompt + 最近消息，不插入记忆
        recent_messages = [
            m for m in self.messages if m.get("role") != "system"
        ][-self.config.max_recent_messages:]

        self.messages = [
            {"role": "system", "content": system_prompt},
            *recent_messages,
        ]

    def _compact_via_llm(self):
        """用 LLM 压缩旧消息，输出 episode + updated_memory。失败时 fallback 到截断。"""
        old_messages = [
            m for m in self.messages if m.get("role") != "system"
        ][:-self.config.max_recent_messages]

        if not old_messages or self.client is None:
            self._compact_via_truncation()
            return

        # 构建压缩 prompt
        prompt = self._COMPACT_TEMPLATE.substitute(
            old_conversation=_messages_to_text(old_messages),
            current_memory=self.store.read_memory(),
            today_episode=self.store.read_today_episode(),
            now_hhmm=datetime.now().strftime("%H:%M"),
            max_memory_chars=str(self.config.max_memory_chars),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "你是记忆整理员。请严格按要求输出 XML，不要输出额外解释。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=self.config.llm_compression_max_tokens,
            )
            text = response.choices[0].message.content or ""

        except Exception as exc:
            if self.config.show_system_messages:
                print(f"[压缩] LLM 调用失败，fallback 到截断: {exc}")
            self._compact_via_truncation()
            return

        # 调试：写入 LLM 原始返回
        self._write_debug_log("compact", text)

        # 解析 XML
        episode = _extract_xml_tag(text, "episode")
        updated_memory = _extract_xml_tag(text, "updated_memory")

        # 写入文件
        if episode:
            self.store.append_episode(episode)
        if updated_memory:
            self.store.write_memory(updated_memory)

        if self.config.show_system_messages:
            print(f"[压缩] old={len(old_messages)}条 → episode={len(episode)}字 + memory={len(updated_memory)}字")

    def _compact_via_truncation(self):
        """截断式压缩。"""
        old_messages = [
            m for m in self.messages if m.get("role") != "system"
        ][:-self.config.max_recent_messages]

        if not old_messages:
            return

        old_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in old_messages
        )
        existing = self.store.read_memory()
        summary = "\n".join(p for p in [existing, old_text] if p)
        truncated = summary[-self.config.max_memory_chars:]

        self.store.write_memory(truncated)

        if self.config.show_system_messages:
            print(f"[压缩-截断] 长期记忆更新为 {len(truncated)} 字符")

    # 上下文重建 

    def rebuild_context(self, system_prompt: str, summary: str, user_question: str) -> None:
        """用 system prompt + 新问题重建上下文。记忆从文件读，不在此处注入。"""
        self.store.write_memory(summary)
        self.messages = [{"role": "system", "content": system_prompt}]
        self.add_user_question(user_question)

    #  会话索引 

    def _index_current_session(self):
        """用 LLM 生成本次会话摘要，写入 session_index.jsonl。"""
        all_messages = [m for m in self.messages if m.get("role") != "system"]
        if len(all_messages) < 4 or self.client is None:
            return

        messages_text = _messages_to_text(all_messages)
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")

        prompt = (
            f"请用中文总结以下对话会话，输出 JSON:\n\n"
            f"对话内容:\n{messages_text}\n\n"
            f'输出格式: {{"summary":"一段150字以内的会话总结",'
            f'"topics":["话题1","话题2"],'
            f'"key_outcomes":["关键结论1"],"user_preferences":["偏好1"]}}\n'
            f"只输出 JSON，不要额外解释。"
        )

        try:
            model = self.config.recall_model or self.config.model_name
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens= self.config.index_max_tokens,
            )
            text = response.choices[0].message.content or ""
            # 调试：写入 LLM 原始返回
            self._write_debug_log("index", text)
            parsed = _safe_parse_json(text)
        except Exception as exc:
            if self.config.show_system_messages:
                print(f"[索引] LLM 调用失败: {exc}")
            return

        entry = {
            "session_id": session_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": parsed.get("summary", "")[:200],
            "topics": parsed.get("topics", []),
            "key_outcomes": parsed.get("key_outcomes", []),
            "user_preferences": parsed.get("user_preferences", []),
            "message_count": len(all_messages),
        }
        self.store.append_session_index(entry, max_entries=self.config.max_session_index_entries)

        if self.config.show_system_messages:
            print(f"[索引] 会话 {session_id} 已索引: {entry['summary'][:60]}...")

    # 记忆召回 

    def recall_from_index(self, user_question: str) -> str:
        """从历史会话索引召回与当前问题相关的记忆。

        读取 session_index.jsonl → LLM 匹配 → 返回记忆文本。
        """
        entries = self.store.load_session_index()
        if not entries:
            return ""

        # 少量会话 全量返回，不需要 LLM
        if len(entries) <= 3:
            return self._entries_to_memory_text(entries)

        # 大量会话 用 LLM 匹配
        index_text = self._index_to_llm_text(entries)

        try:
            model = self.config.recall_model or self.config.model_name
            response = self.client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"用户当前问题: {user_question}\n\n"
                        f"历史会话索引:\n{index_text}\n\n"
                        f"请找出与当前问题相关的历史会话 ID 列表。"
                        f'输出 JSON: {{"relevant_sessions":["id1","id2"]}}'
                        f"只选真正相关的，没有则输出空数组。"
                    ),
                }],
                temperature=0,
                max_tokens= self.config.llm_compression_max_tokens,
            )
            text = response.choices[0].message.content or ""
            parsed = _safe_parse_json(text)
            relevant_ids = parsed.get("relevant_sessions", [])
        except Exception as exc:
            if self.config.show_system_messages:
                print(f"[召回] LLM 匹配失败: {exc}")
            return ""

        # 筛选命中的条目
        relevant = [e for e in entries if e.get("session_id") in relevant_ids]

        if self.config.show_system_messages:
            print(f"[召回] 从 {len(entries)} 次历史会话中匹配到 {len(relevant)} 个相关")

        return self._entries_to_memory_text(relevant)

   

    # debug用 看大模型压缩和索引输出结果 存在文件中

    def _write_debug_log(self, tag: str, text: str):
        """把 LLM 原始返回写入 debug 文件。"""
        log_path = Path(self.config.memory_dir) / "debug_llm_responses.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "tag": tag,
            "text": text,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    #  记忆读取

    def get_memory_for_prompt(self) -> str:
        """获取长期记忆内容，供 agent 注入 system prompt。"""
        return self.store.read_memory()

    def get_episode_for_prompt(self) -> str:
        """获取今日情景记忆，供 agent 注入 system prompt。"""
        return self.store.read_today_episode()


#  解析工具

def _json_safe(value):
    """把 SDK 对象转为 JSON 可序列化的值。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return _json_safe(value.dict())
    return str(value)


def _messages_to_text(messages: List[Dict]) -> str:
    """将消息列表转为纯文本，供压缩 prompt 使用。"""
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = json.dumps(_json_safe(m.get("content")), ensure_ascii=False)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _safe_parse_json(content: str) -> dict:
    """安全解析 LLM 输出的 JSON"""
    content = content.strip()
    m = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", content, re.DOTALL)
    if m:
        content = m.group(1)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {}


def _extract_xml_tag(text: str, tag: str) -> str:
    """从 LLM 输出中提取 <tag>...</tag> 的内容。"""
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# 索引转换
def _index_to_llm_text(entries: List[Dict]) -> str:
    """把索引条目转成供 LLM 匹配的紧凑文本。"""
    lines = []
    for e in entries:#
        topics = ", ".join(e.get("topics", []))
        summary = e.get("summary", "")[:150]
        lines.append(f"[{e.get('session_id','?')}] {e.get('date','?')} | {summary} | 话题: {topics}")
    return "\n".join(lines)

def _entries_to_memory_text(entries: List[Dict]) -> str:
    """把索引条目转成可注入 system prompt 的记忆文本。"""
    if not entries:
        return ""
    parts = []
    for e in entries:
        parts.append(
            f"- [{e.get('date','')}] {e.get('summary','')}"
        )
        for outcome in e.get("key_outcomes", []):
            parts.append(f"    · {outcome}")
        for pref in e.get("user_preferences", []):
            parts.append(f"    · 用户偏好: {pref}")
    return "\n".join(parts)
