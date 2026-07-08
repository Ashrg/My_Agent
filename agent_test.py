"""
记忆压缩对比实验模块

对比两种压缩方式:
  A. 原始截断压缩 (truncation) — summarize_context() 当前的字符截断方式
  B. LLM 语义压缩 (llm)         — 调用大模型做结构化语义摘要

用法:
  from agent_test import run_compression_comparison

  # 在 compact_context 触发时调用
  comparator = CompressionComparator(client, model_name="deepseek-v4-pro")
  report = comparator.compare(
      old_messages=messages_to_compress,
      existing_summary=current_memory_summary,
      max_memory_chars=4000,
  )
  # report 包含两种方式的对比数据
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict


# ============================================================
# 消息预处理：按类型差异化处理
# ============================================================

def preprocess_message(msg: Dict[str, Any]) -> Optional[str]:
    """将一条消息转换为压缩友好的文本表示。

    不同消息类型采用不同策略：
      - thought     → 丢弃（每次重新推理即可）
      - action      → 保留工具名 + 参数摘要
      - observation → 保留（但超长时截断）
      - error       → 完整保留
      - 用户问题     → 完整保留
      - final_answer → 保留结论
    """
    content = msg.get("content", "")
    role = msg.get("role", "")

    if role == "system":
        return None  # system prompt 不参与压缩

    # 尝试解析 JSON 内容
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    # ---- 用户消息 ----
    if role == "user":
        # 用户原始问题 → 完整保留
        if isinstance(parsed, dict) and "observation" in parsed:
            obs = parsed["observation"]
            return _format_observation(obs)
        elif content.startswith("question:"):
            return f"[用户问题] {content}"
        elif content.startswith("历史摘要："):
            return None  # 旧摘要不重复参与压缩
        elif content.startswith("Incorrect_answer_format"):
            return "[格式错误] 模型回复格式不正确，已要求重新回答"
        else:
            return f"[用户] {_truncate_text(content, 500)}"

    # ---- 助手消息 ----
    if role == "assistant":
        if not isinstance(parsed, dict):
            return f"[助手] {_truncate_text(content, 300)}"

        # 有 final_answer → 提取结论
        if "final_answer" in parsed:
            return f"[最终回答] {parsed['final_answer']}"

        # 有 action → 提取工具调用清单
        if "action" in parsed:
            actions = parsed["action"]
            if isinstance(actions, list):
                tools_called = []
                for a in actions:
                    tool = a.get("tool", "unknown")
                    params = {k: _truncate_text(str(v), 80) for k, v in a.items() if k != "tool"}
                    tools_called.append(f"{tool}({json.dumps(params, ensure_ascii=False)})")
                thought = parsed.get("thought", "")
                return f"[执行动作] {', '.join(tools_called)}" + (f" | 思考: {_truncate_text(thought, 100)}" if thought else "")

        # thought 字段 → 丢弃
        if "thought" in parsed and len(parsed) == 1:
            return None

        return f"[助手] {_truncate_text(content, 200)}"

    return f"[{role}] {_truncate_text(content, 200)}"


def _format_observation(obs_text: str) -> str:
    """格式化 observation：超长时截断，保留首尾"""
    obs_str = str(obs_text)
    if len(obs_str) <= 800:
        return f"[工具结果] {obs_str}"
    # 超长：保留前 300 + 后 300 + 统计
    return (
        f"[工具结果-截断] 总长{len(obs_str)}字符\n"
        f"  {obs_str[:300]}\n  ...(省略{len(obs_str)-600}字符)...\n  {obs_str[-300:]}"
    )


def _truncate_text(text: str, max_len: int) -> str:
    """截断文本到 max_len，如果截断则加省略标记"""
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...[截断,原长{len(text)}]"


# ============================================================
# 原始截断压缩（复现当前 summarize_context 逻辑）
# ============================================================

def truncation_summarize(
    messages: List[Dict[str, Any]],
    existing_summary: str = "",
    max_chars: int = 4000,
) -> str:
    """复现当前的 summarize_context() 逻辑：拼接 + 字符截断。

    不修改原代码，这里独立实现一份供对比使用。
    """
    # 过滤 system 消息
    non_system = [m for m in messages if m.get("role") != "system"]

    if not non_system:
        return existing_summary

    old_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in non_system
    )

    summary_parts = [p for p in [existing_summary, old_text] if p]
    summary = "\n".join(summary_parts)

    return summary[-max_chars:]


# ============================================================
# LLM 语义压缩
# ============================================================

COMPRESSION_SYSTEM_PROMPT = """你是对话历史压缩器。你的任务是将多轮对话压缩成结构化的 JSON 摘要。

## 压缩原则

必须保留:
- 用户原始问题/目标（完整保留）
- 已完成的工具调用名和关键结论
- 所有错误和异常信息（完整保留）
- 用户偏好和明确的约束条件
- agent 做出的关键决策

必须丢弃:
- AI 的中间推理过程（thought 字段）
- 重复的 observation（只保留最新一次）
- 纯确认性消息（"好的""继续""明白了"）
- 工具输出中的冗余内容（如完整文件内容 → 只保留文件名和行数）

## 输出格式

必须严格输出以下 JSON（不要用 markdown 代码块包裹）:
{
  "task": "用户的核心任务（一句话）",
  "completed": [{"action": "工具名", "result": "关键结论（一句话）"}],
  "key_findings": ["发现1", "发现2"],
  "errors": ["完整错误信息"],
  "decisions": ["agent 做出的关键决策"],
  "pending": ["尚未完成的事项"]
}"""


def llm_summarize(
    client,
    model_name: str,
    messages: List[Dict[str, Any]],
    existing_summary: str = "",
    max_summary_tokens: int = 500,
) -> Dict[str, Any]:
    """用 LLM 做语义压缩。

    Args:
        client: OpenAI 客户端
        model_name: 模型名
        messages: 要被压缩的旧消息列表
        existing_summary: 已有的记忆摘要
        max_summary_tokens: 摘要 token 上限

    Returns:
        {
            "summary_text": str,    # 压缩后的文本摘要（供注入上下文）
            "summary_json": dict,   # 结构化的压缩结果
            "usage": {...},         # token 消耗
            "duration_ms": float,   # 耗时
        }
    """
    # 1. 预处理消息
    processed_lines = []
    for msg in messages:
        line = preprocess_message(msg)
        if line:
            processed_lines.append(line)

    messages_text = "\n".join(processed_lines)

    # 如果消息太少，不需要压缩
    if len(messages_text.strip()) < 100:
        return {
            "summary_text": existing_summary or messages_text,
            "summary_json": {},
            "usage": None,
            "duration_ms": 0,
        }

    # 2. 构建压缩请求
    existing_text = f"\n【已有记忆摘要】\n{existing_summary}" if existing_summary else "（无已有记忆）"

    user_prompt = f"""{existing_text}

【待压缩的对话记录】（按时间顺序）
{messages_text}

请将以上对话压缩为结构化 JSON 摘要。摘要总长度不超过 {max_summary_tokens} tokens。"""

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # 低温度，保证输出稳定
            max_tokens=max_summary_tokens,
        )
    except Exception as e:
        # LLM 调用失败 → 退回到截断压缩
        fallback = truncation_summarize(messages, existing_summary)
        return {
            "summary_text": fallback,
            "summary_json": {"error": str(e), "fallback": "truncation"},
            "usage": None,
            "duration_ms": (time.time() - start_time) * 1000,
        }

    duration_ms = (time.time() - start_time) * 1000

    content = response.choices[0].message.content or "{}"
    usage = getattr(response, "usage", None)

    # 3. 解析 LLM 输出的 JSON
    summary_json = _safe_parse_json(content)

    # 4. 将结构化 JSON 转为可注入上下文的文本
    summary_text = _json_to_text(summary_json)

    return {
        "summary_text": summary_text,
        "summary_json": summary_json,
        "usage": _usage_to_dict(usage),
        "duration_ms": duration_ms,
    }


def _safe_parse_json(content: str) -> dict:
    """安全解析 LLM 输出的 JSON，处理各种格式问题"""
    import re

    content = content.strip()

    # 去掉可能的 markdown 代码块
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", content, re.DOTALL)
    if match:
        content = match.group(1)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 尝试提取 JSON 对象
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"raw": content, "parse_error": True}


def _json_to_text(data: dict) -> str:
    """将结构化压缩 JSON 转为可注入上下文的紧凑文本"""
    if not data or data.get("parse_error"):
        return str(data.get("raw", ""))

    parts = []

    if data.get("task"):
        parts.append(f"任务: {data['task']}")

    if data.get("completed"):
        items = [f"  {c.get('action','?')}: {c.get('result','')}" for c in data["completed"]]
        parts.append(f"已完成:\n" + "\n".join(items))

    if data.get("key_findings"):
        parts.append(f"关键发现:\n" + "\n".join(f"  · {f}" for f in data["key_findings"]))

    if data.get("errors"):
        parts.append(f"错误记录:\n" + "\n".join(f"  ⚠ {e}" for e in data["errors"]))

    if data.get("decisions"):
        parts.append(f"已做决策:\n" + "\n".join(f"  ✓ {d}" for d in data["decisions"]))

    if data.get("pending"):
        parts.append(f"待处理:\n" + "\n".join(f"  ○ {p}" for p in data["pending"]))

    return "\n".join(parts)


def _usage_to_dict(usage: Any) -> Optional[dict]:
    """将 SDK 返回的 usage 对象转为 dict"""
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "prompt_cache_hit_tokens": getattr(usage, "prompt_cache_hit_tokens", None),
        "prompt_cache_miss_tokens": getattr(usage, "prompt_cache_miss_tokens", None),
    }


# ============================================================
# 估算 token 数（简单估算，不依赖 tiktoken）
# ============================================================

def estimate_tokens(text: str) -> int:
    """简单 token 估算：中文 ~1 token/字，英文 ~1 token/3.5 字符"""
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    other_chars = len(text) - chinese_chars
    return chinese_chars + other_chars // 3


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """估算消息列表的总 token 数"""
    total = 0
    for m in messages:
        if m.get("role") != "system":
            total += estimate_tokens(str(m.get("content", "")))
    return total


# ============================================================
# 对比实验框架
# ============================================================

@dataclass
class CompressionResult:
    """单次压缩的结果"""
    method: str                     # "truncation" | "llm"
    input_message_count: int        # 输入消息数
    input_estimated_tokens: int     # 输入估算 token 数
    output_text: str                # 压缩后的文本
    output_chars: int               # 输出字符数
    output_estimated_tokens: int    # 输出估算 token 数
    compression_ratio: float        # 压缩比 = 输出/输入
    duration_ms: float              # 压缩耗时
    api_usage: Optional[dict] = None  # API token 消耗
    error: Optional[str] = None     # 错误信息


@dataclass
class ComparisonReport:
    """一次对比的完整报告"""
    timestamp: str
    truncation: CompressionResult
    llm: CompressionResult
    diff: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.diff = {
            "char_reduction": (
                f"{(1 - self.llm.output_chars / max(self.truncation.output_chars, 1)) * 100:.1f}%"
                if self.truncation.output_chars > 0 else "N/A"
            ),
            "token_reduction": (
                f"{(1 - self.llm.output_estimated_tokens / max(self.truncation.output_estimated_tokens, 1)) * 100:.1f}%"
                if self.truncation.output_estimated_tokens > 0 else "N/A"
            ),
            "llm_extra_cost_tokens": (
                sum([
                    (self.llm.api_usage or {}).get("prompt_tokens", 0),
                    (self.llm.api_usage or {}).get("completion_tokens", 0),
                ])
                if self.llm.api_usage else 0
            ),
            "llm_slower_by_ms": f"{self.llm.duration_ms - self.truncation.duration_ms:.0f}",
        }


class CompressionComparator:
    """压缩方式对比器

    同时运行截断压缩和 LLM 压缩，记录对比数据。
    不修改原系统行为——始终返回截断压缩的结果给调用方。
    """

    def __init__(
        self,
        client,
        model_name: str = "deepseek-v4-pro",
        max_memory_chars: int = 4000,
        max_summary_tokens: int = 500,
        report_dir: Optional[str] = None,
    ):
        self.client = client
        self.model_name = model_name
        self.max_memory_chars = max_memory_chars
        self.max_summary_tokens = max_summary_tokens
        self.report_dir = Path(report_dir) if report_dir else Path(__file__).parent
        self.history: List[ComparisonReport] = []

    def compare(
        self,
        messages: List[Dict[str, Any]],
        existing_summary: str = "",
    ) -> str:
        """运行对比实验。

        同时执行截断压缩和 LLM 压缩，记录对比数据。
        始终返回截断压缩的结果（不改变调用方的行为）。

        Args:
            messages: 要压缩的消息列表
            existing_summary: 已有的记忆摘要

        Returns:
            str: 截断压缩的结果（保证系统正常运行）
        """
        non_system = [m for m in messages if m.get("role") != "system"]
        input_tokens = estimate_messages_tokens(non_system)

        # ---- A. 截断压缩（当前方式）----
        t0 = time.time()
        truncation_output = truncation_summarize(
            non_system, existing_summary, self.max_memory_chars
        )
        truncation_duration = (time.time() - t0) * 1000

        truncation_result = CompressionResult(
            method="truncation",
            input_message_count=len(non_system),
            input_estimated_tokens=input_tokens,
            output_text=truncation_output,
            output_chars=len(truncation_output),
            output_estimated_tokens=estimate_tokens(truncation_output),
            compression_ratio=estimate_tokens(truncation_output) / max(input_tokens, 1),
            duration_ms=truncation_duration,
        )

        # ---- B. LLM 语义压缩 ----
        llm_output = llm_summarize(
            client=self.client,
            model_name=self.model_name,
            messages=non_system,
            existing_summary=existing_summary,
            max_summary_tokens=self.max_summary_tokens,
        )

        llm_result = CompressionResult(
            method="llm",
            input_message_count=len(non_system),
            input_estimated_tokens=input_tokens,
            output_text=llm_output["summary_text"],
            output_chars=len(llm_output["summary_text"]),
            output_estimated_tokens=estimate_tokens(llm_output["summary_text"]),
            compression_ratio=estimate_tokens(llm_output["summary_text"]) / max(input_tokens, 1),
            duration_ms=llm_output["duration_ms"],
            api_usage=llm_output.get("usage"),
            error=llm_output.get("summary_json", {}).get("error") if isinstance(llm_output.get("summary_json"), dict) else None,
        )

        # ---- 记录对比 ----
        report = ComparisonReport(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            truncation=truncation_result,
            llm=llm_result,
        )
        self.history.append(report)

        return truncation_output  # 返回截断结果，保持系统行为不变

    def print_latest_report(self) -> None:
        """打印最近一次对比报告"""
        if not self.history:
            print("暂无对比数据")
            return

        r = self.history[-1]
        t = r.truncation
        l = r.llm

        print("\n" + "=" * 70)
        print("  记忆压缩对比报告")
        print("=" * 70)
        print(f"  时间: {r.timestamp}")
        print(f"  输入: {t.input_message_count} 条消息, ~{t.input_estimated_tokens} tokens")
        print("-" * 70)
        print(f"  {'指标':<25} {'截断压缩':>18} {'LLM压缩':>18}")
        print("-" * 70)
        print(f"  {'输出字符数':<25} {t.output_chars:>18,} {l.output_chars:>18,}")
        print(f"  {'输出估算 token':<25} {t.output_estimated_tokens:>18,} {l.output_estimated_tokens:>18,}")
        print(f"  {'压缩比':<25} {t.compression_ratio:>18.1%} {l.compression_ratio:>18.1%}")
        print(f"  {'耗时(ms)':<25} {t.duration_ms:>18.0f} {l.duration_ms:>18.0f}")

        if l.api_usage:
            u = l.api_usage
            cost_tokens = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
            cached = u.get("prompt_cache_hit_tokens") or 0
            print(f"  {'LLM压缩API消耗':<25} {cost_tokens:>18,} tokens (cached: {cached:,})")

        print("-" * 70)

        # 展示 LLM 压缩的摘要预览
        print(f"\n  [LLM 压缩结果预览]")
        for line in l.output_text.split("\n")[:15]:
            print(f"  {line}")
        if l.output_text.count("\n") > 15:
            print(f"  ... (共 {len(l.output_text)} 字符)")

        # 展示 LLM 结构化 JSON
        if r.llm.api_usage:
            for report_item in self.history:
                llm_out = llm_summarize.__wrapped__ if hasattr(llm_summarize, '__wrapped__') else None

        print("\n" + "=" * 70)

    def print_summary(self) -> None:
        """打印累计统计摘要"""
        if not self.history:
            print("暂无对比数据")
            return

        n = len(self.history)
        avg_trunc_ratio = sum(h.truncation.compression_ratio for h in self.history) / n
        avg_llm_ratio = sum(h.llm.compression_ratio for h in self.history) / n
        total_llm_tokens = sum(
            h.diff.get("llm_extra_cost_tokens", 0) for h in self.history
        )
        avg_trunc_chars = sum(h.truncation.output_chars for h in self.history) / n
        avg_llm_chars = sum(h.llm.output_chars for h in self.history) / n

        print("\n" + "=" * 70)
        print("  压缩对比累计统计")
        print("=" * 70)
        print(f"  对比次数: {n}")
        print(f"  平均截断压缩比: {avg_trunc_ratio:.1%}")
        print(f"  平均 LLM 压缩比: {avg_llm_ratio:.1%}")
        print(f"  LLM 额外优于截断: {(avg_trunc_chars - avg_llm_chars) / max(avg_trunc_chars, 1) * 100:.1f}%")
        print(f"  LLM 压缩累计额外 token 消耗: {total_llm_tokens:,}")
        print("=" * 70 + "\n")

    def save_report(self, filename: Optional[str] = None) -> Path:
        """保存对比报告到 JSON 文件"""
        if filename is None:
            filename = f"compression_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.report_dir / filename

        report_data = {
            "summary": {
                "total_comparisons": len(self.history),
                "avg_truncation_compression_ratio": sum(h.truncation.compression_ratio for h in self.history) / max(len(self.history), 1),
                "avg_llm_compression_ratio": sum(h.llm.compression_ratio for h in self.history) / max(len(self.history), 1),
            },
            "history": [asdict(h) for h in self.history],
        }

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print(f"[对比报告] 已保存到: {filepath}")
        return filepath


# ============================================================
# 便捷函数：在 agent 中集成对比
# ============================================================

def create_comparator_from_agent(agent) -> CompressionComparator:
    """从 ReactAgent 实例创建压缩对比器。

    用法:
        comparator = create_comparator_from_agent(agent)
        # 在 compact_context 时:
        new_summary = comparator.compare(old_messages, existing_summary)
        # new_summary 是截断压缩的结果，LLM 结果记录在 comparator.history 中
    """
    return CompressionComparator(
        client=agent.client,
        model_name=agent.config.model_name,
        max_memory_chars=agent.config.max_memory_chars,
        max_summary_tokens=min(agent.config.max_memory_chars // 3, 800),
    )


# ============================================================
# 离线测试：不启动完整 agent，直接对比
# ============================================================

def run_offline_comparison(
    messages_path: Optional[str] = None,
    sample_messages: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """离线运行压缩对比（不需要启动 agent）。

    可以从 model_input.jsonl 加载真实对话数据，或直接传入消息列表。

    用法:
        # 从日志文件加载
        run_offline_comparison(messages_path="model_input.jsonl")

        # 或直接传入示例消息
        run_offline_comparison(sample_messages=[...])
    """
    from openai import OpenAI

    client = OpenAI(
        api_key="sk-fdcc52984c944e9eb6a38412c16783bb",
        base_url="https://api.deepseek.com",
    )

    # 加载消息
    if sample_messages:
        messages = sample_messages
    elif messages_path:
        messages = _load_messages_from_log(messages_path)
    else:
        print("请提供 messages_path 或 sample_messages")
        return

    if not messages:
        print("未加载到消息数据")
        return

    print(f"加载了 {len(messages)} 条消息")

    comparator = CompressionComparator(client=client, model_name="deepseek-v4-pro")

    # 运行对比
    comparator.compare(messages)

    # 打印报告
    comparator.print_latest_report()

    # 保存
    comparator.save_report()


def _load_messages_from_log(filepath: str) -> List[Dict[str, Any]]:
    """从 model_input.jsonl 加载消息"""
    path = Path(filepath)
    if not path.exists():
        print(f"文件不存在: {filepath}")
        return []

    with path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        return []

    # 取最后一条记录的消息（最长、最需要压缩的）
    last_record = records[-1]
    return last_record.get("messages", [])


# ============================================================
# 直接运行
# ============================================================

if __name__ == "__main__":
    # 从 model_input.jsonl 加载真实数据做离线对比
    log_path = Path(__file__).parent / "model_input.jsonl"
    if log_path.exists():
        run_offline_comparison(messages_path=str(log_path))
    else:
        print(f"日志文件不存在: {log_path}")
        print("请先运行 agent 生成 model_input.jsonl，或传入 sample_messages")
