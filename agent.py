

import json
import platform
import re
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from string import Template
from typing import Dict, Any, Optional
from prompt_template import react_system_prompt_template
from toolmanager import ToolManager
from agent_config import AgentConfig
from messages import ConversationManager


class ReactAgent:
    """ReAct Agent主类，整合所有功能"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        self.conversation = ConversationManager(self.config, client=self.client)
        self.tool_manager = ToolManager()
        
    def get_operating_system_name(self) -> str:
        """获取操作系统名称"""
        os_map = {
            "Darwin": "macOS",
            "Windows": "Windows", 
            "Linux": "Linux"
        }
        return os_map.get(platform.system(), "Unknown")
        
    def render_system_prompt(self, recalled_memory: str = "") -> str:
        """渲染系统提示模板，并注入记忆内容。"""
        tool_list = self.tool_manager.get_tool_list()

        base = Template(react_system_prompt_template).substitute(
            operating_system=self.get_operating_system_name(),
            tool_list=tool_list,
        )

        # 召回的相关历史记忆（本次会话特定）
        if recalled_memory:
            base += f"\n\n【相关历史记忆】\n{recalled_memory}"

        # 长期记忆 + 每日情景（从文件读取）
        memory = self.conversation.get_memory_for_prompt()
        episode = self.conversation.get_episode_for_prompt()

        if memory:
            base += f"\n\n【长期记忆】\n{memory}"
        if episode:
            base += f"\n\n【今日情景】\n{episode}"

        return base

    def parse_ai_response(self, content: str) -> Dict[str, Any]:
        """解析AI响应内容，将字符串形式返回转化为json"""
        try:
            content = self._strip_markdown_json_fence(content).strip()
            if self.config.show_system_messages:
                print(f"[系统] 解析AI响应内容: {content}")
            return json.loads(content)
        except json.JSONDecodeError:
            if self.config.show_system_messages:
                print("无法解析 JSON 响应，可能是格式错误")
            raise

    def _strip_markdown_json_fence(self, content: str) -> str:
        """去掉模型偶尔包裹的 Markdown JSON 代码块。"""
        match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", content, re.DOTALL)
        return match.group(1) if match else content
            
    def handle_final_answer(self, response_json: Dict[str, Any]) -> str:
        """处理最终答案"""
        final_answer = response_json['final_answer']
        print(f"Answer: {final_answer}")
            
        interaction_count = self.conversation.increment_interaction()
        if self.config.show_system_messages:
            print(f"[系统] 完成第 {interaction_count} 轮交互（final_answer）")
            
        return final_answer
        
    def handle_action(self, response_json: Dict[str, Any]) -> None:
        """处理AI动作执行"""
        action_str = response_json["action"]
        if self.config.show_system_messages:
            print(f"[系统] 执行动作: {action_str}")
            
        try:
            if self.config.show_system_messages:
                print(f"[系统] 正在解析函数调用: {action_str}")
            parsed_actions = self.tool_manager.parse_action_list(action_str)
            result = self.tool_manager.execute_action_list(parsed_actions)
            self.conversation.add_observation(str(result))
            
            if self.config.show_system_messages:
                print(f"观察结果: {result}")
                print(f"[系统] 当前发送总请求数量: {len(self.conversation.messages)}")
                
        except ValueError as e:
            self.conversation.add_error_observation(str(e))
            if self.config.show_system_messages:
                print(f"工具调用错误: {e}")
                print("[系统] 已将错误信息返回给AI")
                
        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            self.conversation.add_error_observation(error_msg)
            if self.config.show_system_messages:
                print(f"工具执行出错: {e}")
                print("[系统] 已将错误信息返回给AI")
                
    def get_user_input(self, prompt: str = "Question: ") -> str:
        """获取用户输入"""
        return input(prompt)

    def should_exit(self, user_input: str) -> bool:
        """判断用户是否主动退出。"""
        return user_input.strip().lower() in {"exit", "quit", "q"}

    def log_model_input(self) -> None:
        """把本次发送给大模型的 messages 写入本地调试文件。"""
        log_path = Path(self.config.model_input_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": self.config.model_name,
            "messages": self.conversation.messages,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[系统] 本次发送给大模型的 messages 已写入: {log_path}")

    def object_to_dict(self, value: Any) -> Any:
        """把 SDK 返回对象转换为可 JSON 序列化的数据。"""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [self.object_to_dict(item) for item in value]
        if isinstance(value, dict):
            return {key: self.object_to_dict(item) for key, item in value.items()}
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        if hasattr(value, "__dict__"):
            return {
                key: self.object_to_dict(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        return str(value)

    def get_nested_value(self, data: Dict[str, Any], path: list[str]) -> Any:
        """读取嵌套字典字段。"""
        current: Any = data
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def log_token_usage(self, response: Any) -> None:
        """记录本次模型调用的 token 消耗和缓存命中。"""
        usage = self.object_to_dict(getattr(response, "usage", None))
        if not isinstance(usage, dict):
            return

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        cached_tokens = (
            self.get_nested_value(usage, ["prompt_tokens_details", "cached_tokens"])
            or usage.get("prompt_cache_hit_tokens")
            or usage.get("cache_hit_tokens")
            or 0
        )
        cache_miss_tokens = (
            usage.get("prompt_cache_miss_tokens")
            or usage.get("cache_miss_tokens")
            or 0
        )

        log_path = Path(self.config.token_usage_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": self.config.model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "cache_miss_tokens": cache_miss_tokens,
            "raw_usage": usage,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.config.show_system_messages:
            print(
                "[系统] Token 消耗已记录: "
                f"input={prompt_tokens}, output={completion_tokens}, "
                f"total={total_tokens}, cached={cached_tokens}, log={log_path}"
            )
        
    def process_turn(self) -> Optional[bool]:
        """处理一轮对话，返回 True 表示需要新输入，False 表示继续推理，None 表示退出。"""
        try:
            if self.config.show_system_messages:
                self.log_model_input()

            # 调用AI模型
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=self.conversation.messages,
            )
            self.log_token_usage(response)
            
            content = response.choices[0].message.content or ""
            self.conversation.add_message("assistant", content)
            
            if self.config.show_system_messages:
                print("[系统]模型完整回复:", content)
                
            # 解析AI响应
            try:
                response_json = self.parse_ai_response(content)
            except :
                obs_msg = '{"Incorrect_answer_format": "回答解析失败，请检查回复是否为合理json格式后重新回答"}'
                self.conversation.add_message("user", obs_msg)
                return False

            if "final_answer" in response_json:
                self.handle_final_answer(response_json)
                return True  # 需要新的用户输入
                
            elif "action" in response_json:
                self.handle_action(response_json)
                return False  # 继续AI推理
                
            else:
                if self.config.show_system_messages:
                    print("[系统] AI响应中没有找到 'final_answer' 或 'action' 字段")
                    print(f"[系统] 响应内容: {response_json}")
                return False
                
        except json.JSONDecodeError:
            return False
        except Exception as e:
            print(f"API 调用失败: {e}")
            return None  # 出错时退出
            
                        
        
        
            
    def run(self) -> None:
        """运行Agent主循环"""
        print("=== ReAct Agent 启动 ===")
        
        # 初始用户输入
        user_input = self.get_user_input()
        if self.should_exit(user_input):
            print("已退出。")
            return

        # 从历史索引召回相关记忆
        recalled = self.conversation.recall_from_index(user_input)

        system_prompt = self.render_system_prompt(recalled_memory=recalled)
        self.conversation.add_system_message(user_input, system_prompt)
        
        step_count = 0 # 步数，非final结果数，执行一次action算一次
        while step_count < self.config.max_steps:
            step_count += 1
            
            need_new_input = self.process_turn()

            if need_new_input is None:
                return
            
            if need_new_input:
                # 检查是否需要刷新系统提示
                if self.conversation.should_refresh_prompt():
                    if self.config.show_system_messages:
                        print(f"[系统] 达到 {self.config.refresh_prompt_interval} 轮交互，重新添加系统提示")
                        
                    user_input = self.get_user_input()
                    if self.should_exit(user_input):
                        print("已退出。")
                        return

                    # 召回相关历史记忆
                    recalled = self.conversation.recall_from_index(user_input)
                    system_prompt = self.render_system_prompt(recalled_memory=recalled)
                    self.conversation.refresh_context_with_prompt(user_input, system_prompt)

                    if self.config.show_system_messages:
                        print(f"[系统] 系统提示已刷新，当前 {len(self.conversation.messages)} 条上下文消息")
                else:
                    user_input = self.get_user_input()
                    if self.should_exit(user_input):
                        print("已退出。")
                        return

                    self.conversation.add_user_question(user_input)
                    
                step_count = 0  # 重置步骤计数
                
        print("任务未完成，已达到最大步骤")
