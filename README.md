# ReAct Agent with LLM Memory System

基于 ReAct 范式的 AI Agent，具备三层记忆系统：长期记忆、每日情景记忆、会话索引召回。

## 项目结构

```
new_agent/
├── run.py                    # 入口
├── agent.py                  # ReAct Agent 主循环
├── agent_config.py           # 配置管理
├── agent_tools.py            # 工具函数（read_file / write_file / run_terminal_command / check_python_file）
├── toolmanager.py            # 工具注册和调用
├── messages.py               # 对话管理 + 记忆存储 + 压缩 + 索引 + 召回
├── prompt_template.py        # System prompt 模板加载
├── Model_Config.py           # API Key（不上传 Git）
│
├── templates/
│   ├── system_prompt.md      # 角色定义和输出格式约束
│   └── compact_prompt.md     # 记忆压缩 LLM prompt 模板
│
├── memory/                   # 运行时生成（不上传 Git）
│   ├── MEMORY.md             # 长期记忆（跨会话累积）
│   ├── YYYY-MM-DD.md         # 每日情景记忆
│   ├── history.jsonl         # 全量对话日志
│   ├── session_index.jsonl   # 会话索引（轻量检索用）
│   └── debug_llm_responses.jsonl  # LLM 原始返回调试
│
├── agent_test.py             # 实验/测试代码
└── tests/                    # 测试数据
```

## 核心功能

### ReAct 循环

```
用户输入 → Thought → Action(工具调用) → Observation → ... → Final Answer
```

支持的工具有 `read_file`、`write_file`、`run_terminal_command`、`check_python_file`，工具定义在 `agent_tools.py` 中。

### 三层记忆系统

| 记忆层 | 文件 | 更新方式 | 作用 |
|--------|------|----------|------|
| 长期记忆 | `MEMORY.md` | LLM 完整覆盖（≤4000字） | 跨会话累积的任务、决策、发现 |
| 每日情景 | `YYYY-MM-DD.md` | 追加式（每天新建） | 当天对话摘要 |
| 会话索引 | `session_index.jsonl` | 追加式（每次压缩生成一条） | 轻量检索 |

### 记忆压缩

当非系统消息数超过 `compact_after_messages` 且交互轮数达到 `refresh_prompt_interval` 时触发：

1. **索引**：LLM 总结当前会话 → 写入 `session_index.jsonl`
2. **压缩**：LLM 合并旧消息 + 现有记忆 → 输出 `<episode>` + `<updated_memory>`
3. **重建**：messages 保留 system prompt + 最近 N 条消息，记忆从文件注入

压缩 LLM 调用失败时自动 fallback 到文本截断。

### 记忆召回

每次新用户输入时，从 `session_index.jsonl` 检索相关历史会话：

- ≤3 条索引：全量返回
- \>3 条：LLM 匹配 → 只返回相关的

召回结果注入 system prompt 的「相关历史记忆」区域。

