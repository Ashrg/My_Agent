from pathlib import Path
from Model_Config import API_KEY

class AgentConfig:
    """Agent配置类,管理所有超参数"""
    
    def __init__(self):
        # API配置
        self.api_key = API_KEY
        self.base_url = "https://api.deepseek.com"
        self.model_name = "deepseek-v4-pro"
        
        # 对话管理配置
        self.max_steps = 10  #单次ai最多执行步骤
        self.refresh_prompt_interval = 3 # 每N轮完整交互后将原始提示词重新发送给ai
        self.max_recent_messages = 6  # 压缩上下文时保留最近多少条消息
        self.max_memory_chars = 4000  # 长期记忆最大字符数
        self.max_episode_chars = 2000  # 每日情景最大字符数
        self.memory_dir = str(Path(__file__).parent / "memory")  # 记忆文件目录
        self.compact_after_messages = 3  # 非系统消息超过此数时触发压缩
        self.llm_compression_max_tokens = 2000  # LLM 压缩输出 token 上限
        self.max_session_index_entries = 100  # 会话索引最多保留条数
        self.index_max_tokens = 1000           # 生成索引总结时最多输出多少 token
        self.recall_max_tokens = 3000          # 召回检索 LLM 调用的 max_tokens
        self.recall_model = "deepseek-v4-pro"  # 召回检索用模型，空=复用主模型
        self.auto_index_session = True        # 是否自动生成会话索引
            
                
        # 工作目录配置
        #self.project_directory = ""  # 这个用处不大，传入时候会将这个目录下的文件作为系统提示词一部分
        
        # 调试配置
        self.show_system_messages = False  # 是否显示系统消息
        self.model_input_log_path = str(Path(__file__).with_name("model_input.jsonl"))
        self.token_usage_log_path = str(Path(__file__).with_name("token_usage.jsonl"))
