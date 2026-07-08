from agent import ReactAgent
from agent_config import AgentConfig
def main():
    """主函数"""
    # 创建自定义配置
    config = AgentConfig()
    config.max_steps = 10
    config.refresh_prompt_interval = 3
    config.show_system_messages = False  # 显示中间信息，如果不需要可以设置为False
    
    # 创建并运行Agent
    agent = ReactAgent(config)
    #print(agent.tool_manager.tools)
    agent.run()


if __name__ == "__main__":
    main()