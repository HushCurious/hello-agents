from hello_agents import SimpleAgent, HelloAgentsLLM
from dotenv import load_dotenv

# 如果当前目录存在 .env，就自动读取
# 我们也已经在终端中加载了 Chapter 4 的配置
load_dotenv()

# 创建 LLM 客户端
llm = HelloAgentsLLM()

# 创建一个最简单的 Agent
agent = SimpleAgent(
    name="AI助手",
    llm=llm,
    system_prompt="你是一个友好、简洁的 AI 助手。"
)

# 第一次对话
response = agent.run("你好！请用三句话介绍一下你自己。")
print("\n第一次回答：")
print(response)

# 第二次对话，用于观察历史消息
response = agent.run("我刚才让你做了什么？")
print("\n第二次回答：")
print(response)

# 查看历史记录
print(f"\n历史消息数：{len(agent.get_history())}")