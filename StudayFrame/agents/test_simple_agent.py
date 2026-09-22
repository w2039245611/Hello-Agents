from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ToolRegistry, CalculatorTool

from StudayFrame.agents.my_simple_agent import MySimpleAgent

load_dotenv()

llm = HelloAgentsLLM()
#
# basic_agent =MySimpleAgent(
#     name="基础助手",
#     llm= llm,
#     system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。"
# )
#
# response = basic_agent.run("你好，请介绍一下自己")
# print(f"基础对话响应：{response}\n")



# 测试2:带工具的Agent
print("=== 测试2:工具增强对话 ===")
tool_registry = ToolRegistry()
calculator = CalculatorTool()

tool_registry.register_tool(calculator)


enhanced_agent = MySimpleAgent(
    name = "增强助手",
    llm=llm,
    system_prompt="你是一个智能助手，可以使用工具来帮助用户。",
    tool_registry =tool_registry,
    enable_tool_calling= True
)

response2 = enhanced_agent.run(input_text="请帮我计算 15*8-32")
print(f"工具增强响应；{response2.str}\n")



