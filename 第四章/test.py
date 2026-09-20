"""
ReAct 智能体单文件实现
包含：LLM 客户端、搜索工具、工具执行器、ReActAgent
运行前请确保 .env 文件中已配置：
  - SERPAPI_API_KEY
  - LLM_API_KEY / LLM_MODEL（可选 LLM_BASE_URL）
"""

import os
import re
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from serpapi import SerpApiClient

load_dotenv()


# ============================================================
# 1. LLM 客户端
# ============================================================
class HelloAgentsLLM:
    """
    一个简单的 LLM 客户端封装，兼容 OpenAI 接口。
    可通过 .env 配置 base_url / model / api_key。
    """

    def __init__(self, model: str = None, api_key: str = None, base_url: str = None):
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")

        if not self.api_key:
            raise ValueError("未找到 LLM_API_KEY，请在 .env 中配置。")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def think(self, messages: list, temperature: float = 0.0) -> str:
        """调用 LLM 并返回文本响应。"""
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            print("✅ 大语言模型响应成功:")
            print(content)
            return content
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            return ""


# ============================================================
# 2. 搜索工具
# ============================================================
def search(query: str) -> str:
    """
    一个基于 SerpApi 的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",      # 国家代码
            "hl": "zh-cn",   # 语言代码
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


# ============================================================
# 3. 工具执行器
# ============================================================
class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """向工具箱中注册一个新工具。"""
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """根据名称获取一个工具的执行函数。"""
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """获取所有可用工具的格式化描述字符串。"""
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])


# ============================================================
# 4. ReAct 提示词模板
# ============================================================
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


# ============================================================
# 5. ReActAgent
# ============================================================
class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """运行 ReAct 智能体来回答一个问题。"""
        self.history = []  # 每次运行时重置历史记录（工作记忆）
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str,
            )

            # 2. 调用 LLM 进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误:LLM 未能返回有效响应。")
                break

            # 3. 解析 LLM 的输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"🤔 思考: {thought}")

            if not action:
                print("警告:未能解析出有效的 Action，流程终止。")
                break

            # 4. 执行 Action
            if action.startswith("Finish"):
                final_answer = re.match(r"Finish\[(.*)\]", action, re.DOTALL).group(1)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                print(f"⚠️ 无法解析的 Action 格式: {action}")
                self.history.append(f"Action: {action}")
                self.history.append(
                    "Observation: 错误：Action 格式无效，请使用 `工具名[输入]` 或 `Finish[答案]`。"
                )
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input)  # 调用真实工具

            print(f"👀 观察: {observation}")

            # 5. 将本轮的 Action 和 Observation 添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        """解析 LLM 的输出，提取 Thought 和 Action。"""
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析 Action 字符串，提取工具名称和输入。"""
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None


# ============================================================
# 6. 主入口
# ============================================================
if __name__ == "__main__":
    # 1. 初始化 LLM 客户端
    llm_client = HelloAgentsLLM()

    # 2. 初始化工具执行器并注册搜索工具
    toolExecutor = ToolExecutor()
    search_description = (
        "一个网页搜索引擎。当你需要回答关于时事、事实以及"
        "在你的知识库中找不到的信息时，应使用此工具。"
    )
    toolExecutor.registerTool("Search", search_description, search)

    # 3. 打印可用工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 创建 ReAct 智能体
    agent = ReActAgent(llm_client=llm_client, tool_executor=toolExecutor, max_steps=5)

    # 5. 运行
    question = "华为最新手机型号及主要卖点是什么？"
    print(f"\n=== 问题: {question} ===")
    answer = agent.run(question)

    print("\n=== 最终答案 ===")
    print(answer)