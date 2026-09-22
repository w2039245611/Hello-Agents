"""Agent基类"""


from abc import ABC, abstractmethod
from typing import Optional

from .config import Config
from .llm import HelloAgentsLLM
from .message import Message


class Agent(ABC):
    """Agent基类"""


    def __init__(
            self,
            name:str,
            llm:HelloAgentsLLM,
            sys_prompt:Optional[str] = None,
            config:Optional[Config] = None
    ):
        self.name = name
        self.llm = llm
        self.sys_prompt = sys_prompt
        self.config = config or Config()
        self._history:list[Message]=[]

    @abstractmethod
    def run(self,input_text:str,**kwargs) -> str:
        """运行Agent"""
        ...

    def add_messages(self,message:Message):
        """添加消息到历史记录"""
        self._history.append(message)


    def clear_history(self):
        """清空历史"""
        self._history.clear()


    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name},provide={self.llm.provider})"



