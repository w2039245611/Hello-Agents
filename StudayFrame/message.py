"""消息系统"""
from datetime import datetime
from typing import Dict, Any, Literal

from pydantic import BaseModel, Field

# 定义消息角色的类型，限制其取值
MessageRole = Literal["user","assistant","system","tool"]


class Message(BaseModel):
    """消息类"""

    content:str
    role:MessageRole
    timestamp:datetime=Field(default_factory=datetime.now)
    metadata: Dict[str,Any] = Field(default_factory=dict)

    def to_dict(self)->Dict[str,Any]:
        """
        转换为字典格式(openAi API格式)
        :return:
        """

        return {
            "role":self.role,
            "content":self.content
        }

    def __str__(self) -> str:
        return f"[{self.role}]{self.content}"

