from typing import Optional, List, Dict, Any, Union, Literal

import time

import shortuuid
from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    model: str = "Meta-Llama-3-8B-Instruct"  # 模型名称
    messages: Union[str, List[Dict[str, str]]] = []  # 消息列表或单条消息
    temperature: Optional[float] = 0.2  # 控制生成文本的多样性
    top_p: Optional[float] = 0.9  # nucleus sampling 的概率阈值
    top_k: Optional[int] = 40  # top-k 采样
    n: Optional[int] = 1  # 生成的响应数量
    max_tokens: Optional[int] = 512  # 生成的最大 token 数
    num_beams: Optional[int] = 1  # beam search 的束数
    stop: Optional[Union[str, List[str]]] = None  # 停止生成的标记
    stream: Optional[bool] = False  # 是否流式输出
    repetition_penalty: Optional[float] = 1.1  # 重复惩罚
    user: Optional[str] = None  # 用户标识
    do_sample: Optional[bool] = True  # 是否进行采样生成


class ChatMessage(BaseModel):
    role: str
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "Meta-Llama-3-8B-Instruct"
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[Any]]
    user: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str = "Meta-Llama-3-8B-Instruct"


class CompletionRequest(BaseModel):
    prompt: Union[str, List[Any]]
    temperature: Optional[float] = 0.2
    n: Optional[int] = 1
    max_tokens: Optional[int] = 512
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    num_beams: Optional[int] = 1
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.1
    user: Optional[str] = None
    do_sample: Optional[bool] = True


class CompletionResponseChoice(BaseModel):
    index: int
    text: str


class CompletionResponse(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: Optional[str] = "text_completion"
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    model: Optional[str] = "Meta-Llama-3-8B-Instruct"
    choices: List[CompletionResponseChoice]
