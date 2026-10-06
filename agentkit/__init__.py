"""agentkit: a small, provider-agnostic tool-calling agent core.

Public surface:
    Agent, Answer          the loop and its structured final answer
    ToolRegistry           register plain Python functions as tools
    from_env               build an LLM (or fallback chain) from environment variables
"""

from .agent import Agent, RunResult
from .llm import LLMError, from_env
from .schemas import Answer
from .tools import ImageData, ToolError, ToolRegistry, ToolResult, image_from_bytes, load_image

__all__ = [
    "Agent", "Answer", "ImageData", "LLMError", "RunResult", "ToolError", "ToolRegistry", "ToolResult",
    "from_env", "image_from_bytes", "load_image",
]
