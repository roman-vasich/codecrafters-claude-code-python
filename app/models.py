from enum import Enum
from typing import Protocol

from pydantic import BaseModel


class Providers(Enum):
    OpenAI = "openai"
    Anthropic = "anthropic"


class Provider(Protocol):
    """A thin wrapper for Provider API that exposes abstructs helpers for the application"""

    @staticmethod
    def tool_schema(model: BaseModel, *, name: str, description: str) -> dict: ...


class AnthropicProvider:
    @staticmethod
    def tool_schema(model: type[BaseModel], *, name: str, description: str) -> dict:
        return {
            "name": name,
            "description": description,
            "input_schema": model.model_json_schema(),
        }


class OpenAIProvider:
    @staticmethod
    def tool_schema(model: type[BaseModel], *, name: str, description: str) -> dict:
        from openai import pydantic_function_tool

        return pydantic_function_tool(model, name=name, description=description)
