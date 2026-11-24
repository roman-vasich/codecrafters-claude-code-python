import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.models import AnthropicProvider, OpenAIProvider, Provider, Providers

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL",
                     default="https://openrouter.ai/api/v1")
BASE_MODEL = os.getenv("OPENROUTER_BASE_MODEL",
                       default="anthropic/claude-haiku-4.5")


class ReadQuery(BaseModel):
    file_path: str = Field(description="The path to the file to read")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    provider: Provider | None = None
    match BASE_MODEL:
        # OpenRouter uses OpenAI compatible API
        case model if "/" in model:
            provider = OpenAIProvider
        case Providers.Anthropic.value:
            provider = AnthropicProvider
        case _:
            raise ValueError(f"Unsuported model passed: {BASE_MODEL}")

    read = provider.tool_schema(
        ReadQuery, name="Read", description="Read and return the contents of a file"
    )

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    chat = await client.chat.completions.create(
        model=BASE_MODEL,
        messages=[{"role": "user", "content": args.p}],
        tools=[read],
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

    # TODO: Uncomment the following line to pass the first stage
    print(chat.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
