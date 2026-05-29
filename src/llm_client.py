"""LangChain chat model factory for PersonalGM."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def get_chat_model(temperature: float = 0.1) -> BaseChatModel:
    """Return a LangChain chat model configured from env. Used for streaming."""

    load_dotenv()
    provider = os.getenv("CHESS_LLM_PROVIDER", "gemini").strip().lower()
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY. Add it to your .env file.")

        model = os.getenv("CHESS_LLM_MODEL", "gemini-2.5-flash")

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
        )

    if provider == "litellm":
        model = os.getenv("CHESS_LLM_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE") or os.getenv("LITELLM_API_BASE")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY or LITELLM_API_KEY for LiteLLM.")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    raise ValueError("Unsupported CHESS_LLM_PROVIDER. Use gemini or litellm.")


def call_llm(messages: list[dict], temperature: float = 0.2) -> str:
    """One-shot blocking completion for OpenAI-style message dictionaries."""

    converted = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))

    response = get_chat_model(temperature=temperature).invoke(converted)
    content = response.content
    if isinstance(content, str):
        return content
    return str(content)
