from __future__ import annotations

import json
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings

from pe_deal_graph.config import EMBEDDING_DIMS, EMBEDDING_MODEL, OPENAI_API_KEY

_client: AsyncOpenAI | None = None
TextFormatT = TypeVar("TextFormatT", bound=BaseModel)

_embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    dimensions=EMBEDDING_DIMS,
    api_key=OPENAI_API_KEY or None,
)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY or None)
    return _client


def llmParameters(
    websearch: bool = True,
    context_size: str = "medium",
    verbosity: str = "medium",
    reasoning_effort: str = "medium",
    model: str = "gpt-5.4",
) -> dict[str, Any]:
    reasoning = {
        "effort": reasoning_effort,
        "summary": "auto",
    }
    web_tool = {
        "type": "web_search",
        "search_context_size": context_size,
    }
    if websearch:
        return {
            "reasoning": reasoning,
            "tools": [web_tool],
            "verbosity": verbosity,
            "model": model,
        }
    return {
        "reasoning": reasoning,
        "verbosity": verbosity,
        "model": model,
    }


async def chat_json(
    system: str,
    user: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Call GPT with JSON response format and return parsed dict."""
    client = _get_client()
    payload: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    else:
        payload["temperature"] = temperature
    resp = await client.chat.completions.create(**payload)
    return json.loads(resp.choices[0].message.content)


async def web_search_reasoning_parse(
    *,
    system: str,
    user: str,
    text_format: type[TextFormatT],
    model: str = "gpt-5.2",
    search_context_size: str = "high",
    reasoning_effort: str = "medium",
) -> TextFormatT:
    """Single-call Responses API with built-in web search + structured output."""
    client = _get_client()
    resp = await client.responses.parse(
        model=model,
        instructions=system,
        input=user,
        tools=[{"type": "web_search_preview", "search_context_size": search_context_size}],
        tool_choice="required",
        reasoning={"effort": reasoning_effort},
        text_format=text_format,
    )
    parsed = resp.output_parsed
    if parsed is None:
        raise ValueError("Structured web-search response did not contain parsed output")
    return parsed


async def embed_query(text: str) -> list[float]:
    """Embed a single search query using LangChain's query embedding path."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    return await _embeddings.aembed_query(cleaned)
