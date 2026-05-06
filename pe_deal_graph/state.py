from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def _replace(left: Any, right: Any) -> Any:
    if right is None:
        return left
    return right


def _replace_list(left: list, right: list) -> list:
    if right is None:
        return left
    return right


def _reset_messages(
    left: list[AnyMessage] | None, right: list[AnyMessage]
) -> list[AnyMessage]:
    """Like add_messages but [] means reset (clear all)."""
    if not right and isinstance(right, list):
        return []
    left = left or []
    return add_messages(left, right)


class AgentState(TypedDict, total=False):
    # ── Conversation (agent) ──────────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]
    context: Annotated[dict | None, _replace]
    errors: Annotated[list[str], _replace_list]

    # ── Search pipeline ───────────────────────────────────────────
    question: Annotated[str, _replace]
    search_query: Annotated[str, _replace]
    bm25_keywords: Annotated[list[str], _replace_list]
    country_code: Annotated[str, _replace]
    rerank_mode: Annotated[str, _replace]
    tool_summary: Annotated[str, _replace]
    progress: Annotated[dict | None, _replace]

    # ── Shortlist (raw data → table UI) ───────────────────────────
    shortlist: Annotated[list[dict], _replace_list]
    target_shortlist: Annotated[list[dict], _replace_list]
    deals_shortlist: Annotated[list[dict], _replace_list]
    acquirers_shortlist: Annotated[list[dict], _replace_list]

    # ── Context generator (private state) ─────────────────────────
    _cg_messages: Annotated[list[AnyMessage], _reset_messages]
    _cg_structured: Annotated[dict | None, _replace]
    _agent_context: Annotated[str, _replace]
    lookup_result: Annotated[dict | None, _replace]
    user_language: Annotated[str, _replace]
