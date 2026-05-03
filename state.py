from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


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
    errors: Annotated[list[str], _replace_list]

    # ── Shortlist (raw data → table UI) ───────────────────────────
    shortlist: Annotated[list[dict], _replace_list]
    shortlist_history: Annotated[list[dict], _replace_list]

    # ── Tool call routing (transient) ─────────────────────────────
    _tool_call_id: Annotated[str, _replace]  # for creating ToolMessage
    _agent_context: Annotated[str, _replace]

    # ── Search pipeline fields (shared with subgraph) ─────────────
    intent: Annotated[str, _replace]
    question: Annotated[str, _replace]
    rerank_mode: Annotated[str, _replace]
    search_query: Annotated[str, _replace]
    bm25_keywords: Annotated[list[str], _replace_list]
    filters: Annotated[dict | None, _replace]
    skip_rerank: Annotated[bool, _replace]
    _embeddings: Annotated[list, _replace_list]

    # ── Context generator (sub-agent internal messages) ────────────
    _cg_messages: Annotated[list[AnyMessage], _reset_messages]
    _cg_structured: Annotated[dict | None, _replace]

    # ── Geo resolution (raw text → resolved by geo_resolver node) ──
    geo_query: Annotated[str | None, _replace]

    # ── Language ───────────────────────────────────────────────────
    user_language: Annotated[str, _replace]

    # ── Lookup ────────────────────────────────────────────────────
    company_name: Annotated[str, _replace]
    lookup_result: Annotated[dict | None, _replace]
