from __future__ import annotations

import langsmith as ls
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from pe_qa_graph.nodes.agent import (
    agent_node,
    prepare_initial_context,
    prepare_search,
    summarize_context,
    summarize_search,
)
from pe_qa_graph.nodes.context_generator import cg_interrupt, cg_research
from pe_qa_graph.nodes.geo import geo_resolver
from pe_qa_graph.nodes.rerank import rerank
from pe_qa_graph.nodes.search import search_embed, search_hybrid
from pe_qa_graph.state import AgentState

# ── Search pipeline (subgraph — visible in Studio) ───────────────────

search_builder = StateGraph(AgentState)


def _search_fan_out(_state: AgentState) -> dict:
    """No-op node for parallel fan-out to search_embed + geo_resolver."""
    return {}


search_builder.add_node("fan_out", _search_fan_out)
search_builder.add_node("search_embed", search_embed)
search_builder.add_node("geo_resolver", geo_resolver)
search_builder.add_node("search_hybrid", search_hybrid)
search_builder.add_node("rerank", rerank)

# Fan-out: search_embed and geo_resolver run in parallel
search_builder.set_entry_point("fan_out")
search_builder.add_edge("fan_out", "search_embed")
search_builder.add_edge("fan_out", "geo_resolver")
# Fan-in: search_hybrid waits for both
search_builder.add_edge("search_embed", "search_hybrid")
search_builder.add_edge("geo_resolver", "search_hybrid")
search_builder.add_edge("search_hybrid", "rerank")
search_builder.add_edge("rerank", END)

search_pipeline = search_builder.compile()

# ── Main graph (agent + tool paths) ──────────────────────────────────

builder = StateGraph(AgentState)


# Nodes
builder.add_node("agent", agent_node)
builder.add_node("prepare_search", prepare_search)
builder.add_node("search_pipeline", search_pipeline)
builder.add_node("summarize_search", summarize_search)
builder.add_node("prepare_initial_context", prepare_initial_context)
builder.add_node("cg_research", cg_research)
builder.add_node("cg_interrupt", cg_interrupt)
builder.add_node("summarize_context", summarize_context)

# Entry point – route first turn to context generator, follow-ups to agent
def _entry_router_node(_state: AgentState, config: RunnableConfig) -> dict:
    # Propagate config.metadata to the root LangSmith trace
    # config["metadata"] is forwarded by LangGraph Server from the HTTP body
    metadata = (config or {}).get("metadata", {})
    if metadata:
        try:
            rt = ls.get_current_run_tree()
            while rt and rt.parent_run:
                rt = rt.parent_run
            if rt:
                rt.extra = rt.extra or {}
                rt.extra.setdefault("metadata", {}).update(metadata)
        except Exception:
            pass
    return {}


builder.add_node("entry_router", _entry_router_node)
builder.set_entry_point("entry_router")


def _route_entry(state: AgentState) -> str:
    """Force first user turn to context generator path (skip agent)."""
    messages = state.get("messages", [])
    if (
        len(messages) == 1
        and isinstance(messages[0], HumanMessage)
        and not state.get("shortlist")
    ):
        return "prepare_initial_context"
    return "agent"


builder.add_conditional_edges(
    "entry_router",
    _route_entry,
    ["prepare_initial_context", "agent"],
)


# ── Routing after agent ──────────────────────────────────────────────

def _route_after_agent(state: AgentState) -> str:
    """Route based on the agent's tool calls (or END if no tool call)."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last_msg = messages[-1]

    tool_calls = getattr(last_msg, "tool_calls", None)
    if not tool_calls:
        return END

    tool_name = tool_calls[0]["name"]
    routes = {
        "search_companies": "prepare_search",
    }
    return routes.get(tool_name, END)


builder.add_conditional_edges(
    "agent",
    _route_after_agent,
    ["prepare_search", END],
)

# Search path: prepare → subgraph (search → rerank) → summarize → agent
builder.add_edge("prepare_search", "search_pipeline")
builder.add_edge("search_pipeline", "summarize_search")
builder.add_edge("summarize_search", "agent")

# Context generator path (first turn only): prepare_initial_context → cg_research → cg_interrupt → summarize → agent
builder.add_edge("prepare_initial_context", "cg_research")
builder.add_edge("cg_research", "cg_interrupt")
builder.add_edge("cg_interrupt", "summarize_context")
builder.add_edge("summarize_context", "agent")

graph = builder.compile()
