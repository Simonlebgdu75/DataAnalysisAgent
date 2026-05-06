from .state import AgentState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from .nodes import (
    agent_node,
    cg_interrupt,
    cg_research,
    prepare_initial_context,
    search_hybrid_tool,
    summarize_context,
)


def _entry_router_node(_state: AgentState) -> dict:
    return {}


builder = StateGraph(AgentState)
builder.add_node("entry_router", _entry_router_node)
builder.add_node("prepare_initial_context", prepare_initial_context)
builder.add_node("cg_research", cg_research)
builder.add_node("cg_interrupt", cg_interrupt)
builder.add_node("summarize_context", summarize_context)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode([search_hybrid_tool]))

builder.add_edge(START, "entry_router")


def _route_entry(state: AgentState) -> str:
    messages = state.get("messages", [])
    if (
        len(messages) == 1
        and isinstance(messages[0], HumanMessage)
        and not state.get("shortlist")
    ):
        if state.get("_cg_structured") and _cg_has_user_answer(state):
            return "summarize_context"
        return "prepare_initial_context"
    return "agent"


def _cg_has_user_answer(state: AgentState) -> bool:
    cg_messages = state.get("_cg_messages", []) or []
    return len(cg_messages) >= 3 and isinstance(cg_messages[-1], HumanMessage)


builder.add_conditional_edges(
    "entry_router",
    _route_entry,
    ["prepare_initial_context", "summarize_context", "agent"],
)
builder.add_edge("prepare_initial_context", "cg_research")
builder.add_edge("cg_research", "cg_interrupt")
builder.add_edge("cg_interrupt", "summarize_context")
builder.add_edge("summarize_context", "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    },
)
builder.add_edge("tools", "agent")
graph = builder.compile()
