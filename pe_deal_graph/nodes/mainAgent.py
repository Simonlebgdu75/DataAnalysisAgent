from ..state import AgentState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, convert_to_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from ..config import SUPPORTED_COUNTRY_CODES
from ..llm.client import llmParameters
from ..llm.prompt import MAIN_AGENT_SYSTEM_PROMPT, MAIN_AGENT_TOOL_RESPONSE_SYSTEM_PROMPT
from .rerank import rerank_results
from .search import search_hybrid as run_hybrid_search
from .utils import build_hybrid_search_summary

SUPPORTED_COUNTRY_CODES_TEXT = ", ".join(SUPPORTED_COUNTRY_CODES)


class HybridSearchToolInput(BaseModel):
    search_query: str = Field(
        description="Compact English semantic retrieval query for vector search."
    )
    bm25_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Four to eight short English BM25 keywords or phrases likely to appear "
            "verbatim in company names, one-liners, product names, or descriptions."
        ),
    )
    country_code: str = Field(
        default="",
        description=(
            "Optional single HQ country filter as an ISO 3166-1 alpha-2 code. "
            f"Use it only if the user explicitly asks for a country restriction. Supported values: {SUPPORTED_COUNTRY_CODES_TEXT}."
        ),
    )
    rerank_context: str = Field(
        default="",
        description=(
            "Short English note for the reranker explaining what makes a company in-scope, "
            "which product, customer, or market signals matter most, and what should be treated "
            "as too broad, too adjacent, or out-of-scope."
        ),
    )


@tool("search_hybrid", args_schema=HybridSearchToolInput)
async def search_hybrid_tool(
    search_query: str,
    bm25_keywords: list[str],
    country_code: str = "",
    rerank_context: str = "",
    runtime: ToolRuntime = None,
) -> Command:
    """Run hybrid search, update the graph state, and append a ToolMessage."""
    normalized_country_code = _normalize_country_code(country_code) or ""
    summary, results = await _run_search_hybrid_payload(
        search_query=search_query,
        bm25_keywords=bm25_keywords,
        country_code=normalized_country_code,
        rerank_context=rerank_context,
    )
    state = runtime.state if runtime is not None and isinstance(runtime.state, dict) else {}
    messages = convert_to_messages(state.get("messages", []))
    context = state.get("context") or {}
    context_user_description = str(context.get("userDescription", "") or "") if isinstance(context, dict) else ""
    merged_context = dict(context) if isinstance(context, dict) else {}
    merged_context.update(
        {
            "latest_user_message": _latest_human_text(messages) or "",
            "search_query": search_query,
            "bm25_keywords": bm25_keywords,
            "country_code": normalized_country_code,
            "rerank_context": rerank_context,
            "tool_summary": summary,
        }
    )

    tool_message = ToolMessage(
        content=summary,
        artifact=results,
        tool_call_id=str(runtime.tool_call_id or "search_hybrid") if runtime is not None else "search_hybrid",
        name="search_hybrid",
    )
    return Command(
        update={
            "messages": [tool_message],
            "context": merged_context,
            "question": state.get("question") or context_user_description or search_query,
            "search_query": search_query,
            "bm25_keywords": bm25_keywords,
            "country_code": normalized_country_code,
            "rerank_mode": "balanced",
            "tool_summary": summary,
            "shortlist": results.get("shortlist", []),
            "target_shortlist": results.get("target_shortlist", []),
            "deals_shortlist": results.get("deals_shortlist", []),
            "acquirers_shortlist": results.get("acquirers_shortlist", []),
            "errors": results.get("errors", []),
            "progress": {
                "step": "search_hybrid",
                "status": "complete",
                "message": "search_hybrid executed and state updated.",
                "target_count": len(results.get("target_shortlist", [])),
                "final_shortlist_count": len(results.get("shortlist", [])),
                "deal_count": len(results.get("deals_shortlist", [])),
                "buyer_count": len(results.get("acquirers_shortlist", [])),
            },
        }
    )


async def _run_search_hybrid_payload(
    *,
    search_query: str,
    bm25_keywords: list[str],
    country_code: str = "",
    rerank_context: str = "",
) -> tuple[str, dict]:
    raw_results = await run_hybrid_search(search_query, bm25_keywords, country_code=country_code)
    results = await rerank_results(
        shortlist=raw_results.get("shortlist", []),
        search_context=search_query,
        rerank_context=rerank_context,
        rerank_mode="balanced",
    )
    return build_hybrid_search_summary(results), results


async def prepare_initial_context(state: AgentState) -> dict:
    """Prepare the first-turn context-generator path from the latest user message."""
    messages = convert_to_messages(state.get("messages", []))
    query = _latest_human_text(messages).strip() or str(state.get("question") or "")

    from .lookup import lookup as _lookup_fn

    lookup_result = await _lookup_fn(
        {
            "company_name": query,
            "shortlist": state.get("shortlist", []),
        }
    )
    company_profile = lookup_result.get("lookup_result")

    out = {
        "question": query,
        "_cg_messages": [],
        "_cg_structured": None,
        "_agent_context": "",
        "lookup_result": None,
    }
    if company_profile:
        out["lookup_result"] = company_profile
    return out


async def summarize_context(state: AgentState) -> dict:
    """Build the hidden context-generator brief passed to the main agent."""
    cg_messages = state.get("_cg_messages", [])
    cg_structured = state.get("_cg_structured")

    raw_context = _format_cg_context(cg_messages)
    transcript = raw_context or "(empty)"
    payload = f"Context generator exchange (verbatim):\n\n{transcript}"

    structured_context = _format_cg_structured_context(cg_structured)
    if structured_context:
        payload += f"\n\nInternal market context:\n\n{structured_context}"

    return {"_agent_context": payload}


async def agent_node(state: AgentState) -> dict:
    messages = convert_to_messages(state.get("messages", []))
    latest_user = _latest_human_text(messages)
    context = state.get("context") or {}
    context_user_description = str(context.get("userDescription", "") or "") if isinstance(context, dict) else ""
    state_context_prompt = _build_state_context_prompt(state)
    agent_context = str(state.get("_agent_context") or "").strip()
    user_lang = str(state.get("user_language") or "").strip() or _detect_user_language(messages)

    if _is_post_search_tool_result(messages):
        parameters = llmParameters(websearch=False)
        llm = ChatOpenAI(
            model=parameters["model"],
            reasoning=parameters["reasoning"],
            verbosity=parameters["verbosity"],
            use_responses_api=True,
        )
        answer_input: list = [("system", MAIN_AGENT_TOOL_RESPONSE_SYSTEM_PROMPT)]
        if user_lang:
            answer_input.append(("system", f"DETECTED USER LANGUAGE: {user_lang}. You MUST answer in {user_lang}."))
        if state_context_prompt:
            answer_input.append(("system", state_context_prompt))
        answer_input.extend(messages)
        response = await llm.ainvoke(answer_input)
        return {
            "messages": [response],
            "_agent_context": "",
            "user_language": user_lang,
            "question": latest_user or context_user_description or str(state.get("question") or ""),
            "progress": {
                "step": "answer",
                "status": "complete",
                "mode": "post_tool_answer",
                "message": "Final answer generated from search_hybrid results.",
                "final_shortlist_count": len(state.get("shortlist", [])),
            },
        }

    parameters = llmParameters(websearch=True)
    llm = ChatOpenAI(
        model=parameters["model"],
        reasoning=parameters["reasoning"],
        verbosity=parameters["verbosity"],
        use_responses_api=True,
    )
    tools = [*parameters.get("tools", []), search_hybrid_tool]
    runnable = llm.bind_tools(
        tools,
        parallel_tool_calls=False,
        strict=True,
    )
    planner_input: list = [("system", MAIN_AGENT_SYSTEM_PROMPT)]
    if user_lang:
        planner_input.append(("system", f"DETECTED USER LANGUAGE: {user_lang}. You MUST answer in {user_lang}."))
    if agent_context:
        planner_input.append(("system", f"INTERNAL CONTEXT GENERATOR BRIEF:\n{agent_context}"))
    if state_context_prompt:
        planner_input.append(("system", state_context_prompt))
    planner_input.extend(messages)
    response = await runnable.ainvoke(planner_input)
    if not response.tool_calls:
        return {
            "messages": [response],
            "_agent_context": "",
            "user_language": user_lang,
            "question": latest_user or context_user_description,
            "progress": {
                "step": "answer",
                "status": "complete",
                "mode": "answer_only",
                "message": "Answered from the current conversation state.",
            },
            "errors": [],
        }

    return {
        "messages": [response],
        "_agent_context": "",
        "user_language": user_lang,
        "question": latest_user or context_user_description,
        "progress": {
            "step": "agent",
            "status": "complete",
            "mode": "search",
            "message": "Agent requested tool execution.",
            "tool_calls": [call.get("name") for call in (response.tool_calls or [])],
        },
        "errors": [],
    }


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if message.type == "human":
            return _message_text(message)
    return ""


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _is_post_search_tool_result(messages: list[BaseMessage]) -> bool:
    if not messages:
        return False
    last_message = messages[-1]
    return isinstance(last_message, ToolMessage) and last_message.name == "search_hybrid"


def _build_state_context_prompt(state: AgentState | dict) -> str:
    context = state.get("context") if isinstance(state, dict) else None
    search_query = str(state.get("search_query") or "").strip() if isinstance(state, dict) else ""
    bm25_keywords = state.get("bm25_keywords") or [] if isinstance(state, dict) else []
    country_code = str(state.get("country_code") or "").strip() if isinstance(state, dict) else ""
    tool_summary = str(state.get("tool_summary") or "").strip() if isinstance(state, dict) else ""
    shortlist = state.get("shortlist") or [] if isinstance(state, dict) else []
    target_shortlist = state.get("target_shortlist") or [] if isinstance(state, dict) else []
    deals_shortlist = state.get("deals_shortlist") or [] if isinstance(state, dict) else []
    acquirers_shortlist = state.get("acquirers_shortlist") or [] if isinstance(state, dict) else []
    progress = state.get("progress") or {} if isinstance(state, dict) else {}

    if not any([context, search_query, bm25_keywords, country_code, tool_summary, shortlist, target_shortlist, deals_shortlist, acquirers_shortlist, progress]):
        return ""

    lines = [
        "Current workflow state:",
        f"- Current step: {progress.get('step') or '(none)'}",
        f"- Current search query: {search_query or '(none)'}",
        f"- Current BM25 keywords: {', '.join(str(kw) for kw in bm25_keywords) if bm25_keywords else '(none)'}",
        f"- Current country filter: {country_code or '(none)'}",
        f"- Current targets count: {len(target_shortlist) or len(shortlist)}",
        f"- Current final shortlist count: {len(shortlist)}",
        f"- Current deals count: {len(deals_shortlist)}",
        f"- Current buyers count: {len(acquirers_shortlist)}",
    ]
    if tool_summary:
        lines.extend(["", "Current result summary:", tool_summary])
    elif context:
        lines.extend(["", "Current context:", str(context)])
    return "\n".join(lines)


def _normalize_country_code(value) -> str | None:
    if isinstance(value, str):
        codes = [part.strip().upper() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        codes = []
        for item in value:
            if not isinstance(item, str):
                return None
            code = item.strip().upper()
            if code:
                codes.append(code)
    else:
        return None

    normalized: list[str] = []
    seen: set[str] = set()
    for code in codes:
        if code not in SUPPORTED_COUNTRY_CODES:
            return None
        if code in seen:
            continue
        seen.add(code)
        normalized.append(code)

    if len(normalized) != 1:
        return None
    return normalized[0]


_LANG_NAMES = {
    "en": "English",
    "fr": "French",
    "it": "Italian",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "nl": "Dutch",
}


def _detect_user_language(messages: list[BaseMessage]) -> str:
    text = _latest_human_text(messages).strip()
    if not text or len(text) < 5:
        return ""
    try:
        from fast_langdetect import detect as ft_detect

        result = ft_detect(text)
        first = result[0] if isinstance(result, list) else result
        lang_code = first.get("lang", "") if isinstance(first, dict) else ""
        return _LANG_NAMES.get(lang_code, lang_code.upper() if lang_code else "")
    except Exception:
        return ""


def _format_cg_context(cg_messages: list) -> str:
    lines: list[str] = []
    for msg in cg_messages:
        content = getattr(msg, "content", "")
        if not content:
            continue
        if isinstance(msg, HumanMessage):
            role = "USER"
        elif isinstance(msg, AIMessage):
            role = "ASSISTANT"
        else:
            role = "MESSAGE"
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def _format_cg_structured_context(cg_structured: dict | None) -> str:
    if not isinstance(cg_structured, dict) or not cg_structured:
        return ""

    company_card = cg_structured.get("company_card") or {}
    lines = [
        "Target market summary:",
        str(cg_structured.get("target_market_summary", "")).strip() or "Not provided.",
        "",
        "Market summary:",
        str(cg_structured.get("market_summary", "")).strip() or "Not provided.",
        "",
        "Company card:",
        f"- Core product: {str(company_card.get('core_product', '')).strip() or 'Not provided.'}",
        f"- Upstream: {str(company_card.get('upstream', '')).strip() or 'Not provided.'}",
        f"- Core: {str(company_card.get('core', '')).strip() or 'Not provided.'}",
        f"- Downstream: {str(company_card.get('downstream', '')).strip() or 'Not provided.'}",
    ]
    return "\n".join(lines).strip()
