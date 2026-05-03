from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from pe_qa_graph.config import OPENAI_API_KEY, SUPPORTED_COUNTRY_CODES
from pe_qa_graph.state import AgentState
from pe_qa_graph.stream import emit_custom, emit_error, emit_metric, emit_phase

# ── Tool definitions for OpenAI function calling ─────────────────────

TOOLS = [
    {
        "type": "web_search",
    },
    {
        "type": "function",
        "function": {
            "name": "search_companies",
            "description": (
                "Search the database using a structured search plan. "
                "Use this when the user wants to find companies or narrow an existing shortlist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_context": {
                        "type": "string",
                        "description": (
                            "Short but precise analyst brief describing the target company universe and exclusions. "
                            "This is used for reranking. The reranker will compare this brief against each company's "
                            "one-liner, products_keywords, and products/services, then decide whether the company should "
                            "be kept or discarded. Write it so the reranker can judge how relevant a company is for the user "
                            "based on the company's products/services and description. "
                            "Preserve the user's real intent, target, and exclusions."
                        ),
                    },
                    "semantic_query": {
                        "type": "string",
                        "description": (
                            "Compact ENGLISH semantic query for vector retrieval. "
                            "Use 8-16 high-signal noun phrases, not a long sentence. "
                            "If a concrete structured filter is active (`linkedin_employee_filter`, or "
                            "`is_pe_backed` set to `pe_backed`/`not_pe_backed`), do not restate that filter "
                            "inside this query. Never include employee-count or PE/LBO ownership terms here "
                            "when the corresponding structured filter is active."
                        ),
                    },
                    "bm25_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "4 to 8 HIGH-SIGNAL ENGLISH keywords or short phrases for lexical retrieval. "
                            "Use terms likely to appear verbatim on company websites. "
                            "If a concrete structured filter is active (`linkedin_employee_filter`, or "
                            "`is_pe_backed` set to `pe_backed`/`not_pe_backed`), do not restate that filter "
                            "in these keywords. Never include employee-count or PE/LBO ownership terms in "
                            "these keywords when the corresponding structured filter is active."
                        ),
                    },
                    "rerank_mode": {
                        "type": "string",
                        "enum": ["broad", "balanced", "strict"],
                        "description": (
                            "How selective the reranker should be. "
                            "Use `broad` when the user wants the widest or most exhaustive relevant list; "
                            "this still runs LLM reranking, but with a high-recall keep threshold. "
                            "Use `balanced` for normal sourcing requests. "
                            "Use `strict` when the user wants only a narrow, exact sub-segment or strong exclusions."
                        ),
                    },
                    "restrict_to_shortlist": {
                        "type": "boolean",
                        "description": (
                            "Choose this explicitly on every call based on the user's likely intent in context. "
                            "Use true when the request is best understood as refining, narrowing, excluding from, "
                            "or otherwise filtering the CURRENT SHORTLIST already loaded in the UI. "
                            "Use false when the request is better understood as starting a new search, "
                            "changing the target market or company anchor, restarting from scratch, "
                            "or refreshing the search universe more broadly."
                        ),
                        "default": True,
                    },
                    "linkedin_employee_filter": {
                        "type": "object",
                        "description": (
                            "Structured employee-count filter. "
                            "Use it only when the user explicitly asks for a LinkedIn employee count constraint."
                        ),
                        "properties": {
                            "operator": {
                                "type": "string",
                                "enum": ["eq", "gt", "gte", "lt", "lte"],
                                "description": "Comparison operator for the LinkedIn employee count.",
                            },
                            "value": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Target LinkedIn employee count.",
                            },
                        },
                        "required": ["operator", "value"],
                    },
                    "is_pe_backed": {
                        "type": "string",
                        "enum": ["not_specified", "pe_backed", "not_pe_backed"],
                        "default": "not_specified",
                        "description": (
                            "Structured PE-backed ownership filter. REQUIRED: you must pick one of the three values. "
                            "The default and neutral value is `not_specified` — use it whenever the user does NOT "
                            "explicitly mention PE-backed ownership. "
                            "`pe_backed` → user explicitly wants companies ALREADY under LBO / sponsor-backed / fund-owned. "
                            "`not_pe_backed` → user explicitly wants companies that are NOT PE-backed. "
                            "`not_specified` → user's request is neutral on ownership (e.g. 'SaaS companies in France', "
                            "'éditeurs logiciels bordelais'). "
                            "Only pick `pe_backed` or `not_pe_backed` when the user uses explicit ownership keywords "
                            "like: LBO, sous LBO, PE-backed, sponsor-backed, fund-owned, détenues par un fonds, "
                            "non PE-backed, non cotées. Do not infer from sector, geography, size, or any non-ownership signal."
                        ),
                    },
                    "geo_query": {
                        "type": "string",
                        "description": (
                            "The geographic constraint from the user's request, in their exact words. "
                            "A dedicated geo-resolver LLM will parse this into structured filters "
                            "(countries, cities with radius, regions). Just pass the user's geographic "
                            "intent as-is — do NOT try to resolve city names, regions, or country codes. "
                            "Examples: 'en France', 'à Paris et Marseille', 'dans l'ouest de la France', "
                            "'bordelaises', 'en Île-de-France', 'in northern Italy', 'banlieue parisienne'. "
                            "Leave empty if no geographic constraint is mentioned."
                        ),
                    },
                },
                "required": ["search_context", "restrict_to_shortlist", "is_pe_backed"],
            },
        },
    },
]

SUPPORTED_COUNTRY_CODES_TEXT = ", ".join(SUPPORTED_COUNTRY_CODES)

AGENT_SYSTEM_PROMPT = f"""\
You are a senior Private Equity analyst assistant. You help investment professionals \
find and analyze companies from a proprietary database of ~100,000 company profiles.

# LANGUAGE — CRITICAL
- Detect the language of the user's LATEST HumanMessage (not tool results, not internal context).
- ALWAYS respond in that language, regardless of the language used in tool results, database data, or internal context.
- If the user writes in English, you MUST respond in English even if the data or context is in French.
- If the user writes in French, respond in French.
- This rule overrides everything else.

# WORKFLOW
- The first user request is handled upstream by the context generator before you are called.
- Your tools are:
  - `web_search` to clarify what a company does or how its market should be defined
  - `search_companies` to search or filter the proprietary database
- Your job is to convert the conversation into the most precise possible company search.

# WHEN TO CALL web_search
Use `web_search` only when the available context is not precise enough to build a high-quality search brief.

Typical cases:
- The context-generator transcript is still vague about what the company actually sells
- The market boundary, sub-segment, value-chain position, or customer type is unclear
- The user asks for a market/company type that requires external clarification before searching

When you use `web_search`:
- Focus on understanding what the company does and the exact market relevant to the user's request
- Use both the context-generator transcript and the user's ask as guidance
- Keep the research targeted and pragmatic
- Do not stop after research: if the conversation is still about company discovery, you must still call `search_companies` in the same response

# WHEN TO CALL search_companies
Call exactly one `search_companies` tool when the conversation is about finding, narrowing, or refreshing a company list.

On every `search_companies` call, you must make an explicit and deliberate choice for `restrict_to_shortlist`.
Do not rely on the system to infer it for you.
Determine this choice from the user's likely intent in context, even when the user does not state it explicitly.

Use `search_companies(restrict_to_shortlist=false)` for:
- A new/global search
- A restart from scratch
- Any turn where there is no current shortlist
- The turn immediately following a context-generator transcript once you have enough precision
- The turn immediately following your own `web_search` if you needed extra clarification
- A request that changes the target market, target company type, company anchor, or overall sourcing objective

Use `search_companies(restrict_to_shortlist=true)` for:
- Narrowing the current shortlist
- Follow-up refinements such as sub-segment, geography, customer type, exclusions, LinkedIn employee count, or PE-backed status
- Requests such as "remove", "exclude", "keep only", "focus on", or "within these results"

# CONTEXT GENERATOR HANDOFF
If an `INTERNAL CONTEXT GENERATOR BRIEF` is present in the system prompt,
you must treat it as the primary search brief.

If the latest conversation message starts with `Context generator exchange (verbatim):`,
you must treat it as the primary search brief.

If it is already precise enough, immediately call `search_companies(restrict_to_shortlist=false)`.
If it is not precise enough to understand the company/market correctly, use `web_search` first,
then call `search_companies(restrict_to_shortlist=false)` using what you learned.

Do not summarize the transcript back to the user first.

# SHORTLIST FOLLOW-UPS
If CURRENT SHORTLIST is loaded, decide whether the latest user message is:
- a refinement of the existing shortlist -> call `search_companies(restrict_to_shortlist=true)`
- or a new sourcing request -> call `search_companies(restrict_to_shortlist=false)`

Make this decision from the user's actual intent, not from shortlist existence alone.
The user does not need to explicitly say "filter the shortlist" or "start a new search" for you to decide.
You should infer the most sensible behavior from the request itself.

Use `restrict_to_shortlist=true` when the user is clearly trying to narrow, filter, exclude from, rank, or segment the current results.
Also use `restrict_to_shortlist=true` when the latest request is best understood as a logical refinement of the companies already shown, even if the user phrases it briefly or implicitly.
Use `restrict_to_shortlist=false` when the user is asking for a different search universe, a different company set, a different company anchor, a restart, or a broader refresh that should not be constrained by the current shortlist.

If the intent is ambiguous, infer the most likely user goal from the latest message and the recent conversation.
Always set `restrict_to_shortlist` explicitly.

# SEARCH PLAN
When you call `search_companies`, you must provide a structured search plan.
Put that plan only in the tool call arguments.
Do not display the search plan, JSON, or any structured args to the user.
Do not echo `search_context`, `semantic_query`, `bm25_keywords`, filters, or raw tool payloads in the visible assistant message.

1. `search_context`
- Preserve the actual target, user goal, and exclusions.
- Use the conversation context naturally; do not just forward a vague user sentence.
- The reranker will read this brief and compare it against each company's `one_liner`, `products_keywords`, and `products/services`, then decide whether to keep or discard the company.
- Write it so the reranker can evaluate how relevant each company is for the user based on the company's products/services and description.
- Be explicit about what should count as a strong match versus a weak or irrelevant match.
- When `linkedin_employee_filter` is set or `is_pe_backed` is set to `pe_backed`/`not_pe_backed`, do not make the reranker depend on those constraints. Focus this brief on the non-filter dimensions: market, product, customer, use case, geography if not already structured, and exclusions not handled by structured filters.

2. `semantic_query`
- ENGLISH ONLY.
- Compact vector-retrieval query.
- Use 8 to 16 high-signal noun phrases max.
- Start with the exact market/category name, then add the most important qualifiers.
- Do not write a paragraph.
- When `linkedin_employee_filter` is set or `is_pe_backed` is set to `pe_backed`/`not_pe_backed`, do NOT repeat those constraints here. This query should focus only on what is not already handled by the structured filters.
- Never include employee-count language or PE/LBO ownership language here when the corresponding structured filter is active. Forbidden examples include: `LBO`, `PE-backed`, `private equity`, `sponsor-backed`, `fund-owned`, `under LBO`, `100 employees`, `>100 employees`, `headcount`, `company size`.

3. `bm25_keywords`
- ENGLISH ONLY.
- Return 4 to 8 discriminative keywords/short phrases.
- Focus on exact market labels, products, customer descriptors, and use cases.
- Use terms likely to appear verbatim on websites.
- Return `[]` only if there are truly no useful lexical terms.
- When `linkedin_employee_filter` is set or `is_pe_backed` is set to `pe_backed`/`not_pe_backed`, do NOT include those constraints in the keywords. Keep the keywords focused on the market, product, customer, and use-case descriptors that still need retrieval help.
- Never include employee-count language or PE/LBO ownership language in the keywords when the corresponding structured filter is active. Forbidden examples include: `LBO`, `PE-backed`, `private equity`, `sponsor-backed`, `fund-owned`, `under LBO`, `100 employees`, `>100 employees`, `headcount`, `company size`.

4. `rerank_mode`
- `broad` = exhaustive / wide-net search for a generic market-mapping request with no specific company anchor. Use it when the user wants as many relevant companies as possible in a market, such as "all companies", "full list", "entire market", or "most exhaustive view", AND the request is not framed around a specific company, a looked-up company profile, a competitor set, a build-up thesis around one company, or an existing shortlist. This still runs the reranker, but with a lenient, high-recall keep threshold.
- `balanced` = default mode for normal sourcing requests.
- `strict` = highly selective filtering. Use when the user says "only", "strictly", "exactly", requests a precise sub-segment, or gives strong exclusions.
- Important: "all companies" does NOT automatically mean `broad`.
- If the search is anchored on a specific company, a company profile from the context generator, a competitor/comparable request around one company, a build-up strategy around one company, or an existing shortlist, do NOT use `broad` even if the user asks for all relevant companies. Use `balanced` by default in those cases.

5. `restrict_to_shortlist`
- REQUIRED on every `search_companies` call.
- `true` means filter the currently loaded shortlist only.
- `false` means launch a new/global database search.
- Choose the value that best matches the user's intent in context.

6. `linkedin_employee_filter`
- Use it only when the user explicitly asks for a LinkedIn employee count constraint.
- Once this filter is used, do not restate the employee-count constraint in `search_context`, `semantic_query`, or `bm25_keywords` except if briefly needed for fidelity. Retrieval and reranking should focus on the remaining dimensions.
- In particular, never include employee-count terms in `semantic_query` or `bm25_keywords` once this filter is active.

7. `is_pe_backed`
- REQUIRED enum: `"not_specified"` | `"pe_backed"` | `"not_pe_backed"`.
- **Default value: `"not_specified"`.** This is the neutral choice and applies no ownership filter. Use it whenever the user does NOT explicitly mention PE-backed ownership.
- `"pe_backed"` → user explicitly wants companies already PE-backed / under LBO / sponsor-backed / owned by a fund.
- `"not_pe_backed"` → user explicitly wants companies that are NOT PE-backed.
- Only pick `"pe_backed"` or `"not_pe_backed"` when the user uses explicit ownership keywords like: `LBO`, `sous LBO`, `PE-backed`, `sponsor-backed`, `fund-owned`, `détenues par un fonds`, `non PE-backed`, `non cotées`.
- Do not infer ownership from sector, geography, size, or any non-ownership signal.
- Examples:
  "SaaS companies in France" → `is_pe_backed: "not_specified"`.
  "éditeurs logiciels bordelais" → `is_pe_backed: "not_specified"`.
  "companies under LBO in France" → `is_pe_backed: "pe_backed"`.
  "sociétés sous LBO avec plus de 200 employés" → `is_pe_backed: "pe_backed"`.
  "non-PE-backed cybersecurity companies in Paris" → `is_pe_backed: "not_pe_backed"`.
- When a concrete filter (`"pe_backed"` or `"not_pe_backed"`) is used, do not restate the PE-backed constraint in `search_context`, `semantic_query`, or `bm25_keywords` except if briefly needed for fidelity.
- Never include PE/LBO ownership terms in `semantic_query` or `bm25_keywords` when a concrete filter is active.

8. `geo_query`
- ALWAYS extract ALL geographic constraints from the user's request into this field.
- A dedicated geo-resolver LLM will parse this into HQ countries, presence countries, cities, and regions. You do NOT need to resolve it yourself.
- Just extract the geographic parts of the request as-is, including country names, city names, region names, and any mention of "presence" or "offices".
- Examples:
  "les sociétés bordelaises dans le SaaS" → geo_query: "bordelaises"
  "à Paris et Marseille" → geo_query: "à Paris et Marseille"
  "dans l'ouest de la France" → geo_query: "ouest de la France"
  "en France uniquement" → geo_query: "en France"
  "in northern Italy" → geo_query: "northern Italy"
  "banlieue parisienne" → geo_query: "banlieue parisienne"
  "dans un rayon de 50km de Lyon" → geo_query: "rayon de 50km de Lyon"
  "sociétés françaises avec une présence en Allemagne" → geo_query: "françaises avec présence en Allemagne"
  "en France et en Italie" → geo_query: "en France et en Italie"
  "based in Switzerland" → geo_query: "based in Switzerland"
- Leave empty ONLY if there is absolutely no geographic constraint in the request.
- Do NOT try to resolve city names, regions, or country codes yourself.
- Once geo_query is set, do not restate the geography constraint in `semantic_query` or `bm25_keywords`.

- Do NOT create any other structured filters for sector, B2B/B2C, business model, company size, or headquarters.
- Express all other constraints inside `search_context`, `semantic_query`, and `bm25_keywords`.

If the latest message is a context-generator transcript or you used `web_search`,
synthesize that information into the search plan. Do not just forward a short label.

# WHEN NOT TO CALL A TOOL
Do not call any tool when the user:
- asks you to clarify or rephrase your previous answer
- asks something unrelated to company search

# AFTER search_companies
- Write a concise analysis in 3-5 sentences.
- Identify patterns, clusters, or notable companies.
- Reference companies by name.
- The full results table is shown separately in the UI, so do not list every company.
- If there are 0 results, say so clearly and suggest how to broaden or adjust the search.
- IMPORTANT: Write the analysis in the user's language, not in the language of the data.

# RULES
- Never fabricate data. Only use information returned by the workflow.
- Do not use `web_search` for routine shortlist filtering.
- Keep responses concise and professional.
- Reference the conversation history naturally."""

# ── LLM instance ─────────────────────────────────────────────────────

_base_llm = ChatOpenAI(
    model="gpt-5.4",
    api_key=OPENAI_API_KEY,
    output_version="responses/v1",
    reasoning={"effort": "medium"},
)

_llm = _base_llm.bind_tools(TOOLS)

SHORTLIST_HISTORY_LIMIT = 5


# ── Node: agent ──────────────────────────────────────────────────────


async def agent_node(state: AgentState) -> dict:
    """Main agent node — calls LLM with function calling."""
    messages = state.get("messages", [])
    emit_phase("agent", "start", message_count=len(messages), shortlist_count=len(state.get("shortlist", [])))

    system_msg = {"role": "system", "content": AGENT_SYSTEM_PROMPT}

    # Detect user language and inject it explicitly so the LLM never guesses
    user_lang = _detect_user_language(state)
    if user_lang:
        system_msg["content"] += f"\n\nDETECTED USER LANGUAGE: {user_lang}. You MUST respond in {user_lang}."

    agent_context = state.get("_agent_context", "")
    if agent_context:
        system_msg["content"] += f"\n\nINTERNAL CONTEXT GENERATOR BRIEF:\n{agent_context}"

    shortlist = state.get("shortlist", [])
    if shortlist:
        system_msg["content"] += (
            f"\n\nCURRENT SHORTLIST: {len(shortlist)} companies loaded. "
            "You can reference them or filter them."
        )

    use_tools = not _is_post_search_tool_result(messages)
    llm = _llm if use_tools else _base_llm
    response = await llm.ainvoke([system_msg] + messages)
    tool_calls = getattr(response, "tool_calls", []) or []
    emit_phase(
        "agent",
        "complete",
        tool_calls=[tc.get("name") for tc in tool_calls],
        final_answer_only=not use_tools,
    )

    # Clear the hidden context once the agent has consumed it.
    return {"messages": [response], "_agent_context": "", "user_language": user_lang}


# ── Node: prepare_search ─────────────────────────────────────────────


async def prepare_search(state: AgentState) -> dict:
    """Extract search args from the agent's tool call."""
    emit_phase("prepare_search", "start")
    ai_msg: AIMessage = state["messages"][-1]
    tool_call = _find_tool_call(ai_msg, "search_companies")

    args = tool_call["args"]
    search_context = str(args.get("search_context", "") or "").strip()
    semantic_query = str(args.get("semantic_query", "") or "").strip()
    bm25_keywords = _normalize_bm25_keywords(args.get("bm25_keywords"))
    restrict = bool(args.get("restrict_to_shortlist", True))
    employee_filter = _normalize_linkedin_employee_filter(args.get("linkedin_employee_filter"))
    pe_backed_filter = _normalize_is_pe_backed(args.get("is_pe_backed"))
    geo_query = _normalize_string_filter(args.get("geo_query"))
    latest_user = _latest_user_message(state).strip()

    if not search_context:
        search_context = latest_user or semantic_query
    if not semantic_query:
        semantic_query = search_context or latest_user
    rerank_mode = _resolve_rerank_mode(
        raw_value=args.get("rerank_mode"),
        latest_user=latest_user,
        search_context=search_context,
    )

    # Return an empty dict when no filter is requested so the state reducer clears
    # any previous structured filter instead of silently keeping it.
    filters: dict[str, object] = {}
    if employee_filter:
        filters["linkedin_employees"] = employee_filter
    if pe_backed_filter is not None:
        filters["is_pe_backed"] = pe_backed_filter
    # Geo filters (country, city, region) are resolved by the geo_resolver node,
    # not here. We just pass geo_query to the state.

    intent = "filter" if restrict else "sourcing"
    # Skip rerank if only geo filter (geo_resolver will add the actual filters)
    skip_rerank = (
        intent == "filter"
        and bool(state.get("shortlist"))
        and not employee_filter
        and pe_backed_filter is None
        and bool(geo_query)
    )
    shortlist_history = _next_shortlist_history(
        state=state,
        latest_user=latest_user,
        next_intent=intent,
    )
    if shortlist_history != state.get("shortlist_history", []):
        current_shortlist = state.get("shortlist", [])
        latest_snapshot = shortlist_history[-1]
        emit_custom(
            "shortlist_archived",
            phase="prepare_search",
            history_id=latest_snapshot.get("id"),
            history_count=len(shortlist_history),
            company_count=len(current_shortlist),
            next_intent=intent,
        )
    emit_custom(
        "search_plan",
        phase="prepare_search",
        intent=intent,
        restrict_to_shortlist=restrict,
        search_context=search_context,
        semantic_query=semantic_query,
        bm25_keywords=bm25_keywords,
        rerank_mode=rerank_mode,
        linkedin_employee_filter=employee_filter,
        is_pe_backed=pe_backed_filter,
        geo_query=geo_query,
        skip_rerank=skip_rerank,
    )
    emit_phase(
        "prepare_search",
        "complete",
        intent=intent,
        restrict_to_shortlist=restrict,
        skip_rerank=skip_rerank,
    )

    return {
        "question": search_context,
        "intent": intent,
        "search_query": semantic_query,
        "bm25_keywords": bm25_keywords,
        "rerank_mode": rerank_mode,
        "filters": filters,
        "geo_query": geo_query,
        "skip_rerank": skip_rerank,
        "_tool_call_id": tool_call["id"],
        "shortlist_history": shortlist_history,
    }


# ── Node: summarize_search ───────────────────────────────────────────


async def summarize_search(state: AgentState) -> dict:
    """Create a compact ToolMessage summarizing search results."""
    emit_phase("summarize_search", "start")
    shortlist = state.get("shortlist", [])
    tool_call_id = state.get("_tool_call_id", "")
    count = len(shortlist)
    for error in state.get("errors", []) or []:
        emit_error("summarize_search", str(error))

    if count == 0:
        summary = "No companies found matching the search criteria."
    else:
        top = shortlist[:10]
        top_lines = "\n".join(
            f"- {r.get('company_name', '?')} | "
            f"{r.get('sector', '?')} | {r.get('linkedin_headquarters', '?')} | "
            f"{r.get('linkedin_employees', '?')} employees | "
            f"Products: {r.get('products_keywords', 'n/a')} | "
            f"Reason: {r.get('rerank_reason', 'n/a')} | "
            f"{(r.get('one_liner', '') or '')[:150]}"
            for r in top
        )
        summary = (
            f"{count} companies found and displayed in the shortlist table (visible to the user).\n"
            f"The user can already see ALL {count} companies with full details in the UI panel.\n"
            f"Do NOT repeat the full list — instead, summarize patterns, highlight key findings, "
            f"or answer the user's question.\n\n"
            f"Top {len(top)} for your reference:\n{top_lines}"
        )
        if count > len(top):
            summary += f"\n... plus {count - len(top)} more in the table."

    emit_metric(
        "summarize_search",
        shortlist_count=count,
        top_companies=[r.get("company_name", "?") for r in shortlist[:10]],
    )
    emit_phase("summarize_search", "complete", shortlist_count=count)

    return {
        "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
    }


async def prepare_initial_context(state: AgentState) -> dict:
    """Prepare context path directly from first user message (no agent tool call)."""
    query = _latest_user_message(state).strip() or state.get("question", "")
    emit_phase("prepare_initial_context", "start", question=query)

    from pe_qa_graph.nodes.lookup import lookup as _lookup_fn

    lookup_result = await _lookup_fn({"company_name": query, "shortlist": state.get("shortlist", [])})
    company_profile = lookup_result.get("lookup_result")

    out = {
        "question": query,
        "_tool_call_id": "",  # no function-calling handshake on first-turn forced context
        "_cg_messages": [],
        "_cg_structured": None,
    }
    if company_profile:
        out["lookup_result"] = company_profile
    emit_phase("prepare_initial_context", "complete", has_lookup_result=bool(company_profile))
    return out


# ── Node: summarize_context ──────────────────────────────────────────


async def summarize_context(state: AgentState) -> dict:
    """Summarize CG output for tool path, or store context for forced initial path."""
    emit_phase("summarize_context", "start")
    cg_messages = state.get("_cg_messages", [])
    cg_structured = state.get("_cg_structured")
    tool_call_id = state.get("_tool_call_id", "")

    raw_context = _format_cg_context(cg_messages)
    transcript = raw_context or "(empty)"
    intro = "Context generator exchange (verbatim):"
    payload = f"{intro}\n\n{transcript}"
    structured_context = _format_cg_structured_context(cg_structured)
    if structured_context:
        payload += f"\n\nInternal market context:\n\n{structured_context}"

    if tool_call_id:
        emit_custom("context_ready", phase="summarize_context", source="context_generator")
        emit_phase("summarize_context", "complete", via_tool_message=True)
        return {
            "_agent_context": payload,
            "messages": [ToolMessage(content="Context generator brief ready.", tool_call_id=tool_call_id)],
        }

    # First-turn forced path: keep the transcript hidden from the visible conversation
    # and pass it to the agent through private state instead.
    emit_custom("context_ready", phase="summarize_context", source="context_generator")
    emit_phase("summarize_context", "complete", via_tool_message=False)
    return {"_agent_context": payload}


# ── Helpers ──────────────────────────────────────────────────────────


def _find_tool_call(ai_msg: AIMessage, tool_name: str) -> dict:
    """Find a specific tool call in an AIMessage."""
    for tc in ai_msg.tool_calls:
        if tc["name"] == tool_name:
            return tc
    return ai_msg.tool_calls[0]


def _latest_user_message(state: AgentState) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


# Language code → full name for clear LLM instructions
_LANG_NAMES = {
    "en": "English", "fr": "French", "it": "Italian", "de": "German",
    "es": "Spanish", "pt": "Portuguese", "nl": "Dutch",
}


def _detect_user_language(state: AgentState) -> str:
    """Detect the language of the user's latest message using fast-langdetect."""
    text = _latest_user_message(state).strip()
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


def _is_post_search_tool_result(messages: list) -> bool:
    """Return True when the agent is answering a search ToolMessage."""
    if not messages:
        return False
    return isinstance(messages[-1], ToolMessage)


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


def _next_shortlist_history(
    *,
    state: AgentState,
    latest_user: str,
    next_intent: str,
) -> list[dict]:
    current_shortlist = state.get("shortlist", [])
    history = list(state.get("shortlist_history", []) or [])

    if not current_shortlist:
        return history

    snapshot = {
        "id": str(uuid4()),
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "question": state.get("question", ""),
        "rerank_mode": state.get("rerank_mode", "balanced"),
        "search_query": state.get("search_query", ""),
        "bm25_keywords": deepcopy(state.get("bm25_keywords", []) or []),
        "filters": deepcopy(state.get("filters") or {}),
        "intent": state.get("intent", ""),
        "archived_by_message": latest_user,
        "next_intent": next_intent,
        "company_count": len(current_shortlist),
        "shortlist": deepcopy(current_shortlist),
    }
    history.append(snapshot)
    return history[-SHORTLIST_HISTORY_LIMIT:]


def _normalize_bm25_keywords(value) -> list[str]:
    items: list[str] = []

    if isinstance(value, str):
        parts = value.split(",")
        items = [p.strip() for p in parts if p and p.strip()]
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                items.append(item.strip())

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
        if len(normalized) >= 8:
            break
    return normalized


def _resolve_rerank_mode(*, raw_value, latest_user: str, search_context: str) -> str:
    """Normalize explicit rerank mode or infer a sensible default from the request."""
    explicit = _normalize_rerank_mode(raw_value)
    if explicit is not None:
        return explicit
    return _infer_rerank_mode(latest_user=latest_user, search_context=search_context)


def _normalize_rerank_mode(value) -> str | None:
    if not isinstance(value, str):
        return None

    mode = value.strip().lower()
    if mode in {"broad", "balanced", "strict"}:
        return mode
    return None


def _infer_rerank_mode(*, latest_user: str, search_context: str) -> str:
    """Infer rerank strictness when the agent omitted the explicit field."""
    text = f"{latest_user}\n{search_context}".lower()

    broad_markers = (
        "all companies",
        "all players",
        "entire market",
        "full list",
        "full universe",
        "exhaustive",
        "broad search",
        "wide net",
        "toutes les boites",
        "toutes les boîtes",
        "tous les acteurs",
        "liste complete",
        "liste complète",
        "le plus large possible",
        "le plus exhaustif possible",
    )
    strict_markers = (
        "only",
        "strictly",
        "exactly",
        "pure play",
        "pure-play",
        "exclude",
        "excluding",
        "uniquement",
        "seulement",
        "strictement",
        "exactement",
        "precis",
        "précis",
        "pas les",
    )

    if any(marker in text for marker in broad_markers):
        return "broad"
    if any(marker in text for marker in strict_markers):
        return "strict"
    return "balanced"


def _normalize_linkedin_employee_filter(value) -> dict | None:
    if not isinstance(value, dict):
        return None

    raw_operator = str(value.get("operator", "") or "").strip().lower()
    raw_value = _coerce_positive_int(value.get("value"))

    operator_aliases = {
        "=": "eq",
        "==": "eq",
        "eq": "eq",
        ">": "gt",
        "gt": "gt",
        ">=": "gte",
        "gte": "gte",
        "<": "lt",
        "lt": "lt",
        "<=": "lte",
        "lte": "lte",
    }
    operator = operator_aliases.get(raw_operator)

    if operator is None or raw_value is None:
        return None

    return {"operator": operator, "value": raw_value}


def _normalize_is_pe_backed(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pe_backed", "true", "1", "yes"}:
            return True
        if normalized in {"not_pe_backed", "false", "0", "no"}:
            return False
        if normalized in {"not_specified", "", "any", "unspecified", "none", "null"}:
            return None
    return None


def _normalize_country_code(value) -> str | None:
    codes: list[str] = []

    if isinstance(value, str):
        parts = [part.strip().upper() for part in value.split(",")]
        codes = [part for part in parts if part]
    elif isinstance(value, list):
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


def _normalize_string_filter(value) -> str | None:
    """Normalize a free-text filter (city, region). Returns cleaned string or None."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _coerce_positive_int(value) -> int | None:
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        iv = int(value)
        return iv if iv > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            iv = int(stripped)
            return iv if iv > 0 else None
    return None
