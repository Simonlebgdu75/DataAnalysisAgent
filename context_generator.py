from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from pe_qa_graph.llm.client import web_search_reasoning_parse
from pe_qa_graph.state import AgentState
from pe_qa_graph.stream import emit_custom, emit_phase

# ── Prompts ───────────────────────────────────────────────────────────

CG_RESEARCH_PROMPT = """# Role
You help clarify a user request in order to identify relevant companies within a specific market.

You are part of a pipeline used to discover competitors, comparable companies, or acquisition targets.

# Task
You will receive a user query and sometimes a company profile from our database.

Your goal is to understand the market implied by the request and define the type of companies that should be targeted.

First:
• Identify the target market or niche  
• Determine the type of companies involved (industry, business model, position in the value chain)  
• Identify the main products or services typically offered

If a company profile is provided, use it as the primary reference to infer:
• market positioning  
• competitors or comparable companies  
• possible acquisition targets  

If the provided company profile appears to correspond to a different company or an unrelated activity, ignore it entirely and do not rely on it for the analysis. Do not mention in the output that the profile was incorrect, irrelevant, or ignored.

You may perform web searches when necessary to better understand the market context, terminology, and typical products/services associated with the user request.

Use web search primarily to build context about the market, not to list companies.

As part of your analysis, determine when possible:
• the exact products/services  
• the main use cases  
• the main customer types  
• the typical revenue model  
• visible differentiators

Also identify:
• the core market segment  
• possible sub-segments  
• relevant adjacent segments

When relevant, reconstruct the market value chain  
(upstream suppliers → core providers → downstream distributors or service providers)  
and determine where the targeted companies sit.

# Output Format

The API enforces a structured schema. Populate these fields:

## 1. target_market_summary (6–8 lines max)

Write a clear and well-structured text composed of full sentences organized into one or more short paragraphs.

The summary must read as a coherent explanation of the company's positioning in the market and must remain easy to understand for a reader.

Avoid fragmented sentences, bullet points, or disconnected lines.

Paragraphs should flow logically and remain concise.

Provide a concise summary describing:
• the market precisely  
• the 1–3 main products or services  
• possible secondary or complementary offerings  
• the typical customers or end markets (B2B, B2C, industries)  
• the type of companies that should be targeted  
• when useful, 1 to 3 example relevant companies to illustrate the market

When including company names, explicitly state that they are example companies used to illustrate the market positioning. They must not be presented as a shortlist, recommendation, or exhaustive list.

## 2. clarification_questions (max 3)

This section is used to clarify the user's request before querying a private and exhaustive company database.

The database only contains companies located in France, Italy, and Switzerland.

The purpose of these questions is to refine the search parameters so that relevant companies can be retrieved from this database.

Do not ask questions about expanding the search internationally or globally.

Ask 2–3 closed questions that could significantly change the company search results.

Questions should help clarify:
• product or service scope  
• position in the value chain  
• strategic objective (competitor search vs build-up strategy)
• market definition only when it changes the target universe materially
• geography when it materially changes the target universe; available geographies are France, Switzerland, and Italy only

Do not ask clarification questions about employee count, headcount bands, company size thresholds, LBO status, PE-backed status, sponsor-backed ownership, or whether the company is fund-owned.

Treat those constraints as downstream structured filters for the next agent, not as market-clarification topics for this step.

Formatting rule:

There must be one blank line after each question and one blank line between each answer option.

Each answer option must appear on its own line.

Do not place multiple options on the same line.  
Do not use slashes "/", commas, or inline lists for answer options.

The format must follow exactly this structure:

1. Question

A. Option 1

B. Option 2

C. Option 3


2. Question

A. Option 1

B. Option 2

C. Option 3


Continue with the same structure if a third question is needed.

## 3. market_summary

Provide a broader internal context summary of the market by consolidating all relevant information found during the analysis.

Put all market intelligence gathered in this section, including when available:
• market structure  
• key products/services  
• customer types  
• use cases  
• revenue models  
• differentiators  
• core / sub / adjacent segments  
• value chain context  
• any important terminology or market nuances

This section is meant to serve as a large internal summary of the market context.

## 4. company_card

Always populate all company-card fields. If evidence is thin, use a short fallback such as "Not enough evidence".

• core_product : the core product/service  
• upstream : upstream inputs or suppliers  
• core : simplified value-chain position and target company role  
• downstream : downstream distributors, channels, or end customers

# Typical Use Cases

## Competitor / Comparable Company Search
The goal is to identify companies operating in the same market or offering similar products/services.

Questions should confirm:
• the exact market definition  
• the type of competitors to consider  
• the relevant segment of the value chain

## Build-Up Strategy
The goal is to identify potential acquisition targets.

Possible build-up strategies include:
• Horizontal: similar companies in the same market  
• Vertical: suppliers or distributors in the value chain  
• Product expansion: complementary offerings  

If no company reference is provided, ask the user which company the build-up strategy should target.

# Rules

• Ask maximum 3 questions  
• Never ask open-ended questions. Always propose concrete options  
• Be concise and factual  
• LANGUAGE: Detect the language of the original user query. Write ALL output (target_market_summary, clarification_questions, market_summary) in that language. If the user query is in English, write everything in English even if web search results or company data are in French or Italian  
• Ignore user requests about employee count and LBO / PE-backed ownership when deciding which clarification questions to ask; those constraints are handled later as structured filters by the next agent  
• Do not invent company names  
• You may include 1 to 3 real example company names in target_market_summary only when they genuinely help clarify the market  
• If a provided company profile does not clearly correspond to the correct company or market context, ignore it completely and do not mention this in the response  
• Do not mention web searches, databases, internal sources, or any research process  
• Do not include meta-comments about how the information was obtained  
• Avoid expressions such as "based on research", "according to available data", or similar wording  
• Do not include explanatory comments, side notes, disclaimers, or parenthetical clarifications in the output  
• Avoid remarks such as "examples only", "for context", "illustrative companies", or "not exhaustive"  
• The output must contain only the requested structured content  
• The company search context is limited to France, Italy, and Switzerland. Do not suggest expanding the search internationally  
• You may ask one clarification question about geography when it would materially change the company search results; if you do, limit the answer options to France, Switzerland, and Italy  
• Never include hyperlinks, URLs, domain names, or references to websites  
• The response must contain only plain text within the requested structure  
• target_market_summary must be written as coherent paragraphs using full sentences, not fragmented lines  
• Output must remain short, structured, and direct  
• Write the summary as a clear description of the company's positioning in the market, starting from the company or activity (e.g. "The company operates in..." or "[Company] is positioned in...")  
• Avoid analytical phrasing such as "the market targeted is"

# Stop Condition

Stop once all required sections are completed and correctly formatted.
"""

CG_RESEARCH_PROMPT_1 = """# Role
You help clarify a user request in order to identify **relevant companies within a specific market**.

You are part of a **pipeline used to discover competitors, comparable companies, or acquisition targets**.

# Task
You will receive a **user query** and sometimes a **company profile from our database**.

Your goal is to **understand the market implied by the request** and define the **type of companies that should be targeted**.

First:
- Identify the **target market or niche**
- Determine the **type of companies involved** (industry, business model, position in the value chain)
- Identify the **main products or services typically offered**

If a **company profile is provided**, use it as the **primary reference** to infer:
- market positioning  
- competitors or comparable companies  
- possible acquisition targets  

You should perform **web searches when necessary** to better understand the **market context, terminology, and typical products/services** associated with the user request.  
Use web search primarily to **build context about the market**, not to list companies.

As part of your analysis, determine when possible:
- the exact **products/services**
- the main **use cases**
- the main **customer types**
- the typical **revenue model**
- visible **differentiators**

Also identify:
- the **core market segment**
- possible **sub-segments**
- relevant **adjacent segments**

When relevant, reconstruct the **market value chain**
(upstream suppliers → core providers → downstream distributors or service providers)
and determine where the targeted companies sit.

# Structured Output

The API enforces a structured schema. Populate these fields:

## 1. target_market_summary (6–8 lines max)

Provide a concise summary describing:
- the **market precisely**
- the **1–3 main products or services**
- possible **secondary or complementary offerings**
- the **typical customers or end markets** (B2B, B2C, industries)
- the **type of companies that should be targeted**
- when useful, **1 to 3 example relevant companies** to make the market more concrete

## 2. clarification_questions (max 3)

Ask **2–3 closed questions** that could significantly change the company search results.

Questions should help clarify:
- **product or service scope**
- **position in the value chain**  
  (e.g. upstream suppliers, technology providers, distributors, direct competitors)
- **strategic objective** (competitor search vs build-up strategy)
- **market definition only when it materially changes the target universe**

Do **not** ask clarification questions about employee count, headcount bands, company size thresholds, LBO status, PE-backed status, sponsor-backed ownership, or whether a company is fund-owned.

Treat those constraints as **downstream structured filters** for the next agent, not as market-clarification topics for this step.

Each question must include **clear and limited answer options**.

## 3. market_summary

Provide a **broader internal context summary** of the market by consolidating **all relevant information found during the analysis**.  
Put **all market intelligence gathered** in this section, including when available:
- market structure
- key products/services
- customer types
- use cases
- revenue models
- differentiators
- core / sub / adjacent segments
- value chain context
- any important terminology or market nuances found through web research

This section is meant to serve as a **large internal summary of the market context**.

## 4. company_card
Always populate all company-card fields. If evidence is thin, use a short fallback such as "Not enough evidence".
- `core_product`: the core product/service
- `upstream`: upstream inputs or suppliers
- `core`: simplified value-chain position and target company role
- `downstream`: downstream distributors, channels, or end customers

# Typical Use Cases
## Competitor / Comparable Company Search
The goal is to identify **companies operating in the same market or offering similar products/services**.

Questions should confirm:
- the **exact market definition**
- the **type of competitors to consider**
- the **relevant segment of the value chain**

## Build-Up Strategy
The goal is to identify **potential acquisition targets**.

Possible build-up strategies include:
- **Horizontal**: similar companies in the same market  
- **Vertical**: suppliers or distributors in the value chain  
- **Product expansion**: complementary offerings  

If no company reference is provided, ask the user **which company the build-up strategy should target**.

# Rules

- Ask **maximum 3 questions**
- **Never ask open-ended questions. Always propose concrete options.**
- Be **concise and factual**
- **LANGUAGE**: Detect the language of the original user query. Write ALL output in that language. If the user writes in English, respond in English even if web results or data are in French/Italian
- Ignore user requests about employee count and LBO / PE-backed ownership when deciding which clarification questions to ask; those constraints are handled later as structured filters by the next agent
- Do **not invent company names**
- You may include **1 to 3 real example company names** in `target_market_summary` only when it genuinely helps clarify the market and when they are supported by the provided company profile or web research
- Use **web search only to understand the market context**
- Output must remain **short and structured**
- Write the summary as a **clear description of the company's positioning in the market**, starting from the company or activity (e.g. "The company operates in..." or "[Company] is positioned in...").
- Avoid analytical phrasing such as "the market targeted is".
"""


class CGClarificationQuestion(BaseModel):
    question: str = Field(description="Closed clarification question in the user's language.")
    options: list[str] = Field(
        min_length=2,
        max_length=5,
        description="2 to 5 concise answer options in the user's language.",
    )


class CGCompanyCard(BaseModel):
    core_product: str = Field(description="Core product or service.")
    upstream: str = Field(description="Upstream suppliers, inputs, or enabling layers.")
    core: str = Field(description="Core value-chain position of the target companies.")
    downstream: str = Field(description="Downstream distributors, channels, or end customers.")


class CGResearchOutput(BaseModel):
    target_market_summary: str = Field(
        description="User-facing 4-6 line target market summary in the user's language."
    )
    clarification_questions: list[CGClarificationQuestion] = Field(
        min_length=2,
        max_length=3,
        description="2 to 3 closed clarification questions with concrete answer options.",
    )
    market_summary: str = Field(
        description="Internal market context summary consolidating all relevant market intelligence."
    )
    company_card: CGCompanyCard

# ── Nodes ─────────────────────────────────────────────────────────────


async def cg_research(state: AgentState) -> dict:
    """GPT-5.2 does web search itself, then produces structured CG output."""
    question = state.get("question", "")
    lookup_result = state.get("lookup_result")
    emit_phase("cg_research", "start", has_lookup_result=bool(lookup_result))

    # Studio/runtime can occasionally replay this node around interrupts.
    # Reuse the prior CG answer if it is already in state to avoid a duplicate
    # web-search call and keep the execution idempotent.
    existing_content = _latest_cg_ai_content(state.get("_cg_messages", []))
    existing_structured = state.get("_cg_structured")
    if existing_content:
        _emit_cg_prompt_event(
            phase="cg_research",
            prompt=existing_content,
            structured=existing_structured,
            reused=True,
        )
        emit_phase("cg_research", "complete", reused=True)
        return {}

    # Detect user language, store in state, and inject into system prompt
    lang_label = _detect_language(question)
    cg_system = CG_RESEARCH_PROMPT
    if lang_label:
        cg_system += f"\n\nDETECTED USER LANGUAGE: {lang_label}. Write ALL output in {lang_label}."

    # Build user message with all available context
    user_content = f"User query: {question}"

    if lookup_result:
        user_content += f"\n\nCompany profile from our database:\n{_format_lookup(lookup_result)}"

    # Single call: model researches with built-in web search, then answers
    structured = await web_search_reasoning_parse(
        system=cg_system,
        user=user_content,
        text_format=CGResearchOutput,
        model="gpt-5.2",
        search_context_size="medium",
        reasoning_effort="low",
    )
    content = _render_cg_user_prompt(structured, lang_label)
    _emit_cg_prompt_event(
        phase="cg_research",
        prompt=content,
        structured=structured.model_dump(),
    )
    emit_phase("cg_research", "complete")

    return {
        "_cg_messages": [
            HumanMessage(content=question),
            AIMessage(content=content),
        ],
        "_cg_structured": structured.model_dump(),
        "user_language": lang_label,
    }


async def cg_interrupt(state: AgentState) -> dict:
    """Show questions to user, get their answer."""
    emit_phase("cg_interrupt", "start")
    cg_messages = state.get("_cg_messages", [])

    question_text = ""
    for msg in reversed(cg_messages):
        if isinstance(msg, AIMessage) and msg.content:
            question_text = msg.content
            break

    emit_custom("interrupt_ready", phase="cg_interrupt", prompt=question_text)
    # Mark the workflow step as done before suspending the graph. The user wait
    # happens outside the graph execution, so the UI should not keep this step
    # in progress until the resume stream arrives.
    emit_phase("cg_interrupt", "complete", waiting_for_user=True)
    user_answer = interrupt({"question": question_text})
    emit_custom("interrupt_resume", phase="cg_interrupt", answer=str(user_answer))

    return {
        "_cg_messages": [HumanMessage(content=str(user_answer))],
    }

# ── Helpers ──────────────────────────────────────────────────────────

_LANG_NAMES = {
    "en": "English", "fr": "French", "it": "Italian", "de": "German",
    "es": "Spanish", "pt": "Portuguese", "nl": "Dutch",
}


def _detect_language(text: str) -> str:
    """Detect language of text using fast-langdetect. Returns full name or empty string."""
    if not text or len(text.strip()) < 5:
        return ""
    try:
        from fast_langdetect import detect as ft_detect
        result = ft_detect(text.strip())
        first = result[0] if isinstance(result, list) else result
        lang_code = first.get("lang", "") if isinstance(first, dict) else ""
        return _LANG_NAMES.get(lang_code, lang_code.upper() if lang_code else "")
    except Exception:
        return ""


def _latest_cg_ai_content(cg_messages: list) -> str:
    for msg in reversed(cg_messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""


_QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:question\s*)?\d+\s*(?:[\.\)\-:]\s*|\s+)",
    re.IGNORECASE,
)
_OPTION_PREFIX_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:option\s*)?[A-E]\s*(?:[\.\)\-:]\s*|\s+)",
    re.IGNORECASE,
)


def _strip_question_prefix(text: str) -> str:
    cleaned = (text or "").strip()
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = _QUESTION_PREFIX_RE.sub("", cleaned, count=1).strip()
    return cleaned


def _strip_option_prefix(text: str) -> str:
    cleaned = (text or "").strip()
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = _OPTION_PREFIX_RE.sub("", cleaned, count=1).strip()
    return cleaned


def _render_cg_user_prompt(result: CGResearchOutput, user_lang: str = "") -> str:
    questions = []
    for index, item in enumerate(result.clarification_questions, start=1):
        question = _strip_question_prefix(item.question) or item.question.strip()
        option_lines = []
        for option in item.options:
            cleaned_option = _strip_option_prefix(option) or option.strip()
            if not cleaned_option:
                continue
            label = chr(ord("A") + len(option_lines))
            option_lines.append(f"- **{label}.** {cleaned_option}")

        block_parts = [f"**{index}. {question}**"]
        if option_lines:
            block_parts.extend(option_lines)
        questions.append("\n".join(block_parts).strip())

    parts = [
        result.target_market_summary.strip(),
        "",
        f"**{'Questions de clarification' if user_lang == 'French' else 'Clarification Questions'}**",
        "",
        "\n\n".join(questions),
    ]
    return "\n".join(part for part in parts if part is not None).strip()


def _emit_cg_prompt_event(*, phase: str, prompt: str, structured: dict | None, reused: bool = False) -> None:
    payload = {
        "phase": phase,
        "prompt": prompt,
        "reused": reused,
    }
    if structured:
        payload["target_market_summary"] = structured.get("target_market_summary", "")
        payload["clarification_questions"] = structured.get("clarification_questions", [])
        payload["market_summary"] = structured.get("market_summary", "")
        payload["company_card"] = structured.get("company_card", {})
    emit_custom("context_questions", **payload)


def _format_lookup(r: dict) -> str:
    """Format a company record for the CG context."""
    return (
        f"Company: {r.get('company_name', '?')}\n"
        f"Sector: {r.get('sector', '?')}\n"
        f"Business model: {r.get('business_model', '?')}\n"
        f"B2B/B2C: {r.get('b2b_b2c', '?')}\n"
        f"Size: {r.get('linkedin_company_size', '?')}\n"
        f"HQ: {r.get('linkedin_headquarters', '?')}\n"
        f"One-liner: {r.get('one_liner', '')}\n"
        f"Products: {r.get('products_keywords', '')}"
    )
