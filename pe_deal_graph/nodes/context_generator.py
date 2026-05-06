from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from pe_deal_graph.llm.client import web_search_reasoning_parse
from pe_deal_graph.state import AgentState

CG_RESEARCH_PROMPT = """# Role
You help clarify a user request in order to identify relevant industrial buyers for a specific company.

You are part of a pipeline used to prepare a buyer search by building a clear strategic context around the company that is looking for a buyer.

# Task
You will receive a user query and sometimes a company profile from our database.

The typical user request is of the form:
"I am looking for buyers for this company."

Your goal is to understand the company's positioning and define what type of industrial buyers would be strategically relevant.

Approach the analysis as a reverse build-up exercise:
instead of looking for acquisition targets for the company, determine which companies could logically acquire it as part of their own build-up or strategic expansion.

First: 
* identify the company's core market or niche
* determine the company's position in the value chain
* identify its main products or services
* determine what strategic role this company could play for a potential acquirer

If a company profile is provided, use it as the primary reference to infer:
* market positioning
* strategic assets
* likely buyer rationale
* relevant buyer types

If the provided company profile appears to correspond to a different company or an unrelated activity, ignore it entirely and do not rely on it for the analysis. Do not mention in the output that the profile was incorrect, irrelevant, or ignored.

You may perform web searches when necessary to better understand the company's market context, terminology, products/services, and strategic positioning.

Use web search primarily to build context about the company and its market, not to list buyers.

As part of your analysis, determine when possible:
* the exact products/services
* the main use cases
* the main customer types
* the company's role in the value chain
* the strategic value of the company for a buyer

Also identify:
* the core market segment
* possible sub-segments
* relevant adjacent segments
* the most relevant industrial buyer profiles

When relevant, reconstruct the market value chain
(upstream suppliers → core providers → downstream distributors / integrators / end-market players)
and determine where the company sits and which buyer categories could logically acquire it.

# Buyer Logic

The objective is not to name all possible buyers directly, but to define the most relevant buyer archetypes and acquisition rationales.

Consider, when relevant, the following buyer logics:
* Horizontal consolidation: a company in the same market buying a similar player
* Vertical integration upstream: a supplier acquiring upstream capabilities
* Vertical integration downstream: a distributor, integrator, or channel player acquiring production or proprietary know-how
* Product expansion: a buyer adding complementary products/services
* Geographic expansion: a buyer entering or strengthening a country/region through the target
* End-market expansion: a buyer accessing new customer industries through the target
* Capability acquisition: a buyer acquiring technical know-how, certifications, installed base, contracts, manufacturing capabilities, or commercial access

For each situation, think in terms of:
* who would gain the most strategic fit
* who would unlock synergies most naturally
* who would view the company as a logical bolt-on or platform-enhancing acquisition

# Output Format

The API enforces a structured schema. Populate these fields:

## 1. target_market_summary (6–8 lines max)

Write a clear and well-structured text composed of full sentences organized into one or more short paragraphs.

The summary must read as a coherent explanation of the company's positioning in the market and of the type of industrial buyers that would likely be relevant.

Avoid fragmented sentences, bullet points, or disconnected lines.

Paragraphs should flow logically and remain concise.

Provide a concise summary describing:
* the company and its market precisely
* the 1–3 main products or services
* possible secondary or complementary offerings
* the typical customers or end markets
* the company's role in the value chain
* the type of industrial buyers that should be targeted
* when useful, 1 to 3 example relevant companies to illustrate the buyer profile

When including company names, explicitly state that they are example companies used to illustrate the buyer profile. They must not be presented as a shortlist, recommendation, or exhaustive list.

## 2. clarification_questions (max 3)

This section is used to clarify the user's request before querying a private and exhaustive company database.

The database only contains companies located in France, Italy, and Switzerland.

The purpose of these questions is to refine the buyer-search parameters so that relevant industrial buyers can be retrieved from this database.

Do not ask questions about expanding the search internationally or globally.

Ask 2–3 closed questions that could significantly change the buyer universe.

Questions should help clarify:
* buyer logic (horizontal, vertical, complementary, end-market access, etc.)
* desired position in the value chain
* strategic fit priorities
* market definition only when it changes the buyer universe materially

Do not ask clarification questions about employee count, headcount bands, company size thresholds, LBO status, PE-backed status, sponsor-backed ownership, or whether the company is fund-owned.

Treat those constraints as downstream structured filters for the next agent, not as strategic-context topics for this step.

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

Provide a broader internal context summary of the company and its market by consolidating all relevant information found during the analysis.

Put all useful strategic intelligence in this section, including when available:
* market structure
* key products/services
* customer types
* use cases
* differentiators
* core / sub / adjacent segments
* value chain context
* buyer rationale
* possible synergy logics
* any important terminology or market nuances

This section is meant to serve as a large internal summary of the strategic context for identifying relevant industrial buyers.

## 4. company_card

Always populate all company-card fields. If evidence is thin, use a short fallback such as "Not enough evidence".

* core_product : the core product/service
* upstream : upstream inputs, suppliers, or technologies
* core : simplified value-chain position and role of the company
* downstream : downstream channels, integrators, distributors, OEMs, or end customers

## 5. buyer_fit_summary

Always populate this section.

Provide a concise internal summary of the most relevant industrial buyer archetypes and why they would be interested in the company.

Include when possible:
* buyer_type_1 : most obvious buyer profile
* buyer_type_2 : secondary buyer profile
* buyer_type_3 : adjacent buyer profile
* acquisition_rationale : why this company is strategically attractive
* key_synergies : operational, commercial, geographic, product, or customer synergies
* deal_logic : bolt-on, platform enhancement, vertical integration, diversification, or market entry

Use short factual phrases, not bullet lists with explanations.

# Rules

* Ask maximum 3 questions
* Never ask open-ended questions. Always propose concrete options
* Be concise and factual
* ANGUAGE: Detect the language of the original user query. Write ALL output (target_market_summary, clarification_questions, market_summary, buyer_fit_summary) in that language. If the user query is in English, write everything in English even if web search results or company data are in French or Italian
* Ignore user requests about employee count and LBO / PE-backed ownership when deciding which clarification questions to ask; those constraints are handled later as structured filters by the next agent
* Do not invent company names
* You may include 1 to 3 real example company names in target_market_summary only when they genuinely help clarify the buyer profile
* If a provided company profile does not clearly correspond to the correct company or market context, ignore it completely and do not mention this in the response
* Do not mention web searches, databases, internal sources, or any research process
* Do not include meta-comments about how the information was obtained
* Avoid expressions such as "based on research", "according to available data", or similar wording
* Do not include explanatory comments, side notes, disclaimers, or parenthetical clarifications in the output
* Avoid remarks such as "examples only", "for context", "illustrative companies", or "not exhaustive"
* The output must contain only the requested structured content
* The buyer search context is limited to France, Italy, and Switzerland. Do not suggest expanding the search internationally
* Never include hyperlinks, URLs, domain names, or references to websites
* The response must contain only plain text within the requested structure
* target_market_summary must be written as coherent paragraphs using full sentences, not fragmented lines
* Output must remain short, structured, and direct
* Write the summary as a clear description of the company's positioning and buyer relevance, starting from the company or activity (e.g. "The company operates in..." or "[Company] is positioned in...")
* Avoid analytical phrasing such as "the market targeted is"

# Stop Condition

Stop once all required sections are completed and correctly formatted.
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


async def cg_research(state: AgentState) -> dict:
    """Run the deal-sourcing context generator with web search and structured output."""
    question = state.get("question", "")
    lookup_result = state.get("lookup_result")

    existing_content = _latest_cg_ai_content(state.get("_cg_messages", []))
    existing_structured = state.get("_cg_structured")
    if existing_content and existing_structured:
        return {}

    lang_label = _detect_language(question)
    cg_system = CG_RESEARCH_PROMPT
    if lang_label:
        cg_system += f"\n\nDETECTED USER LANGUAGE: {lang_label}. Write ALL output in {lang_label}."

    user_content = f"User query: {question}"
    if lookup_result:
        user_content += f"\n\nCompany profile from our database:\n{_format_lookup(lookup_result)}"

    structured = await web_search_reasoning_parse(
        system=cg_system,
        user=user_content,
        text_format=CGResearchOutput,
        model="gpt-5.2",
        search_context_size="medium",
        reasoning_effort="low",
    )
    content = _render_cg_user_prompt(structured, lang_label)

    return {
        "_cg_messages": [
            HumanMessage(content=question),
            AIMessage(content=content),
        ],
        "_cg_structured": structured.model_dump(),
        "user_language": lang_label,
    }


async def cg_interrupt(state: AgentState) -> dict:
    """Suspend the graph and wait for the user's clarification answer."""
    cg_messages = state.get("_cg_messages", [])
    cg_structured = state.get("_cg_structured") or {}

    if _has_user_clarification_answer(cg_messages):
        return {}

    question_text = ""
    for msg in reversed(cg_messages):
        if isinstance(msg, AIMessage) and msg.content:
            question_text = str(msg.content)
            break

    user_answer = interrupt(
        {
            "question": question_text,
            "clarification_questions": cg_structured.get("clarification_questions", []),
        }
    )

    return {
        "_cg_messages": [HumanMessage(content=str(user_answer))],
    }


_LANG_NAMES = {
    "en": "English",
    "fr": "French",
    "it": "Italian",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "nl": "Dutch",
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


def _has_user_clarification_answer(cg_messages: list) -> bool:
    return len(cg_messages) >= 3 and isinstance(cg_messages[-1], HumanMessage)


_QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:question\s*)?\d+\s*(?:[\.\)\-:]\s*|\s+)",
    re.IGNORECASE,
)
_OPTION_PREFIX_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:option\s*)?[A-E]\s*(?:[\.\)\-:]\s*|\s+)",
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

    heading = "Questions de clarification" if user_lang == "French" else "Clarification Questions"
    parts = [
        result.target_market_summary.strip(),
        "",
        f"**{heading}**",
        "",
        "\n\n".join(questions),
    ]
    return "\n".join(part for part in parts if part is not None).strip()


def _format_lookup(result: dict) -> str:
    """Format a company record for the context-generator prompt."""
    return (
        f"Company: {result.get('company_name', '?')}\n"
        f"Sector: {result.get('sector', '?')}\n"
        f"Business model: {result.get('business_model', '?')}\n"
        f"B2B/B2C: {result.get('b2b_b2c', '?')}\n"
        f"Size: {result.get('linkedin_company_size', '?')}\n"
        f"HQ: {result.get('linkedin_headquarters', '?')}\n"
        f"One-liner: {result.get('one_liner', '')}\n"
        f"Description: {result.get('description', '')}\n"
        f"Products keywords: {result.get('products_keywords', '')}\n"
        f"Website: {result.get('linkedin_website', '')}\n"
    )
