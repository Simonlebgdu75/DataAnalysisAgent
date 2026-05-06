from __future__ import annotations

from pe_deal_graph.llm.client import chat_json
from pe_deal_graph.state import AgentState
from shared.db.connection import get_pool
from shared.db.financials import get_actes
from shared.db.legal import enrich_single_with_legal
from shared.db.sql_templates import WEBSITE_FUZZY_MATCH, WEBSITE_TRIGRAM_MATCH

_NORMALIZE_SYSTEM = """\
The user searched for a company name that was not found in the database (maybe it's in \
a different language or has a different spelling). Suggest the most likely English / \
canonical version of the name.

Output JSON: {"normalized": "the normalized company name"}"""


async def lookup(state: AgentState) -> dict:
    """Lookup a company by name — shortlist first, then DB fallback."""
    company_name = state.get("company_name", "")
    if not company_name:
        return {"lookup_result": None, "errors": ["No company name provided"]}

    pool = await get_pool("linkedin")
    lookup_result: dict | None = None

    shortlist = state.get("shortlist", [])
    if shortlist:
        lookup_result = _find_in_shortlist(company_name, shortlist)

    if not lookup_result:
        rows = await pool.fetch(WEBSITE_FUZZY_MATCH, company_name)
        if rows:
            lookup_result = dict(rows[0])

    if not lookup_result:
        rows = await pool.fetch(WEBSITE_TRIGRAM_MATCH, company_name)
        if rows:
            lookup_result = dict(rows[0])

    if not lookup_result:
        try:
            result = await chat_json(
                system=_NORMALIZE_SYSTEM,
                user=f"Company name to normalize: {company_name}",
                model="gpt-4.1-mini",
            )
            normalized = result.get("normalized", "")
            if normalized and normalized.lower() != company_name.lower():
                if shortlist:
                    lookup_result = _find_in_shortlist(normalized, shortlist)
                if not lookup_result:
                    rows = await pool.fetch(WEBSITE_FUZZY_MATCH, normalized)
                    if rows:
                        lookup_result = dict(rows[0])
                if not lookup_result:
                    rows = await pool.fetch(WEBSITE_TRIGRAM_MATCH, normalized)
                    if rows:
                        lookup_result = dict(rows[0])
        except Exception:
            pass

    if lookup_result:
        siren = lookup_result.get("siren")
        if siren:
            actes = await get_actes(pool, siren)
            if actes:
                lookup_result["actes"] = actes
            lookup_result = await enrich_single_with_legal(lookup_result, pool)

    return {"lookup_result": lookup_result}


def _find_in_shortlist(name: str, shortlist: list[dict]) -> dict | None:
    """Find a company in the shortlist by name (case-insensitive, partial match)."""
    name_lower = name.lower()

    for row in shortlist:
        if (row.get("company_name") or "").lower() == name_lower:
            return row

    for row in shortlist:
        company = (row.get("company_name") or "").lower()
        if name_lower in company or company in name_lower:
            return row

    return None
