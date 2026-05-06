from __future__ import annotations

import asyncio
import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from pe_deal_graph.db.connection import get_pool
from pe_deal_graph.db.sql_templates import (
    DEAL_COLUMNS,
    SELECT_ACQUIRERS_FOR_DEALS,
    SELECT_INVESTORS_FOR_TARGETS,
)
from pe_deal_graph.llm.prompt import (
    build_rerank_system_prompt,
    build_rerank_user_prompt,
    normalize_rerank_mode,
)
from pe_deal_graph.nodes.utils import format_products_services
from pe_deal_graph.state import AgentState

logger = logging.getLogger(__name__)

BATCH_SIZE = 15
EXCLUDED_INVESTOR_NAMES = {"management"}


class RerankDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="linkedin_slug of the company")
    verdict: Literal["keep", "discard"] = Field(description="Whether to keep or discard the company")
    reason: str = Field(description="Short reason for the verdict")


class RerankDecisions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[RerankDecision] = Field(
        description="Ordered list of keep/discard decisions for the batch."
    )


_RERANK_LLM = ChatOpenAI(
    model="gpt-4.1-mini",
    use_responses_api=True,
)

_RERANK_RUNNABLE = _RERANK_LLM.with_structured_output(
    RerankDecisions,
    method="json_schema",
    strict=True,
)


async def rerank(state: AgentState) -> dict:
    """LLM-based company reranking using only company and product information."""
    return await rerank_results(
        shortlist=state.get("shortlist", []),
        search_context=state.get("question", ""),
        rerank_mode=state.get("rerank_mode"),
        rerank_context=state.get("context"),
    )


async def rerank_results(
    *,
    shortlist: list[dict],
    search_context: str,
    rerank_mode: str | None = None,
    rerank_context: str | dict | None = None,
) -> dict:
    """Rerank a shortlist and derive deals/acquirers from the kept targets."""
    results = shortlist
    rerank_mode = normalize_rerank_mode(rerank_mode)
    errors: list[str] = []

    if not results:
        return {
            "shortlist": [],
            "target_shortlist": [],
            "deals_shortlist": [],
            "acquirers_shortlist": [],
            "errors": ["No results to rerank"],
        }

    batches = [results[i : i + BATCH_SIZE] for i in range(0, len(results), BATCH_SIZE)]
    tasks = [
        _rerank_batch(batch, index, len(batches), search_context, rerank_context, rerank_mode, errors)
        for index, batch in enumerate(batches, start=1)
    ]
    batch_results = await asyncio.gather(*tasks)

    keep_reasons: dict[str, str] = {}
    for kept in batch_results:
        keep_reasons.update(kept)

    filtered: list[dict] = []
    for result in results:
        linkedin_slug = result.get("linkedin_slug")
        if linkedin_slug in keep_reasons:
            enriched = dict(result)
            enriched["rerank_reason"] = keep_reasons[linkedin_slug]
            enriched["target_rank"] = len(filtered) + 1
            filtered.append(enriched)

    deals_shortlist = await _load_deals(filtered)
    acquirers_shortlist = await _build_acquirers_shortlist(filtered, deals_shortlist)
    shareholding_slugs = _collect_shareholding_entity_slugs(filtered, acquirers_shortlist)
    shareholders_by_entity_slug = await _load_shareholders_by_target_slugs(shareholding_slugs)
    final_shortlist = _build_final_shortlist(
        filtered,
        deals_shortlist,
        acquirers_shortlist,
        shareholders_by_entity_slug=shareholders_by_entity_slug,
    )

    return {
        "shortlist": final_shortlist,
        "target_shortlist": filtered,
        "deals_shortlist": deals_shortlist,
        "acquirers_shortlist": acquirers_shortlist,
        "errors": errors if errors else [],
    }


async def _rerank_batch(
    batch: list[dict],
    batch_index: int,
    total_batches: int,
    search_context: str,
    rerank_context: str | dict | None,
    rerank_mode: str,
    errors: list[str],
) -> dict[str, str]:
    def _fmt_rev(r: dict) -> str:
        rev = r.get("revenue")
        src = r.get("revenue_source")
        if rev is not None:
            try: v = float(rev)
            except (ValueError, TypeError): return "n/a"
            if v >= 1e9: fmt = f"{v/1e9:.1f}Md"
            elif v >= 1e6: fmt = f"{v/1e6:.1f}M"
            elif v >= 1e3: fmt = f"{v/1e3:.0f}K"
            else: fmt = f"{v:.0f}"
            if src in ("inpi_consolidated", "inpi"):
                return f"{fmt}€ (INPI ±22%)"
            return f"{fmt}€ ({src or 'real'})"
        p50 = r.get("revenue_est_p50")
        if p50 is not None:
            try:
                p50f = float(p50)
                p35 = float(r.get("revenue_est_p35") or p50f)
                p65 = float(r.get("revenue_est_p65") or p50f)
            except (ValueError, TypeError):
                return "n/a"
            def _f(v):
                if v >= 1e9: return f"{v/1e9:.1f}Md"
                if v >= 1e6: return f"{v/1e6:.1f}M"
                if v >= 1e3: return f"{v/1e3:.0f}K"
                return f"{v:.0f}"
            return f"~{_f(p35)}-{_f(p65)}€ (ML estimate, MAPE 49%)"
        return "n/a"

    companies_text = "\n".join(
        f"- ID={row.get('linkedin_slug')} | {row.get('company_name', '?')} | "
        f"{row.get('one_liner', '') or 'No description'} | "
        f"Revenue: {_fmt_rev(row)} | "
        f"Products: {row.get('products_keywords', '') or 'n/a'} | "
        f"Products/Services: {format_products_services(row)}"
        for row in batch
    )

    try:
        response = await _RERANK_RUNNABLE.ainvoke(
            [
                ("system", build_rerank_system_prompt(rerank_mode)),
                (
                    "user",
                    build_rerank_user_prompt(
                        search_context=search_context,
                        rerank_context=rerank_context,
                        companies=companies_text,
                        rerank_mode=rerank_mode,
                    ),
                ),
            ]
        )
    except Exception as exc:
        errors.append(f"Rerank error: {exc}")
        return {
            row["linkedin_slug"]: "rerank error - kept by default"
            for row in batch
            if row.get("linkedin_slug") is not None
        }

    keep_reasons: dict[str, str] = {}
    for decision in response.decisions:
        if decision.verdict == "keep":
            keep_reasons[str(decision.id)] = decision.reason

    return keep_reasons


async def _load_deals(shortlist: list[dict]) -> list[dict]:
    if not shortlist:
        return []

    ids = [row["linkedin_slug"] for row in shortlist if row.get("linkedin_slug")]
    if not ids:
        return []

    pool = await get_pool()
    deal_rows = await pool.fetch(
        f"""
        SELECT {DEAL_COLUMNS}
        FROM deal d
        WHERE d.linkedin_slug = ANY($1::text[])
        ORDER BY d.deal_year DESC NULLS LAST, d.company_name
        """,
        ids,
    )
    deals_shortlist = [
        dict(row)
        for row in deal_rows
        if not _is_excluded_investor_name(row.get("acquirer_name"))
    ]
    target_meta_by_slug = {
        str(row["linkedin_slug"]): {
            "target_rrf_score": float(row.get("rrf_score", 0) or 0),
            "target_rank": row.get("target_rank", rank),
            "target_rerank_reason": row.get("rerank_reason"),
            "target_company_name": row.get("company_name"),
        }
        for rank, row in enumerate(shortlist, start=1)
        if row.get("linkedin_slug")
    }

    for deal in deals_shortlist:
        deal["deal_key"] = _deal_key(deal)
        deal["deal_origin"] = "deal"
        deal["is_synthetic"] = False
        linkedin_slug = deal.get("linkedin_slug")
        if linkedin_slug and str(linkedin_slug) in target_meta_by_slug:
            deal.update(target_meta_by_slug[str(linkedin_slug)])

    investor_rows = await pool.fetch(SELECT_INVESTORS_FOR_TARGETS, ids)
    for row in investor_rows:
        if _is_excluded_investor_name(row.get("acquirer_name")):
            continue
        target_slug = row.get("target_linkedin_slug")
        if not target_slug:
            continue
        target_meta = target_meta_by_slug.get(str(target_slug))
        if not target_meta:
            continue
        deals_shortlist.append(_build_shareholding_deal(row, target_meta, str(target_slug)))

    deals_shortlist.sort(
        key=lambda deal: (
            _int_or_default(deal.get("target_rank"), 999999),
            -_int_or_default(deal.get("deal_year"), 0),
            str(deal.get("company_name") or ""),
            str(deal.get("deal_key") or ""),
        ),
    )

    deals_by_slug: dict[str, list[dict]] = {}
    for row in deals_shortlist:
        linkedin_slug = row.get("linkedin_slug")
        if linkedin_slug:
            deals_by_slug.setdefault(str(linkedin_slug), []).append(row)

    for entry in shortlist:
        linkedin_slug = entry.get("linkedin_slug")
        if linkedin_slug and linkedin_slug in deals_by_slug:
            entry["deals"] = deals_by_slug[linkedin_slug]

    return deals_shortlist


async def _build_acquirers_shortlist(shortlist: list[dict], deals_shortlist: list[dict]) -> list[dict]:
    if not shortlist:
        return []

    pool = await get_pool()
    target_by_slug = {
        str(row["linkedin_slug"]): row
        for row in shortlist
        if row.get("linkedin_slug")
    }

    acquirer_rows = []
    deal_ids = [
        int(row["id"])
        for row in deals_shortlist
        if not row.get("is_synthetic") and row.get("id") is not None
    ]
    if deal_ids:
        acquirer_rows = await pool.fetch(SELECT_ACQUIRERS_FOR_DEALS, deal_ids)
    deals_by_id = {
        int(deal["id"]): deal
        for deal in deals_shortlist
        if not deal.get("is_synthetic") and deal.get("id") is not None
    }

    acquirers_by_key: dict[str, dict] = {}
    linked_deal_ids: set[int] = set()

    for row in acquirer_rows:
        if _is_excluded_investor_name(row.get("acquirer_name")):
            continue
        deal_id = int(row["deal_id"])
        deal = deals_by_id.get(deal_id)
        if not deal:
            continue
        linked_deal_ids.add(deal_id)
        key = f"fund:{row['fund_id']}"
        entry = _get_or_create_fund_entry(acquirers_by_key, key, row)
        _attach_deal_to_acquirer(entry, deal)

    for deal in deals_shortlist:
        if deal.get("is_synthetic"):
            continue
        deal_id = deal.get("id")
        acquirer_name = str(deal.get("acquirer_name") or "").strip()
        if (
            deal_id is None
            or int(deal_id) in linked_deal_ids
            or not acquirer_name
            or _is_excluded_investor_name(acquirer_name)
        ):
            continue
        key = f"name:{acquirer_name.lower()}"
        entry = acquirers_by_key.get(key)
        if entry is None:
            entry = {
                "acquirer_key": key,
                "fund_id": None,
                "acquirer_name": acquirer_name,
                "company_name": acquirer_name,
                "linkedin_slug": None,
                "logo": None,
                "one_liner": None,
                "description": None,
                "sector": None,
                "linkedin_headquarters": None,
                "linkedin_employees": None,
                "linkedin_website": None,
                "country_codes": None,
                "formatted_locations": None,
                "organization_type": None,
                "specialties": None,
                "aum_amount": None,
                "aum_currency": None,
                "aum_year": None,
                "funds_raised": None,
                "strategies": None,
                "rrf_score": 0.0,
                "acquirer_score": 0.0,
                "best_target_rank": None,
                "deal_count": 0,
                "deals": [],
                "shareholding_targets": [],
                "target_companies": [],
                "relation_sources": [],
                "_deal_ids": set(),
                "_relation_sources": set(),
                "_target_slugs": set(),
            }
            acquirers_by_key[key] = entry
        _attach_deal_to_acquirer(entry, deal)

    investor_rows = await pool.fetch(SELECT_INVESTORS_FOR_TARGETS, list(target_by_slug.keys()))
    for row in investor_rows:
        if _is_excluded_investor_name(row.get("acquirer_name")):
            continue
        target_slug = row.get("target_linkedin_slug")
        if not target_slug:
            continue
        target = target_by_slug.get(str(target_slug))
        if not target:
            continue
        key = f"fund:{row['fund_id']}"
        entry = _get_or_create_fund_entry(acquirers_by_key, key, row)
        _attach_deal_to_acquirer(entry, _build_shareholding_deal(row, target, str(target_slug)))
        _attach_shareholding_to_acquirer(entry, target, row)

    acquirers_shortlist = list(acquirers_by_key.values())
    for entry in acquirers_shortlist:
        entry["deals"] = sorted(
            entry["deals"],
            key=lambda deal: (
                _int_or_default(deal.get("target_rank"), 999999),
                -_int_or_default(deal.get("deal_year"), 0),
                str(deal.get("company_name") or ""),
            ),
        )
        entry["target_companies"] = sorted(
            entry["target_companies"],
            key=lambda company: (
                _int_or_default(company.get("target_rank"), 999999),
                -float(company.get("rrf_score") or 0),
                str(company.get("company_name") or ""),
            ),
        )
        entry["shareholding_targets"] = sorted(
            entry["shareholding_targets"],
            key=lambda company: (
                _int_or_default(company.get("target_rank"), 999999),
                str(company.get("relationship_type") or ""),
                str(company.get("company_name") or ""),
            ),
        )
        entry["relation_sources"] = sorted(entry.pop("_relation_sources", set()))
        entry["deal_count"] = len(entry["deals"])
        entry["shareholding_count"] = len(entry["shareholding_targets"])
        entry.pop("_deal_ids", None)
        entry.pop("_target_slugs", None)

    acquirers_shortlist.sort(
        key=lambda entry: (
            -float(entry.get("rrf_score") or 0),
            _int_or_default(entry.get("best_target_rank"), 999999),
            -int(entry.get("deal_count") or 0),
            -int(entry.get("shareholding_count") or 0),
            str(entry.get("company_name") or entry.get("acquirer_name") or ""),
        )
    )
    return acquirers_shortlist


def _collect_shareholding_entity_slugs(
    target_shortlist: list[dict],
    acquirers_shortlist: list[dict],
) -> list[str]:
    slugs: set[str] = set()
    for row in target_shortlist:
        slug = str(row.get("linkedin_slug") or "").strip()
        if slug:
            slugs.add(slug)
    for row in acquirers_shortlist:
        slug = str(row.get("linkedin_slug") or "").strip()
        if slug:
            slugs.add(slug)
    return sorted(slugs)


async def _load_shareholders_by_target_slugs(target_slugs: list[str]) -> dict[str, list[dict]]:
    if not target_slugs:
        return {}

    pool = await get_pool()
    rows = await pool.fetch(SELECT_INVESTORS_FOR_TARGETS, target_slugs)
    shareholders_by_target: dict[str, list[dict]] = {}
    seen_by_target: dict[str, set[tuple]] = {}
    for row in rows:
        if _is_excluded_investor_name(row.get("acquirer_name")):
            continue
        target_slug = str(row.get("target_linkedin_slug") or "")
        if not target_slug:
            continue
        _append_shareholder_item(
            shareholders_by_target,
            seen_by_target,
            target_slug,
            _shareholder_item_from_row(dict(row)),
        )
    return shareholders_by_target


def _get_or_create_fund_entry(acquirers_by_key: dict[str, dict], key: str, row: dict) -> dict:
    entry = acquirers_by_key.get(key)
    if entry is None:
        entry = {
            "acquirer_key": key,
            "fund_id": row["fund_id"],
            "acquirer_name": row["acquirer_name"],
            "company_name": row.get("linkedin_company_name") or row["acquirer_name"],
            "linkedin_slug": row.get("acquirer_linkedin_slug"),
            "logo": row.get("logo"),
            "one_liner": row.get("one_liner"),
            "description": row.get("description"),
            "sector": row.get("sector"),
            "linkedin_headquarters": row.get("linkedin_headquarters"),
            "linkedin_employees": row.get("linkedin_employees"),
            "linkedin_website": row.get("linkedin_website"),
            "country_codes": row.get("country_codes"),
            "formatted_locations": row.get("formatted_locations"),
            "organization_type": row.get("organization_type"),
            "specialties": row.get("specialties"),
            "aum_amount": row.get("aum_amount"),
            "aum_currency": row.get("aum_currency"),
            "aum_year": row.get("aum_year"),
            "funds_raised": row.get("funds_raised"),
            "strategies": row.get("strategies"),
            "rrf_score": 0.0,
            "acquirer_score": 0.0,
            "best_target_rank": None,
            "deal_count": 0,
            "deals": [],
            "shareholding_targets": [],
            "target_companies": [],
            "relation_sources": [],
            "_deal_ids": set(),
            "_relation_sources": set(),
            "_target_slugs": set(),
        }
        acquirers_by_key[key] = entry
    return entry


def _attach_deal_to_acquirer(entry: dict, deal: dict) -> None:
    deal_key = _deal_key(deal)
    if deal_key not in entry["_deal_ids"]:
        entry["_deal_ids"].add(deal_key)
        entry["deals"].append(dict(deal))

    target_score = float(deal.get("target_rrf_score") or 0)
    target_rank = deal.get("target_rank")
    if target_score > float(entry.get("rrf_score") or 0):
        entry["rrf_score"] = target_score
        entry["acquirer_score"] = target_score
    if target_rank is not None and (
        entry.get("best_target_rank") is None or target_rank < entry["best_target_rank"]
    ):
        entry["best_target_rank"] = target_rank

    target_slug = deal.get("linkedin_slug")
    if target_slug and target_slug not in entry["_target_slugs"]:
        entry["_target_slugs"].add(target_slug)
        entry["target_companies"].append(
            {
                "linkedin_slug": target_slug,
                "company_name": deal.get("company_name"),
                "rrf_score": target_score,
                "target_rank": target_rank,
                "rerank_reason": deal.get("target_rerank_reason"),
                "relation_source": "deal",
            }
        )
    entry["_relation_sources"].add("deal")


def _attach_shareholding_to_acquirer(entry: dict, target: dict, row: dict) -> None:
    target_slug = target.get("linkedin_slug")
    if not target_slug:
        return

    target_score = float(target.get("rrf_score", 0) or 0)
    target_rank = target.get("target_rank")
    if target_rank is None:
        target_rank = None
    if target_score > float(entry.get("rrf_score") or 0):
        entry["rrf_score"] = target_score
        entry["acquirer_score"] = target_score
    if target_rank is not None and (
        entry.get("best_target_rank") is None or target_rank < entry["best_target_rank"]
    ):
        entry["best_target_rank"] = target_rank

    relation_type = row.get("relationship_type") or "unknown"
    existing_shareholding = {
        (
            item.get("linkedin_slug"),
            item.get("relationship_type"),
            item.get("entry_year"),
            item.get("exit_year"),
        )
        for item in entry["shareholding_targets"]
    }
    shareholding_key = (
        target_slug,
        relation_type,
        row.get("entry_year"),
        row.get("exit_year"),
    )
    if shareholding_key not in existing_shareholding:
        entry["shareholding_targets"].append(
            {
                "linkedin_slug": target_slug,
                "company_name": target.get("company_name"),
                "rrf_score": target_score,
                "target_rank": target_rank,
                "rerank_reason": target.get("rerank_reason"),
                "relationship_type": relation_type,
                "entry_year": row.get("entry_year"),
                "exit_year": row.get("exit_year"),
            }
        )

    if target_slug not in entry["_target_slugs"]:
        entry["_target_slugs"].add(target_slug)
        entry["target_companies"].append(
            {
                "linkedin_slug": target_slug,
                "company_name": target.get("company_name"),
                "rrf_score": target_score,
                "target_rank": target_rank,
                "rerank_reason": target.get("rerank_reason"),
                "relation_source": "shareholding",
            }
        )

    entry["_relation_sources"].add(
        "shareholding_current" if relation_type == "current" else "shareholding_previous"
    )


def _deal_key(deal: dict) -> str:
    if deal.get("deal_key"):
        return str(deal["deal_key"])
    if deal.get("is_synthetic") and deal.get("synthetic_id"):
        return str(deal["synthetic_id"])
    if deal.get("id") is not None:
        return f"deal:{deal['id']}"
    return f"deal:unknown:{deal.get('linkedin_slug') or 'none'}:{deal.get('acquirer_name') or 'none'}"


def _build_shareholding_deal(row: dict, target: dict, target_slug: str) -> dict:
    relationship_type = str(row.get("relationship_type") or "unknown")
    entry_year = row.get("entry_year")
    exit_year = row.get("exit_year")
    synthetic_id = (
        f"shareholding:{row.get('fund_id')}:{target_slug}:{relationship_type}:"
        f"{entry_year or 'none'}:{exit_year or 'none'}"
    )
    if relationship_type == "current":
        deal_type = "shareholding_current"
        year = entry_year
        description = (
            f"Current shareholder in {target.get('company_name') or row.get('target_company_name') or target_slug}"
        )
        if entry_year:
            description += f" since {entry_year}"
    elif relationship_type == "previous":
        deal_type = "shareholding_previous"
        year = exit_year or entry_year
        description = (
            f"Previous shareholder in {target.get('company_name') or row.get('target_company_name') or target_slug}"
        )
        if entry_year and exit_year:
            description += f" from {entry_year} to {exit_year}"
        elif exit_year:
            description += f" until {exit_year}"
    else:
        deal_type = f"shareholding_{relationship_type}"
        year = entry_year or exit_year
        description = (
            f"{relationship_type.capitalize()} shareholder relation with "
            f"{target.get('company_name') or row.get('target_company_name') or target_slug}"
        )

    return {
        "id": synthetic_id,
        "synthetic_id": synthetic_id,
        "deal_key": synthetic_id,
        "is_synthetic": True,
        "deal_origin": "shareholding",
        "company_name": target.get("company_name") or row.get("target_company_name"),
        "linkedin_slug": target_slug,
        "country": None,
        "acquirer_name": row.get("acquirer_name"),
        "fund_id": row.get("fund_id"),
        "deal_type": deal_type,
        "deal_date": None,
        "deal_year": year,
        "deal_date_parsed": None,
        "deal_month": None,
        "description": description,
        "source_url": None,
        "website": None,
        "source_type": "fund_portfolio",
        "loaded_at": None,
        "relationship_type": relationship_type,
        "entry_year": entry_year,
        "exit_year": exit_year,
        "latest_revenue_amount": row.get("latest_revenue_amount"),
        "latest_revenue_currency": row.get("latest_revenue_currency"),
        "latest_revenue_year": row.get("latest_revenue_year"),
        "revenue_source": row.get("revenue_source"),
        "target_rrf_score": float(target.get("target_rrf_score", target.get("rrf_score", 0)) or 0),
        "target_rank": target.get("target_rank"),
        "target_rerank_reason": target.get("target_rerank_reason", target.get("rerank_reason")),
        "target_company_name": target.get("target_company_name", target.get("company_name")),
    }


def _apply_shortlist_rationale(entry: dict) -> None:
    rationale = _build_shortlist_rationale(entry)
    if not rationale:
        return

    entry["shortlist_rationale"] = rationale
    if entry.get("description") not in (None, "", "n.a.", "n.d.") and entry.get("profile_description") in (
        None,
        "",
        "n.a.",
        "n.d.",
    ):
        entry["profile_description"] = entry.get("description")
    entry["description"] = rationale

    if entry.get("rerank_reason") in (None, "", "n.a.", "n.d."):
        entry["rerank_reason"] = rationale


def _build_shortlist_rationale(entry: dict) -> str | None:
    entity_type = str(entry.get("entity_type") or "").strip().lower()
    entity_roles = {
        str(role).strip().lower()
        for role in (entry.get("entity_roles") or [])
        if str(role).strip()
    }
    if entity_type == "buyer" or "buyer" in entity_roles:
        return _build_buyer_shortlist_rationale(entry)
    return _build_target_shortlist_rationale(entry)


def _build_buyer_shortlist_rationale(entry: dict) -> str | None:
    acquisition_clauses: list[str] = []
    seen_targets: set[str] = set()

    for deal in _collect_buyer_relation_deals(entry):
        if _is_shareholding_relation_deal(deal):
            continue

        target_name = _get_entity_name(deal, ["company_name", "target_company_name"])
        target_key = _normalize_name_key(target_name)
        if not target_name or target_key in seen_targets:
            continue

        seen_targets.add(target_key)
        clause = f"it acquired {target_name}"
        deal_year = _get_year_label(deal.get("deal_year"))
        if deal_year:
            clause += f" in {deal_year}"

        target_reason = _clean_rationale_fragment(
            deal.get("target_rerank_reason") or deal.get("rerank_reason")
        )
        if target_reason:
            clause += f", which matches your search because {target_reason}"

        acquisition_clauses.append(clause)
        if len(acquisition_clauses) >= 2:
            break

    holding_clause = ""
    current_holdings = entry.get("shareholding_targets") or []
    for holding in current_holdings:
        if not isinstance(holding, dict):
            continue
        if str(holding.get("relationship_type") or "").strip().lower() != "current":
            continue

        target_name = _get_entity_name(holding, ["company_name"])
        target_key = _normalize_name_key(target_name)
        if not target_name or target_key in seen_targets:
            continue

        clause = f"currently holds {target_name}"
        entry_year = _get_year_label(holding.get("entry_year"))
        if entry_year:
            clause += f" since {entry_year}"

        target_reason = _clean_rationale_fragment(holding.get("rerank_reason"))
        if target_reason:
            clause += f", which matches your search because {target_reason}"

        holding_clause = clause
        break

    if holding_clause:
        acquisition_clauses.append(holding_clause)

    if acquisition_clauses:
        return f"Buyer because {_join_clauses(acquisition_clauses)}."

    backing_clause = _build_backing_clause(entry)
    if backing_clause:
        relation_reason = _find_relation_reason(entry)
        clauses = [backing_clause]
        if relation_reason:
            clauses.append(f"it matches the screened profile because {relation_reason}")
        return f"Buyer candidate because {_join_clauses(clauses)}."

    return None


def _build_target_shortlist_rationale(entry: dict) -> str | None:
    clauses: list[str] = []

    backing_clause = _build_backing_clause(entry)
    if backing_clause:
        clauses.append(backing_clause)

    rerank_reason = _find_relation_reason(entry)
    if rerank_reason:
        clauses.append(f"it matches your search because {rerank_reason}")

    if not clauses:
        return None

    return f"Relevant because {_join_clauses(clauses)}."


def _collect_buyer_relation_deals(entry: dict) -> list[dict]:
    combined: list[dict] = []
    seen_keys: set[str] = set()

    for field in ("source_deals", "deals"):
        for item in entry.get(field) or []:
            if not isinstance(item, dict):
                continue
            deal_key = _deal_key(item)
            if deal_key in seen_keys:
                continue
            seen_keys.add(deal_key)
            combined.append(item)

    return combined


def _is_shareholding_relation_deal(deal: dict) -> bool:
    raw = str(deal.get("deal_type") or "").strip().lower()
    normalized = "".join(ch for ch in raw if ch.isalnum())
    return normalized.startswith("shareholding")


def _collect_shareholder_names(items: list[dict], *, relationship_type: str) -> list[str]:
    names: list[str] = []
    seen_names: set[str] = set()
    expected_relationship = relationship_type.strip().lower()

    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("relationship_type") or "").strip().lower() != expected_relationship:
            continue

        name = _get_entity_name(item, ["company_name", "acquirer_name"])
        key = _normalize_name_key(name)
        if not name or key in seen_names:
            continue

        seen_names.add(key)
        names.append(name)

    return names


def _build_backing_clause(entry: dict) -> str:
    backers = _collect_shareholder_names(entry.get("shareholding") or [], relationship_type="current")
    if backers:
        return f"it is backed by {_join_names(backers[:3])}"

    financing_clauses = _collect_financing_signal_clauses(entry)
    if financing_clauses:
        return _join_clauses(financing_clauses[:2])

    return ""


def _collect_financing_signal_clauses(entry: dict) -> list[str]:
    clauses: list[str] = []
    seen_signals: set[tuple[str, str, str]] = set()

    for deal in entry.get("source_deals") or []:
        if not isinstance(deal, dict):
            continue
        if _is_shareholding_relation_deal(deal):
            continue
        if not _is_lbo_like_deal(deal):
            continue

        deal_type = _normalize_deal_type(deal.get("deal_type"))
        acquirer_name = _get_entity_name(deal, ["acquirer_name"])
        deal_year = _get_year_label(deal.get("deal_year"))
        signal_key = (deal_type, _normalize_name_key(acquirer_name), deal_year)
        if signal_key in seen_signals:
            continue
        seen_signals.add(signal_key)

        clause = _describe_financing_signal(deal_type, acquirer_name, deal_year)
        if clause:
            clauses.append(clause)
        if len(clauses) >= 2:
            break

    return clauses


def _describe_financing_signal(deal_type: str, acquirer_name: str, deal_year: str) -> str:
    if deal_type == "lbo":
        clause = "it went through an LBO"
    elif deal_type in {"vc", "venturecapital"}:
        clause = "it raised VC"
    elif deal_type == "equityfundraising":
        clause = "it raised equity"
    else:
        clause = "it has financing backing"

    if acquirer_name:
        clause += f" from {acquirer_name}"
    if deal_year:
        clause += f" in {deal_year}"
    return clause


def _find_relation_reason(entry: dict) -> str:
    direct_reason = _clean_rationale_fragment(
        entry.get("rerank_reason") or entry.get("target_rerank_reason")
    )
    if direct_reason:
        return direct_reason

    for field in ("target_companies", "shareholding_targets", "source_deals"):
        for item in entry.get(field) or []:
            if not isinstance(item, dict):
                continue
            reason = _clean_rationale_fragment(
                item.get("rerank_reason") or item.get("target_rerank_reason")
            )
            if reason:
                return reason

    return ""


def _get_entity_name(item: dict, keys: list[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value in (None, "", "n.a.", "n.d."):
            continue
        text = " ".join(str(value).strip().split())
        if text:
            return text
    return ""


def _clean_rationale_fragment(value) -> str:
    if value in (None, "", "n.a.", "n.d."):
        return ""

    text = " ".join(str(value).strip().split()).rstrip(" .;:,")
    if not text:
        return ""

    first_word, separator, rest = text.partition(" ")
    if first_word in {"It", "This", "The", "A", "An", "These", "Those"}:
        text = first_word.lower() + (separator + rest if separator else "")

    return text


def _get_year_label(value) -> str:
    if value in (None, "", "n.a.", "n.d."):
        return ""
    text = str(value).strip()
    return text if text else ""


def _normalize_deal_type(value) -> str:
    raw = str(value or "").strip().lower()
    return "".join(ch for ch in raw if ch.isalnum())


def _join_clauses(clauses: list[str]) -> str:
    compact = [clause.strip().rstrip(" .;:,") for clause in clauses if clause and clause.strip()]
    if not compact:
        return ""
    if len(compact) == 1:
        return compact[0]
    if len(compact) == 2:
        return f"{compact[0]} and {compact[1]}"
    return "; ".join(compact[:-1]) + f"; and {compact[-1]}"


def _join_names(names: list[str]) -> str:
    compact = [name.strip() for name in names if name and name.strip()]
    if not compact:
        return ""
    if len(compact) == 1:
        return compact[0]
    if len(compact) == 2:
        return f"{compact[0]} and {compact[1]}"
    return ", ".join(compact[:-1]) + f", and {compact[-1]}"


def _build_final_shortlist(
    target_shortlist: list[dict],
    deals_shortlist: list[dict],
    acquirers_shortlist: list[dict],
    *,
    shareholders_by_entity_slug: dict[str, list[dict]] | None = None,
) -> list[dict]:
    target_by_slug = {
        str(row["linkedin_slug"]): row
        for row in target_shortlist
        if row.get("linkedin_slug")
    }
    acquirer_index = _build_acquirer_index(acquirers_shortlist)
    shareholders_by_target = shareholders_by_entity_slug or _build_shareholders_by_target(acquirers_shortlist)

    final_by_key: dict[str, dict] = {}
    for deal in deals_shortlist:
        if _is_build_up_deal(deal):
            entry = _get_or_create_final_buyer_entry(final_by_key, deal, acquirer_index)
            _attach_source_deal(entry, deal)
            _attach_build_up(entry, deal, target_by_slug)
            continue

        entry = _get_or_create_final_target_entry(final_by_key, deal, target_by_slug)
        _attach_source_deal(entry, deal)

    final_shortlist = list(final_by_key.values())
    for entry in final_shortlist:
        entry_slug = str(entry.get("linkedin_slug") or "")
        if entry_slug and entry_slug in shareholders_by_target:
            entry["shareholding"] = list(shareholders_by_target[entry_slug])
            entry["shareholding_count"] = len(entry["shareholding"])
        entry["source_deals"] = sorted(
            entry["source_deals"],
            key=lambda deal: (
                _int_or_default(deal.get("target_rank"), 999999),
                -_int_or_default(deal.get("deal_year"), 0),
                str(deal.get("company_name") or ""),
                str(deal.get("deal_key") or ""),
            ),
        )
        entry["build_up"] = sorted(
            entry["build_up"],
            key=lambda item: (
                -_int_or_default(item.get("deal_year"), 0),
                str(item.get("company_name") or ""),
            ),
        )
        entry["shareholding"] = sorted(
            entry["shareholding"],
            key=lambda item: (
                str(item.get("relationship_type") or ""),
                -_int_or_default(item.get("entry_year"), 0),
                str(item.get("company_name") or ""),
            ),
        )
        entry["source_deal_count"] = len(entry["source_deals"])
        entry["build_up_count"] = len(entry["build_up"])
        entry["shareholding_count"] = len(entry["shareholding"])
        entry["is_under_lbo"] = _is_entry_under_lbo(entry)
        entry["has_sector_build_up"] = entry["build_up_count"] > 0
        entry["entity_roles"] = sorted(entry["entity_roles"])
        _apply_shortlist_rationale(entry)
        entry.pop("_source_deal_keys", None)
        entry.pop("_build_up_keys", None)

    final_shortlist.sort(
        key=lambda entry: (
            -_float_or_default(entry.get("final_score"), 0.0),
            _int_or_default(entry.get("best_target_rank"), 999999),
            -_int_or_default(entry.get("source_deal_count"), 0),
            -_int_or_default(entry.get("build_up_count"), 0),
            str(entry.get("company_name") or ""),
        )
    )
    return final_shortlist


def _build_acquirer_index(acquirers_shortlist: list[dict]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for acquirer in acquirers_shortlist:
        if acquirer.get("fund_id") is not None:
            index[f"fund:{acquirer['fund_id']}"] = acquirer
        if acquirer.get("linkedin_slug"):
            index[f"slug:{str(acquirer['linkedin_slug']).lower()}"] = acquirer
        for name in (acquirer.get("company_name"), acquirer.get("acquirer_name")):
            normalized = _normalize_name_key(name)
            if normalized:
                index[f"name:{normalized}"] = acquirer
    return index


def _build_shareholders_by_target(acquirers_shortlist: list[dict]) -> dict[str, list[dict]]:
    shareholders_by_target: dict[str, list[dict]] = {}
    seen_by_target: dict[str, set[tuple]] = {}
    for acquirer in acquirers_shortlist:
        for shareholding in acquirer.get("shareholding_targets") or []:
            target_slug = str(shareholding.get("linkedin_slug") or "")
            if not target_slug:
                continue
            _append_shareholder_item(
                shareholders_by_target,
                seen_by_target,
                target_slug,
                _shareholder_item_from_acquirer(acquirer, shareholding),
            )
    return shareholders_by_target


def _shareholder_item_from_acquirer(acquirer: dict, shareholding: dict) -> dict:
    return {
        "fund_id": acquirer.get("fund_id"),
        "company_name": acquirer.get("company_name") or acquirer.get("acquirer_name"),
        "acquirer_name": acquirer.get("acquirer_name"),
        "linkedin_slug": acquirer.get("linkedin_slug"),
        "logo": acquirer.get("logo"),
        "one_liner": acquirer.get("one_liner"),
        "description": acquirer.get("description"),
        "aum_amount": acquirer.get("aum_amount"),
        "aum_currency": acquirer.get("aum_currency"),
        "aum_year": acquirer.get("aum_year"),
        "funds_raised": acquirer.get("funds_raised"),
        "strategies": acquirer.get("strategies"),
        "relationship_type": shareholding.get("relationship_type"),
        "entry_year": shareholding.get("entry_year"),
        "exit_year": shareholding.get("exit_year"),
        "source_deal_key": None,
    }


def _shareholder_item_from_row(row: dict) -> dict:
    return {
        "fund_id": row.get("fund_id"),
        "company_name": row.get("linkedin_company_name") or row.get("acquirer_name"),
        "acquirer_name": row.get("acquirer_name"),
        "linkedin_slug": row.get("acquirer_linkedin_slug"),
        "logo": row.get("logo"),
        "one_liner": row.get("one_liner"),
        "description": row.get("description"),
        "aum_amount": row.get("aum_amount"),
        "aum_currency": row.get("aum_currency"),
        "aum_year": row.get("aum_year"),
        "funds_raised": row.get("funds_raised"),
        "strategies": row.get("strategies"),
        "relationship_type": row.get("relationship_type"),
        "entry_year": row.get("entry_year"),
        "exit_year": row.get("exit_year"),
        "source_deal_key": None,
    }


def _append_shareholder_item(
    shareholders_by_target: dict[str, list[dict]],
    seen_by_target: dict[str, set[tuple]],
    target_slug: str,
    item: dict,
) -> None:
    dedupe_key = (
        item.get("fund_id"),
        item.get("company_name"),
        item.get("relationship_type"),
        item.get("entry_year"),
        item.get("exit_year"),
    )
    existing = seen_by_target.setdefault(target_slug, set())
    if dedupe_key in existing:
        return
    existing.add(dedupe_key)
    shareholders_by_target.setdefault(target_slug, []).append(item)


def _get_or_create_final_target_entry(
    final_by_key: dict[str, dict],
    deal: dict,
    target_by_slug: dict[str, dict],
) -> dict:
    target_slug = str(deal.get("linkedin_slug") or "")
    target_profile = target_by_slug.get(target_slug, {})
    company_name = target_profile.get("company_name") or deal.get("company_name")
    entity_key = _entity_key(
        fund_id=None,
        linkedin_slug=target_slug or target_profile.get("linkedin_slug"),
        company_name=company_name,
    )
    entry = final_by_key.get(entity_key)
    if entry is None:
        entry = dict(target_profile) if target_profile else {}
        entry.update(
            {
                "entity_key": entity_key,
                "entity_type": "target",
                "entity_roles": ["target"],
                "company_name": company_name,
                "linkedin_slug": target_slug or target_profile.get("linkedin_slug"),
                "final_score": _float_or_default(
                    target_profile.get("rrf_score", deal.get("target_rrf_score")),
                    0.0,
                ),
                "best_target_rank": _int_or_default(
                    target_profile.get("target_rank", deal.get("target_rank")),
                    999999,
                ),
                "source_deals": [],
                "build_up": [],
                "shareholding": [],
                "_source_deal_keys": set(),
                "_build_up_keys": set(),
            }
        )
        final_by_key[entity_key] = entry
    else:
        _merge_entity_metadata(entry, target_profile)
        if "target" not in entry["entity_roles"]:
            entry["entity_roles"].append("target")
    return entry


def _get_or_create_final_buyer_entry(
    final_by_key: dict[str, dict],
    deal: dict,
    acquirer_index: dict[str, dict],
) -> dict:
    acquirer = _lookup_acquirer_for_deal(deal, acquirer_index)
    company_name = (
        acquirer.get("company_name")
        if acquirer
        else deal.get("acquirer_name")
    )
    entity_key = _entity_key(
        fund_id=acquirer.get("fund_id") if acquirer else deal.get("fund_id"),
        linkedin_slug=acquirer.get("linkedin_slug") if acquirer else None,
        company_name=company_name,
    )
    entry = final_by_key.get(entity_key)
    if entry is None:
        entry = dict(acquirer) if acquirer else {}
        entry.update(
            {
                "entity_key": entity_key,
                "entity_type": "buyer",
                "entity_roles": ["buyer"],
                "company_name": company_name,
                "acquirer_name": (
                    acquirer.get("acquirer_name")
                    if acquirer
                    else deal.get("acquirer_name")
                ),
                "final_score": _float_or_default(
                    (acquirer or {}).get("acquirer_score", (acquirer or {}).get("rrf_score", deal.get("target_rrf_score"))),
                    0.0,
                ),
                "best_target_rank": _int_or_default(
                    (acquirer or {}).get("best_target_rank", deal.get("target_rank")),
                    999999,
                ),
                "source_deals": [],
                "build_up": [],
                "shareholding": [],
                "_source_deal_keys": set(),
                "_build_up_keys": set(),
            }
        )
        final_by_key[entity_key] = entry
    else:
        _merge_entity_metadata(entry, acquirer)
        if not entry.get("acquirer_name"):
            entry["acquirer_name"] = (
                acquirer.get("acquirer_name")
                if acquirer
                else deal.get("acquirer_name")
            )
        if "buyer" not in entry["entity_roles"]:
            entry["entity_roles"].append("buyer")
    return entry


def _merge_entity_metadata(entry: dict, source: dict | None) -> None:
    if not isinstance(source, dict) or not source:
        return

    for key, value in source.items():
        if key.startswith("_"):
            continue
        if value in (None, "", [], {}, "n.a.", "n.d."):
            continue
        existing = entry.get(key)
        if existing in (None, "", [], {}, "n.a.", "n.d."):
            entry[key] = value


def _attach_source_deal(entry: dict, deal: dict) -> None:
    deal_key = str(deal.get("deal_key") or _deal_key(deal))
    if deal_key in entry["_source_deal_keys"]:
        return
    entry["_source_deal_keys"].add(deal_key)
    entry["source_deals"].append(dict(deal))
    target_score = _float_or_default(deal.get("target_rrf_score"), 0.0)
    if target_score > _float_or_default(entry.get("final_score"), 0.0):
        entry["final_score"] = target_score
    target_rank = _int_or_default(deal.get("target_rank"), 999999)
    if target_rank < _int_or_default(entry.get("best_target_rank"), 999999):
        entry["best_target_rank"] = target_rank


def _attach_build_up(entry: dict, deal: dict, target_by_slug: dict[str, dict]) -> None:
    target_slug = str(deal.get("linkedin_slug") or "")
    target_profile = target_by_slug.get(target_slug, {})
    company_name = target_profile.get("company_name") or deal.get("company_name")
    build_up_key = (
        str(deal.get("deal_key") or _deal_key(deal)),
        target_slug,
        company_name,
    )
    if build_up_key in entry["_build_up_keys"]:
        return
    entry["_build_up_keys"].add(build_up_key)
    item = dict(target_profile) if target_profile else {}
    item.update(
        {
            "company_name": company_name,
            "linkedin_slug": target_slug or target_profile.get("linkedin_slug"),
            "deal_key": deal.get("deal_key"),
            "deal_date": deal.get("deal_date"),
            "deal_year": deal.get("deal_year"),
            "deal_type": deal.get("deal_type"),
            "target_rank": deal.get("target_rank"),
        }
    )
    entry["build_up"].append(item)


def _is_entry_under_lbo(entry: dict) -> bool:
    source_deals = entry.get("source_deals") or []
    shareholding = entry.get("shareholding") or []

    has_lbo_like_deal = any(_is_lbo_like_deal(deal) for deal in source_deals)
    has_current_participation = any(
        str(item.get("relationship_type") or "").strip().lower() == "current"
        for item in shareholding
    )
    return has_lbo_like_deal or has_current_participation


def _lookup_acquirer_for_deal(deal: dict, acquirer_index: dict[str, dict]) -> dict | None:
    fund_id = deal.get("fund_id")
    if fund_id is not None and f"fund:{fund_id}" in acquirer_index:
        return acquirer_index[f"fund:{fund_id}"]
    for name in (deal.get("acquirer_name"),):
        normalized = _normalize_name_key(name)
        if normalized and f"name:{normalized}" in acquirer_index:
            return acquirer_index[f"name:{normalized}"]
    return None


def _entity_key(*, fund_id, linkedin_slug, company_name) -> str:
    if linkedin_slug:
        return f"slug:{str(linkedin_slug).lower()}"
    if fund_id is not None:
        return f"fund:{fund_id}"
    normalized = _normalize_name_key(company_name)
    return f"name:{normalized or 'unknown'}"


def _normalize_name_key(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _is_excluded_investor_name(value) -> bool:
    normalized = _normalize_name_key(value).rstrip(".,;:")
    return normalized in EXCLUDED_INVESTOR_NAMES


def _is_build_up_deal(deal: dict) -> bool:
    raw = str(deal.get("deal_type") or "").strip().lower()
    normalized = "".join(ch for ch in raw if ch.isalnum())
    return normalized == "buildup"


def _is_lbo_like_deal(deal: dict) -> bool:
    raw = str(deal.get("deal_type") or "").strip().lower()
    normalized = "".join(ch for ch in raw if ch.isalnum())
    return normalized in {"lbo", "vc", "venturecapital", "equityfundraising"}


def _int_or_default(value, default: int) -> int:
    if value in (None, "", "n.a.", "n.d."):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _float_or_default(value, default: float) -> float:
    if value in (None, "", "n.a.", "n.d."):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
