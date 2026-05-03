from __future__ import annotations

import asyncio
import json
import logging

from shared.db.connection import get_pool
from pe_qa_graph.llm.client import chat_json
from pe_qa_graph.llm.prompts import (
    build_rerank_system_prompt,
    build_rerank_user_prompt,
    normalize_rerank_mode,
)
from pe_qa_graph.state import AgentState
from pe_qa_graph.stream import emit_error, emit_metric, emit_phase

logger = logging.getLogger(__name__)
BATCH_SIZE = 15  # companies per LLM call
RERANK_MAX_ATTEMPTS = 3
RERANK_RETRY_BASE_DELAY_SECONDS = 0.75



async def rerank(state: AgentState) -> dict:
    """LLM-based per-result relevance verification, aligned with user's full context."""
    results = state.get("shortlist", [])
    search_context = state.get("question", "")
    rerank_mode = normalize_rerank_mode(state.get("rerank_mode"))
    errors: list[str] = []

    if not results:
        emit_phase("rerank", "complete", shortlist_count=0, batch_count=0)
        return {"shortlist": [], "errors": ["No results to rerank"]}

    if state.get("skip_rerank"):
        emit_metric(
            "rerank",
            input_count=len(results),
            kept_count=len(results),
            discarded_count=0,
            batch_count=0,
            rerank_mode="skipped",
            status="country_filter_only",
        )
        emit_phase(
            "rerank",
            "complete",
            shortlist_count=len(results),
            batch_count=0,
            rerank_mode="skipped",
            reason="country_filter_only",
        )
        return {
            "shortlist": results,
            "errors": [],
        }

    # Split into batches and process all results in parallel for every rerank mode.
    batches = [results[i : i + BATCH_SIZE] for i in range(0, len(results), BATCH_SIZE)]
    emit_phase(
        "rerank",
        "start",
        shortlist_count=len(results),
        batch_count=len(batches),
        rerank_mode=rerank_mode,
    )
    tasks = [
        _rerank_batch(batch, batch_index, len(batches), search_context, rerank_mode, errors)
        for batch_index, batch in enumerate(batches, start=1)
    ]
    batch_results = await asyncio.gather(*tasks)

    # Collect kept IDs → reason mapping
    keep_reasons: dict[str, str] = {}
    for kept in batch_results:
        keep_reasons.update(kept)

    # Filter results, preserving RRF order, and inject rerank reason
    filtered = []
    for r in results:
        rid = r.get("linkedin_slug")
        if rid in keep_reasons:
            enriched = dict(r)
            enriched["rerank_reason"] = keep_reasons[rid]
            filtered.append(enriched)

    # Log provenance of each kept company
    logger.info("=== RERANK RESULTS: %d companies kept ===", len(filtered))
    for i, r in enumerate(filtered, 1):
        source = r.get("_match_source", "unknown")
        logger.info(
            "  #%d | %s (id=%s) | rrf=%.4f | source=%s",
            i,
            r.get("company_name", "?"),
            r.get("linkedin_slug"),
            float(r.get("rrf_score") or 0),
            source,
        )

    emit_metric(
        "rerank",
        input_count=len(results),
        kept_count=len(filtered),
        discarded_count=max(len(results) - len(filtered), 0),
        batch_count=len(batches),
        rerank_mode=rerank_mode,
        error_count=len(errors),
    )
    emit_phase(
        "rerank",
        "complete",
        shortlist_count=len(filtered),
        batch_count=len(batches),
        rerank_mode=rerank_mode,
    )

    filtered = await _enrich_shortlist(filtered)

    return {
        "shortlist": filtered,
        "errors": errors if errors else [],
    }


async def _rerank_batch(
    batch: list[dict],
    batch_index: int,
    total_batches: int,
    search_context: str,
    rerank_mode: str,
    errors: list[str],
) -> dict[str, str]:
    """Rerank a batch of results, return {linkedin_slug: reason} for kept items."""
    emit_metric(
        "rerank",
        batch_index=batch_index,
        total_batches=total_batches,
        batch_size=len(batch),
        rerank_mode=rerank_mode,
        status="start",
    )
    def _fmt_revenue(r: dict) -> str:
        """Compact revenue hint for rerank LLM context. Source-aware."""
        rev = r.get("revenue")
        src = r.get("revenue_source")
        if rev is not None:
            try:
                v = float(rev)
            except (ValueError, TypeError):
                return "n/a"
            if v >= 1e9: fmt = f"{v/1e9:.1f}Md"
            elif v >= 1e6: fmt = f"{v/1e6:.1f}M"
            elif v >= 1e3: fmt = f"{v/1e3:.0f}K"
            else: fmt = f"{v:.0f}"
            # source trust : yfinance/fund_portfolio = precise, INPI = ±22%
            if src in ("inpi_consolidated", "inpi"):
                return f"{fmt}€ (INPI ±22%)"
            return f"{fmt}€ ({src or 'real'})"
        # Fallback: ML estimate P35-P65 (MAPE 49%, fourchette 1.7x)
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
        f"- ID={r.get('linkedin_slug')} | {r.get('company_name', '?')} | "
        f"{r.get('one_liner', '') or 'No description'} | "
        f"Revenue: {_fmt_revenue(r)} | "
        f"Products: {r.get('products_keywords', '') or 'n/a'} | "
        f"Products/Services: {_format_products_services(r)}"
        for r in batch
    )

    system_prompt = build_rerank_system_prompt(rerank_mode)
    user_prompt = build_rerank_user_prompt(
        search_context=search_context or "(none)",
        companies=companies_text,
        rerank_mode=rerank_mode,
    )

    result: dict | None = None
    last_error: Exception | None = None
    for attempt in range(1, RERANK_MAX_ATTEMPTS + 1):
        try:
            result = await chat_json(
                system=system_prompt,
                user=user_prompt,
                model="gpt-4.1-mini",
            )
            break
        except Exception as e:
            last_error = e
            if attempt >= RERANK_MAX_ATTEMPTS:
                break

            delay = RERANK_RETRY_BASE_DELAY_SECONDS * attempt
            logger.warning(
                "rerank batch failed, retrying",
                extra={
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "attempt": attempt,
                    "max_attempts": RERANK_MAX_ATTEMPTS,
                    "delay_seconds": delay,
                    "error": str(e),
                },
            )
            emit_metric(
                "rerank",
                batch_index=batch_index,
                total_batches=total_batches,
                batch_size=len(batch),
                rerank_mode=rerank_mode,
                attempt=attempt,
                max_attempts=RERANK_MAX_ATTEMPTS,
                status="retry",
            )
            await asyncio.sleep(delay)

    if result is None:
        message = f"Rerank error: {last_error}" if last_error is not None else "Rerank error: unknown error"
        errors.append(message)
        emit_error("rerank", message, batch_index=batch_index, total_batches=total_batches)
        emit_metric(
            "rerank",
            batch_index=batch_index,
            total_batches=total_batches,
            batch_size=len(batch),
            kept_count=len(batch),
            rerank_mode=rerank_mode,
            attempt=RERANK_MAX_ATTEMPTS,
            max_attempts=RERANK_MAX_ATTEMPTS,
            status="fallback_keep_all",
        )
        # On error, keep all results from this batch
        return {r["linkedin_slug"]: "rerank error – kept by default" for r in batch if r.get("linkedin_slug") is not None}

    decisions = result.get("decisions", [])
    keep_reasons: dict[str, str] = {}
    for d in decisions:
        if d.get("verdict") == "keep":
            rid = d.get("id")
            if rid is not None:
                keep_reasons[str(rid)] = d.get("reason", "")

    emit_metric(
        "rerank",
        batch_index=batch_index,
        total_batches=total_batches,
        batch_size=len(batch),
        kept_count=len(keep_reasons),
        rerank_mode=rerank_mode,
        status="complete",
    )

    return keep_reasons


async def _enrich_shortlist(shortlist: list[dict]) -> list[dict]:
    """Light enrichment for shortlist table display.

    Only adds revenue (formatted) for the table.
    Full details (actes, legal, dirigeants, fund_info, products) are
    fetched on demand via the Company Data API when a user clicks a row.
    """
    pool = await get_pool("linkedin")

    from shared.db.funds import enrich_shortlist_light
    shortlist = await enrich_shortlist_light(shortlist, pool)

    return shortlist


def _format_products_services(r: dict) -> str:
    """Serialize products_services into text for reranking."""
    value = r.get("products_services")
    if not value:
        return "n/a"

    # products_services is often stored as JSON string in DB rows
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                name = (
                    item.get("Product category")
                    or item.get("Product name (exact)")
                    or item.get("Product name")
                    or ""
                )
                desc = item.get("Description") or ""
                if name and name != "n.d.":
                    parts.append(f"{name}: {desc}" if desc else name)
            else:
                parts.append(str(item))
        text = "; ".join(parts)
        return text if text else "n/a"

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    return str(value)
