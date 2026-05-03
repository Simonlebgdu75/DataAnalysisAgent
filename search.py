from __future__ import annotations

import asyncio
import logging
import math
import re

from pe_qa_graph.config import MAX_VECTOR_RESULTS
from shared.db.connection import get_pool
from shared.db.sql_templates import HYBRID_SEARCH_PRODUCTS, HYBRID_SEARCH_PROFILES
from pe_qa_graph.llm.client import embed
from pe_qa_graph.state import AgentState
from pe_qa_graph.stream import emit_metric, emit_phase

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────


def _get_search_query(state: AgentState) -> str:
    return state.get("search_query", "")


def _has_shortlist_filter_intent(state: AgentState) -> bool:
    return state.get("intent") == "filter" and bool(state.get("shortlist"))


def _normalize_keyword(value: str) -> str:
    return " ".join(value.strip().split())


def _get_bm25_keywords(state: AgentState, limit: int = 8) -> list[str]:
    """Get high-signal keywords produced by the agent search plan."""
    raw_keywords = state.get("bm25_keywords") or []
    out: list[str] = []
    seen: set[str] = set()

    if isinstance(raw_keywords, list):
        for kw in raw_keywords:
            if not isinstance(kw, str):
                continue
            normalized = _normalize_keyword(kw)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(normalized)
            if len(out) >= limit:
                break

    return out


# Escape punctuation defensively before feeding raw user terms to Tantivy/ParadeDB.
# This covers the documented query-string specials and extra punctuation such as "&"
# that can appear in market acronyms like "S&OP".
_TANTIVY_SPECIAL = re.compile(r"""[^\w\s]""")


def _sanitize_bm25_token(term: str) -> str:
    """Escape punctuation in a single Tantivy token."""
    return _TANTIVY_SPECIAL.sub(r"\\\g<0>", term)


def _sanitize_bm25_term(term: str) -> str:
    """Build a Tantivy-safe term or grouped multi-word clause."""
    normalized = _normalize_keyword(term)
    if not normalized:
        return ""

    tokens = [_sanitize_bm25_token(token) for token in normalized.split() if token]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return f"({' '.join(tokens)})"


# Field boosts for profile BM25 search (Tantivy syntax: field:term^boost)
PROFILE_BM25_FIELDS = {
    "company_name": 2.0,
    "one_liner": 1.5,
    "products_keywords": 1.2,
    "description": 1.0,
}

# Field boosts for product BM25 search
PRODUCT_BM25_FIELDS = {
    "product_name": 2.0,
    "description": 1.0,
}


def _build_tantivy_query(keywords: list[str], fields: dict[str, float]) -> str:
    """Build a Tantivy query string with field-level boosts.

    Example output: '(company_name:fintech)^2 OR description:fintech'
    """
    terms = [_sanitize_bm25_term(kw.strip()) for kw in keywords if isinstance(kw, str) and kw.strip()]
    if not terms:
        return ""
    clauses: list[str] = []
    for term in terms:
        for field, boost in fields.items():
            if boost != 1.0:
                clauses.append(f"({field}:{term})^{boost}")
            else:
                clauses.append(f"{field}:{term}")
    return " OR ".join(clauses)


def _build_bm25_query_from_keywords(keywords: list[str]) -> str:
    """Build Tantivy query for profile search with field boosts."""
    return _build_tantivy_query(keywords, PROFILE_BM25_FIELDS)


def _build_product_bm25_query_from_keywords(keywords: list[str]) -> str:
    """Build Tantivy query for product search with field boosts."""
    return _build_tantivy_query(keywords, PRODUCT_BM25_FIELDS)


def _row_text_for_shortlist_bm25(row: dict) -> str:
    fields = (
        row.get("company_name", ""),
        row.get("one_liner", ""),
        row.get("products_keywords", ""),
        row.get("description", ""),
        row.get("sector", ""),
        row.get("business_model", ""),
        row.get("linkedin_headquarters", ""),
    )
    return " ".join(str(v or "") for v in fields).lower()


def _bm25_rank_shortlist(rows: list[dict], keywords: list[str], limit: int = MAX_VECTOR_RESULTS) -> list[dict]:
    terms = [kw.lower() for kw in keywords if isinstance(kw, str) and kw.strip()]
    if not rows:
        return []
    if not terms:
        return rows[:limit]

    scored: list[dict] = []
    for row in rows:
        text = _row_text_for_shortlist_bm25(row)
        score = 0.0
        for t in terms:
            if t in text:
                score += 1.0
        enriched = dict(row)
        enriched["rank"] = score
        scored.append(enriched)

    scored.sort(key=lambda r: float(r.get("rank", 0.0)), reverse=True)
    return scored[:limit]


def _apply_shortlist_filters(shortlist: list[dict], filters: dict | None) -> list[dict]:
    """Apply the currently supported structured filters in-memory on the shortlist."""
    if not shortlist or not isinstance(filters, dict):
        return shortlist

    employee_filter = filters.get("linkedin_employees")
    pe_backed_filter = _coerce_bool(filters.get("is_pe_backed"))
    country_code = filters.get("country_code")

    filtered = shortlist
    if isinstance(employee_filter, dict):
        filtered = [
            row
            for row in filtered
            if _matches_employee_filter(row.get("linkedin_employees"), employee_filter)
        ]

    if pe_backed_filter is not None:
        filtered = [
            row
            for row in filtered
            if _matches_bool_filter(row.get("is_pe_backed"), pe_backed_filter)
        ]

    if isinstance(country_code, str) and country_code.strip():
        expected = country_code.strip().upper()
        filtered = [
            row
            for row in filtered
            if _matches_country_code(row.get("country_codes"), expected)
        ]

    # HQ country name filter (headquarters-based)
    hq_country_names = filters.get("hq_country_names")
    if isinstance(hq_country_names, list) and hq_country_names:
        name_set = {n.lower() for n in hq_country_names}
        filtered = [
            row for row in filtered
            if (row.get("hq_country_name") or "").lower() in name_set
        ]

    # Multi-city bounding boxes (resolved by geo_resolver)
    geo_cities = filters.get("geo_cities")
    if isinstance(geo_cities, list) and geo_cities:
        def _in_any_bbox(row: dict) -> bool:
            lat, lon = row.get("hq_lat"), row.get("hq_lon")
            if lat is None or lon is None:
                return False
            for city in geo_cities:
                if not isinstance(city, dict) or city.get("lat") is None:
                    continue
                r = city.get("radius_km", 30)
                dlat = r * 0.009
                dlon = r * 0.009 / max(math.cos(math.radians(city["lat"])), 0.1)
                if city["lat"] - dlat <= lat <= city["lat"] + dlat and city["lon"] - dlon <= lon <= city["lon"] + dlon:
                    return True
            return False
        filtered = [row for row in filtered if _in_any_bbox(row)]

    # Multi-region (resolved by geo_resolver)
    hq_regions = filters.get("hq_regions")
    if isinstance(hq_regions, list) and hq_regions:
        region_set = {r.lower() for r in hq_regions}
        filtered = [
            row for row in filtered
            if (row.get("hq_region") or "").lower() in region_set
        ]

    return filtered


def _coerce_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        digits = re.findall(r"\d+", stripped)
        if digits:
            return int(digits[0])
    return None


def _coerce_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
    return None


def _matches_employee_filter(value, employee_filter: dict) -> bool:
    count = _coerce_int(value)
    target = _coerce_int(employee_filter.get("value"))
    operator = str(employee_filter.get("operator") or "").lower()

    if count is None or target is None:
        return False

    if operator == "eq":
        return count == target
    if operator == "gt":
        return count > target
    if operator == "gte":
        return count >= target
    if operator == "lt":
        return count < target
    if operator == "lte":
        return count <= target
    return False


def _matches_bool_filter(value, expected: bool) -> bool:
    actual = _coerce_bool(value)
    return actual is not None and actual == expected


def _matches_country_code(value, expected: str) -> bool:
    if not isinstance(expected, str) or not expected:
        return True
    if value is None:
        return False
    if isinstance(value, list):
        return expected in [v.strip().upper() for v in value if isinstance(v, str)]
    # Fallback for string values
    return expected in [v.strip().upper() for v in str(value).split(",") if v.strip()]


def _is_country_only_shortlist_filter(state: AgentState) -> bool:
    if not _has_shortlist_filter_intent(state):
        return False
    filters = state.get("filters") or {}
    return isinstance(filters, dict) and set(filters.keys()) == {"country_code"}


def _build_hybrid_filters(state: AgentState) -> tuple[str, str, list]:
    """Build SQL filter clauses for hybrid queries.

    Returns (profile_filters, product_filters, params).
    - profile_filters: inlined in vector_hits/bm25_hits WHERE clauses (uses HNSW iterative scan)
    - product_filters: applied as WHERE on the final JOIN (post-HNSW, on ~50 rows)
    """
    filters = state.get("filters") or {}
    if not isinstance(filters, dict) or not filters:
        return "", "", []

    profile_conditions, filter_params = _build_company_filter_conditions(filters, param_offset=4)
    if not profile_conditions:
        return "", "", []

    # Profiles: inline AND clauses, no alias (pgvector 0.8 iterative index scan)
    profile_sql = " AND " + " AND ".join(profile_conditions)
    # Products: post-filter on final JOIN with cwd alias (HNSW top-2000 unfiltered, filter ~200 fused rows)
    product_conditions, _ = _build_company_filter_conditions(filters, param_offset=4, table_alias="cwd")
    product_sql = "WHERE " + " AND ".join(product_conditions)
    return profile_sql, product_sql, filter_params


def _build_company_filter_conditions(
    filters: dict,
    *,
    param_offset: int,
    table_alias: str = "",
) -> tuple[list[str], list]:
    """Build WHERE clauses for company_linkedin_data filters.

    If table_alias is set (e.g. "cwd"), columns are prefixed: cwd.linkedin_employees.
    """
    clauses: list[str] = []
    params: list = []
    idx = param_offset
    prefix = f"{table_alias}." if table_alias else ""

    employee_filter = filters.get("linkedin_employees")
    if isinstance(employee_filter, dict):
        target = _coerce_int(employee_filter.get("value"))
        operator = str(employee_filter.get("operator") or "").lower()
        sql_operator = {
            "eq": "=",
            "gt": ">",
            "gte": ">=",
            "lt": "<",
            "lte": "<=",
        }.get(operator)
        if target is not None and sql_operator is not None:
            clauses.append(f"{prefix}linkedin_employees {sql_operator} ${idx}")
            params.append(target)
            idx += 1

    pe_backed_filter = _coerce_bool(filters.get("is_pe_backed"))
    if pe_backed_filter is not None:
        clauses.append(f"{prefix}is_pe_backed = ${idx}")
        params.append(pe_backed_filter)
        idx += 1

    country_code = filters.get("country_code")
    if isinstance(country_code, str) and country_code.strip():
        clauses.append(f"${idx} = ANY({prefix}country_codes)")
        params.append(country_code.strip().upper())
        idx += 1

    # HQ country name filter (headquarters-based, not presence-based)
    hq_country_names = filters.get("hq_country_names")
    if isinstance(hq_country_names, list) and hq_country_names:
        clauses.append(f"{prefix}hq_country_name = ANY(${idx}::text[])")
        params.append(hq_country_names)
        idx += 1

    # Multi-city geo bounding boxes (resolved by geo_resolver node)
    geo_cities = filters.get("geo_cities")
    if isinstance(geo_cities, list) and geo_cities:
        city_clauses = []
        for city in geo_cities:
            if not isinstance(city, dict) or city.get("lat") is None or city.get("lon") is None:
                continue
            radius_km = city.get("radius_km", 30)
            dlat = radius_km * 0.009
            dlon = radius_km * 0.009 / max(math.cos(math.radians(city["lat"])), 0.1)
            city_clauses.append(
                f"({prefix}hq_lat BETWEEN ${idx} AND ${idx + 1} AND {prefix}hq_lon BETWEEN ${idx + 2} AND ${idx + 3})"
            )
            params.extend([city["lat"] - dlat, city["lat"] + dlat, city["lon"] - dlon, city["lon"] + dlon])
            idx += 4
        if city_clauses:
            clauses.append(f"({' OR '.join(city_clauses)})")

    # Multi-region filter (resolved by geo_resolver node)
    hq_regions = filters.get("hq_regions")
    if isinstance(hq_regions, list) and hq_regions:
        clauses.append(f"{prefix}hq_region = ANY(${idx}::text[])")
        params.append(hq_regions)
        idx += 1

    return clauses, params


# ── Node: Embed ──────────────────────────────────────────────────────


async def search_embed(state: AgentState) -> dict:
    """Embed the search query (OpenAI API call)."""
    if _has_shortlist_filter_intent(state):
        emit_phase("search_embed", "skipped", reason="shortlist_filter")
        return {"_embeddings": []}

    query = _get_search_query(state)
    if not query:
        emit_phase("search_embed", "complete", embedding_count=0)
        return {"_embeddings": [], "errors": ["No search query"]}

    emit_phase("search_embed", "start", search_query=query)
    try:
        embeddings = await embed([query])
    except Exception as e:
        emit_phase("search_embed", "complete", embedding_count=0)
        return {"_embeddings": [], "errors": [f"Embedding error: {e}"]}

    emit_metric("search_embed", embedding_count=len(embeddings))
    emit_phase("search_embed", "complete", embedding_count=len(embeddings))
    return {"_embeddings": embeddings}


# ── Enrich shortlist with products from company_products table ────────


async def _enrich_with_products(shortlist: list[dict], pool) -> list[dict]:
    """Attach products_services JSONB to each shortlist row via a single query."""
    if not shortlist:
        return shortlist
    ids = [r["linkedin_slug"] for r in shortlist if r.get("linkedin_slug")]
    if not ids:
        return shortlist
    rows = await pool.fetch(
        "SELECT linkedin_slug, product_name, product_type, description "
        "FROM company_products WHERE linkedin_slug = ANY($1)",
        ids,
    )
    by_id: dict[str, list] = {}
    for r in rows:
        by_id.setdefault(r["linkedin_slug"], []).append({
            "Product category": r["product_name"],
            "Type": r["product_type"],
            "Description": r["description"],
        })
    for entry in shortlist:
        lid = entry.get("linkedin_slug")
        if lid and lid in by_id:
            entry["products_services"] = by_id[lid]
    return shortlist


# ── Node: Hybrid search (vector + BM25 + RRF in SQL) ────────────────


async def search_hybrid(state: AgentState) -> dict:
    """Single-query hybrid search: vector + ParadeDB BM25 fused with RRF in SQL."""

    # ── Shortlist filter path ──────────────────────────────────────────
    if _has_shortlist_filter_intent(state):
        shortlist = state.get("shortlist", [])
        filters = state.get("filters") or {}
        # Geo filters are already resolved by the geo_resolver node
        filtered = _apply_shortlist_filters(shortlist, filters)
        if _is_country_only_shortlist_filter(state):
            logger.info(
                "shortlist country filter path: input=%s filtered=%s preserved_order=true",
                len(shortlist), len(filtered),
            )
            emit_metric(
                "search_hybrid",
                source="shortlist_country_filter",
                input_count=len(shortlist),
                result_count=len(filtered),
            )
            emit_phase(
                "search_hybrid",
                "complete",
                shortlist_count=len(filtered),
                source="shortlist_country_filter",
            )
            return {"shortlist": filtered}
        bm25_keywords = _get_bm25_keywords(state)
        ranked = _bm25_rank_shortlist(filtered, bm25_keywords, limit=MAX_VECTOR_RESULTS)
        logger.info(
            "shortlist filter path: input=%s filtered=%s ranked=%s",
            len(shortlist), len(filtered), len(ranked),
        )
        emit_metric("search_hybrid", source="shortlist", input_count=len(shortlist), result_count=len(ranked))
        emit_phase("search_hybrid", "complete", shortlist_count=len(ranked), source="shortlist")
        return {"shortlist": ranked}

    # ── Normal hybrid path ───────────────────────────────────────────
    embeddings = state.get("_embeddings", [])
    bm25_keywords = _get_bm25_keywords(state)
    profile_bm25 = _build_bm25_query_from_keywords(bm25_keywords)
    product_bm25 = _build_product_bm25_query_from_keywords(bm25_keywords)

    if (not embeddings or not embeddings[0]) and not profile_bm25:
        emit_phase("search_hybrid", "complete", shortlist_count=0)
        return {"shortlist": [], "errors": ["No embedding and no BM25 query"]}

    emit_phase("search_hybrid", "start", keyword_count=len(bm25_keywords))

    emb = embeddings[0] if embeddings else [0.0] * 1536
    if not profile_bm25:
        # Fallback: build Tantivy query from the raw search query
        raw_query = _get_search_query(state)
        fallback_kws = [raw_query] if raw_query else []
        profile_bm25 = _build_tantivy_query(fallback_kws, PROFILE_BM25_FIELDS)
        product_bm25 = _build_tantivy_query(fallback_kws, PRODUCT_BM25_FIELDS)

    # No-match placeholder if still empty
    effective_profile_bm25 = profile_bm25 or "xyzzy_no_match_98765"
    effective_product_bm25 = product_bm25 or "xyzzy_no_match_98765"

    logger.info("BM25 profile query: %s", effective_profile_bm25[:200])
    logger.info("BM25 product query: %s", effective_product_bm25[:200])

    pool = await get_pool("linkedin")

    # Geo filters are already resolved by the geo_resolver node (runs in parallel with search_embed)
    profile_filters, product_filters, filter_params = _build_hybrid_filters(state)
    limit = MAX_VECTOR_RESULTS

    # Run profiles + products hybrid in parallel
    # Profiles: filters inlined in WHERE (HNSW iterative scan)
    # Products: HNSW runs unfiltered, filters applied post-fusion on ~50 rows
    profile_sql = HYBRID_SEARCH_PROFILES.format(filters=profile_filters)
    product_sql = HYBRID_SEARCH_PRODUCTS.format(filters=product_filters)

    profile_rows, product_rows = await asyncio.gather(
        pool.fetch(profile_sql, str(emb), effective_profile_bm25, limit, *filter_params),
        pool.fetch(product_sql, str(emb), effective_product_bm25, limit, *filter_params),
    )

    # Merge by linkedin_slug: sum RRF scores across profile + product searches
    by_id: dict[str, dict] = {}
    for row in profile_rows:
        rid = row["linkedin_slug"]
        entry = dict(row)
        vrank = row.get("vrank")
        brank = row.get("brank")
        if vrank is not None and brank is not None:
            entry["_match_source"] = "profile:vector+bm25"
        elif vrank is not None:
            entry["_match_source"] = "profile:vector"
        else:
            entry["_match_source"] = "profile:bm25"
        by_id[rid] = entry
    for row in product_rows:
        rid = row["linkedin_slug"]
        vrank = row.get("vrank")
        brank = row.get("brank")
        if vrank is not None and brank is not None:
            source = "product:vector+bm25"
        elif vrank is not None:
            source = "product:vector"
        else:
            source = "product:bm25"
        if rid in by_id:
            # Company in both: sum scores and track both sources
            by_id[rid]["rrf_score"] = float(by_id[rid].get("rrf_score", 0)) + float(row["rrf_score"])
            by_id[rid]["_match_source"] += f" + {source}"
        else:
            by_id[rid] = dict(row)
            by_id[rid]["_match_source"] = source

    shortlist = sorted(by_id.values(), key=lambda r: r.get("rrf_score", 0), reverse=True)[:limit]

    logger.info(
        "hybrid search: profiles=%s products=%s merged=%s (cap=%s)",
        len(profile_rows), len(product_rows), len(shortlist), limit,
    )
    emit_metric(
        "search_hybrid",
        profile_count=len(profile_rows),
        product_count=len(product_rows),
        merged_count=len(shortlist),
        cap=limit,
    )
    # Enrich with products from company_products table
    shortlist = await _enrich_with_products(shortlist, pool)

    emit_phase("search_hybrid", "complete", shortlist_count=len(shortlist))
    return {"shortlist": shortlist}
