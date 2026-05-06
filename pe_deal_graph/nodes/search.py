from __future__ import annotations

import asyncio
import logging

from pe_deal_graph.config import MAX_VECTOR_RESULTS
from pe_deal_graph.db.connection import get_pool
from pe_deal_graph.db.sql_templates import (
    HYBRID_SEARCH_PRODUCTS,
    HYBRID_SEARCH_PROFILES,
)
from pe_deal_graph.llm.client import embed_query
from pe_deal_graph.nodes.utils import (
    build_bm25_query,
    escape_tantivy_term,
    normalize_bm25_keywords,
)

logger = logging.getLogger(__name__)


PROFILE_BM25_FIELDS = {
    "company_name": 1.5,
    "one_liner": 1.5,
    "products_keywords": 1.2,
    "description": 1.0,
}

PRODUCT_BM25_FIELDS = {
    "product_name": 2.0,
    "description": 1.0,
}

async def search_hybrid(
    query: str,
    bm25_keywords: list[str] | None = None,
    country_code: str | None = None,
    limit: int = MAX_VECTOR_RESULTS,
) -> dict[str, list[dict]]:
    """Run the hybrid search and return a company shortlist enriched with products."""
    search_query = str(query or "").strip()
    keywords = normalize_bm25_keywords(bm25_keywords)

    if not search_query:
        return {"shortlist": [], "deals_shortlist": [], "acquirers_shortlist": []}

    try:
        embedding = await embed_query(search_query)
    except Exception as exc:
        logger.warning("embedding failed in deal hybrid search: %s", exc)
        embedding = []

    terms = [escape_tantivy_term(keyword) for keyword in keywords]
    fallback = escape_tantivy_term(search_query)
    profile_bm25 = build_bm25_query(terms, PROFILE_BM25_FIELDS, fallback=fallback)
    product_bm25 = build_bm25_query(terms, PRODUCT_BM25_FIELDS, fallback=fallback)

    emb = embedding or [0.0] * 1536

    pool = await get_pool()
    normalized_country_code = _normalize_country_code(country_code)
    filter_params: list[str] = []
    profile_filters = ""
    product_company_filters = ""
    if normalized_country_code:
        profile_filters = " AND $4 = ANY(country_codes)"
        product_company_filters = " AND $4 = ANY(cwd.country_codes)"
        filter_params.append(normalized_country_code)

    profile_sql = HYBRID_SEARCH_PROFILES.format(filters=profile_filters)
    product_sql = HYBRID_SEARCH_PRODUCTS.format(filters="", company_filters=product_company_filters)

    profile_rows, product_rows = await asyncio.gather(
        pool.fetch(profile_sql, str(emb), profile_bm25, limit, *filter_params),
        pool.fetch(product_sql, str(emb), product_bm25, limit, *filter_params),
    )

    by_id: dict[str, dict] = {}
    for row in profile_rows:
        linkedin_slug = row["linkedin_slug"]
        entry = dict(row)
        vrank = row.get("vrank")
        brank = row.get("brank")
        if vrank is not None and brank is not None:
            entry["_match_source"] = "profile:vector+bm25"
        elif vrank is not None:
            entry["_match_source"] = "profile:vector"
        else:
            entry["_match_source"] = "profile:bm25"
        by_id[linkedin_slug] = entry

    for row in product_rows:
        linkedin_slug = row["linkedin_slug"]
        vrank = row.get("vrank")
        brank = row.get("brank")
        if vrank is not None and brank is not None:
            source = "product:vector+bm25"
        elif vrank is not None:
            source = "product:vector"
        else:
            source = "product:bm25"

        if linkedin_slug in by_id:
            by_id[linkedin_slug]["rrf_score"] = float(by_id[linkedin_slug].get("rrf_score", 0)) + float(row["rrf_score"])
            by_id[linkedin_slug]["_match_source"] += f" + {source}"
        else:
            by_id[linkedin_slug] = dict(row)
            by_id[linkedin_slug]["_match_source"] = source

    shortlist = sorted(by_id.values(), key=lambda row: row.get("rrf_score", 0), reverse=True)[:limit]
    if not shortlist:
        return {"shortlist": [], "deals_shortlist": [], "acquirers_shortlist": []}

    ids = [row["linkedin_slug"] for row in shortlist if row.get("linkedin_slug")]
    if not ids:
        return {"shortlist": shortlist, "deals_shortlist": [], "acquirers_shortlist": []}

    rows = await pool.fetch(
        "SELECT linkedin_slug, product_name, product_type, description "
        "FROM company_products WHERE linkedin_slug = ANY($1)",
        ids,
    )
    by_product_id: dict[str, list] = {}
    for row in rows:
        by_product_id.setdefault(row["linkedin_slug"], []).append({
            "Product category": row["product_name"],
            "Type": row["product_type"],
            "Description": row["description"],
        })

    for entry in shortlist:
        linkedin_slug = entry.get("linkedin_slug")
        if linkedin_slug and linkedin_slug in by_product_id:
            entry["products_services"] = by_product_id[linkedin_slug]

    return {
        "shortlist": shortlist,
        "deals_shortlist": [],
        "acquirers_shortlist": [],
    }


def _normalize_country_code(value) -> str | None:
    if not isinstance(value, str):
        return None
    code = value.strip().upper()
    if len(code) != 2 or not code.isalpha():
        return None
    return code
