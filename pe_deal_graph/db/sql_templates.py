from __future__ import annotations

RRF_K = 60.0
VECTOR_WEIGHT = 1.5
BM25_WEIGHT = 0.7
VECTOR_DISTANCE_THRESHOLD = 0.90
BM25_SCORE_THRESHOLD = 1.0

# ── Column list shared by all queries (company_linkedin_data) ──────────
_CWD_COLUMNS = """\
cwd.linkedin_slug, cwd.company_name, cwd.sector, cwd.business_model, cwd.b2b_b2c,
       cwd.one_liner, cwd.description, cwd.products_keywords,
       cwd.kpis, cwd.contact, cwd.linkedin_headquarters, cwd.linkedin_company_size,
       cwd.linkedin_employees, cwd.linkedin_website,
       cwd.linkedin_industries, cwd.linkedin_about,
       cwd.specialties, cwd.organization_type, cwd.founded, cwd.funding, cwd.investors,
       cwd.country_codes, cwd.formatted_locations, cwd.slogan, cwd.crunchbase_url,
       cwd.linkedin_url, cwd.logo, cwd.similar_companies,
       cwd.management, cwd.siren, cwd.is_pe_backed, cwd.has_deal,
       cwd.revenue, cwd.revenue_source,
       cel.p35 AS revenue_est_p35, cel.p50 AS revenue_est_p50,
       cel.p65 AS revenue_est_p65, cel.bucket_p50 AS revenue_est_bucket"""

DEAL_COLUMNS = """\
d.id, d.company_name, d.linkedin_slug, d.country,
d.acquirer_name, d.fund_id, d.deal_type, d.deal_date, d.deal_year,
d.deal_date_parsed, d.deal_month,
d.description, d.source_url, d.website, d.source_type, d.loaded_at"""

ACQUIRER_COLUMNS = """\
f.id AS fund_id, f.name AS acquirer_name, f.linkedin_slug AS acquirer_linkedin_slug,
f.aum_amount, f.aum_currency, f.aum_year, f.funds_raised, f.strategies,
cwd.company_name AS linkedin_company_name, cwd.logo, cwd.one_liner, cwd.description,
cwd.sector, cwd.linkedin_headquarters, cwd.linkedin_employees, cwd.linkedin_website,
cwd.country_codes, cwd.formatted_locations, cwd.organization_type, cwd.specialties"""

SELECT_ACQUIRERS_FOR_DEALS = """\
WITH deal_fund_links AS (
    SELECT d.id AS deal_id, df.fund_id
    FROM deal d
    JOIN deal_fund df ON df.deal_id = d.id
    WHERE d.id = ANY($1::int[])
    UNION
    SELECT d.id AS deal_id, d.fund_id
    FROM deal d
    WHERE d.id = ANY($1::int[])
      AND d.fund_id IS NOT NULL
)
SELECT l.deal_id,
       {columns}
FROM deal_fund_links l
JOIN fund f ON f.id = l.fund_id
LEFT JOIN company_linkedin_data cwd ON cwd.linkedin_slug = f.linkedin_slug
ORDER BY l.deal_id, f.name
""".format(columns=ACQUIRER_COLUMNS)

SELECT_INVESTORS_FOR_TARGETS = """\
SELECT fp.linkedin_slug AS target_linkedin_slug,
       fp.company_name AS target_company_name,
       fp.relationship_type,
       fp.entry_year,
       fp.exit_year,
       -- Target revenue read from the canonical pre-computed column
       -- (yfinance > INPI consolidated > fund_portfolio > INPI ind).
       -- Always EUR since populate_revenue normalizes everything.
       target_cwd.revenue AS latest_revenue_amount,
       CASE WHEN target_cwd.revenue IS NOT NULL THEN 'EUR' END AS latest_revenue_currency,
       CASE WHEN target_cwd.revenue_source = 'fund_portfolio'
            THEN fp.latest_revenue_year END AS latest_revenue_year,
       target_cwd.revenue_source AS revenue_source,
       {columns}
FROM fund_portfolio fp
JOIN fund f ON f.id = fp.fund_id
LEFT JOIN company_linkedin_data cwd ON cwd.linkedin_slug = f.linkedin_slug
LEFT JOIN company_linkedin_data target_cwd ON target_cwd.linkedin_slug = fp.linkedin_slug
WHERE fp.linkedin_slug = ANY($1::text[])
ORDER BY fp.linkedin_slug, fp.relationship_type, fp.entry_year DESC NULLS LAST, f.name
""".format(columns=ACQUIRER_COLUMNS)

# ── Hybrid search: vector + ParadeDB BM25 + RRF in a single query ────
# $1 = embedding vector, $2 = BM25 query text, $3 = limit

HYBRID_SEARCH_PROFILES = """\
WITH vector_hits AS (
    SELECT linkedin_slug,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS vrank
    FROM company_linkedin_data
    WHERE (has_deal = TRUE OR is_pe_backed = TRUE)
      AND embedding IS NOT NULL
      AND (embedding <=> $1::vector) < {vector_dist_thresh}
      {{filters}}
    ORDER BY embedding <=> $1::vector
    LIMIT $3
),
bm25_hits AS (
    SELECT linkedin_slug,
           ROW_NUMBER() OVER (ORDER BY pdb.score(linkedin_slug) DESC) AS brank
    FROM company_linkedin_data
    WHERE (has_deal = TRUE OR is_pe_backed = TRUE)
      AND linkedin_slug @@@ pdb.parse($2, lenient => true)
      AND pdb.score(linkedin_slug) > {bm25_thresh}
      {{filters}}
    ORDER BY pdb.score(linkedin_slug) DESC
    LIMIT $3
),
fused AS (
    SELECT COALESCE(v.linkedin_slug, b.linkedin_slug) AS linkedin_slug,
           COALESCE({vec_w}/({rrf_k}+v.vrank), 0) + COALESCE({bm25_w}/({rrf_k}+b.brank), 0) AS rrf_score,
           CASE WHEN v.linkedin_slug IS NOT NULL AND b.linkedin_slug IS NOT NULL THEN 2 ELSE 1 END AS overlap,
           LEAST(COALESCE(v.vrank, 999999), COALESCE(b.brank, 999999)) AS best_rank,
           v.vrank, b.brank
    FROM vector_hits v
    FULL OUTER JOIN bm25_hits b USING (linkedin_slug)
)
SELECT {columns},
       f.rrf_score, f.vrank, f.brank
FROM fused f
JOIN company_linkedin_data cwd USING (linkedin_slug)
LEFT JOIN ca_estimates_linkedin cel ON cel.linkedin_slug = cwd.linkedin_slug
ORDER BY f.rrf_score DESC, f.overlap DESC, f.best_rank ASC
LIMIT $3
""".format(
    rrf_k=RRF_K,
    vec_w=VECTOR_WEIGHT,
    bm25_w=BM25_WEIGHT,
    vector_dist_thresh=VECTOR_DISTANCE_THRESHOLD,
    bm25_thresh=BM25_SCORE_THRESHOLD,
    columns=_CWD_COLUMNS,
)

# Products (company_products → best product per company → RRF)
# Retrieval stays on company_products; company filter is applied only at the end.

PRODUCT_VECTOR_OVERSAMPLE = 2000

HYBRID_SEARCH_PRODUCTS = """\
WITH product_topk AS (
    SELECT cp.linkedin_slug,
           cp.embedding <=> $1::vector AS dist
    FROM company_products cp
    WHERE cp.embedding IS NOT NULL
      {{filters}}
    ORDER BY cp.embedding <=> $1::vector
    LIMIT {oversample}
),
vector_hits AS (
    SELECT linkedin_slug,
           ROW_NUMBER() OVER (ORDER BY min_dist) AS vrank
    FROM (
        SELECT linkedin_slug, MIN(dist) AS min_dist
        FROM product_topk
        GROUP BY linkedin_slug
        HAVING MIN(dist) < {vector_dist_thresh}
    ) deduped
    ORDER BY min_dist
    LIMIT $3
),
bm25_raw AS (
    SELECT cp.linkedin_slug,
           pdb.score(cp.id) AS rank
    FROM company_products cp
    WHERE cp.id @@@ pdb.parse($2, lenient => true)
      AND pdb.score(cp.id) > {bm25_thresh}
      {{filters}}
),
bm25_best AS (
    SELECT linkedin_slug,
           MAX(rank) AS rank
    FROM bm25_raw
    GROUP BY linkedin_slug
),
bm25_hits AS (
    SELECT linkedin_slug,
           ROW_NUMBER() OVER (ORDER BY rank DESC) AS brank
    FROM bm25_best
    ORDER BY rank DESC
    LIMIT $3
),
fused AS (
    SELECT COALESCE(v.linkedin_slug, b.linkedin_slug) AS linkedin_slug,
           COALESCE({vec_w}/({rrf_k}+v.vrank), 0) + COALESCE({bm25_w}/({rrf_k}+b.brank), 0) AS rrf_score,
           CASE WHEN v.linkedin_slug IS NOT NULL AND b.linkedin_slug IS NOT NULL THEN 2 ELSE 1 END AS overlap,
           LEAST(COALESCE(v.vrank, 999999), COALESCE(b.brank, 999999)) AS best_rank,
           v.vrank, b.brank
    FROM vector_hits v
    FULL OUTER JOIN bm25_hits b USING (linkedin_slug)
)
SELECT {columns},
       f.rrf_score, f.vrank, f.brank
FROM fused f
JOIN company_linkedin_data cwd USING (linkedin_slug)
LEFT JOIN ca_estimates_linkedin cel ON cel.linkedin_slug = cwd.linkedin_slug
WHERE (cwd.has_deal = TRUE OR cwd.is_pe_backed = TRUE)
  {{company_filters}}
ORDER BY f.rrf_score DESC, f.overlap DESC, f.best_rank ASC
LIMIT $3
""".format(
    columns=_CWD_COLUMNS,
    rrf_k=RRF_K,
    vec_w=VECTOR_WEIGHT,
    bm25_w=BM25_WEIGHT,
    vector_dist_thresh=VECTOR_DISTANCE_THRESHOLD,
    bm25_thresh=BM25_SCORE_THRESHOLD,
    oversample=PRODUCT_VECTOR_OVERSAMPLE,
)
