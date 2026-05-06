from __future__ import annotations

import json
import re
from decimal import Decimal

_TANTIVY_SPECIAL = re.compile(r"""[+^`:{}"\[\]()~!\\*\-']""")


def normalize_bm25_keywords(keywords: list[str] | None, max_keywords: int = 8) -> list[str]:
    normalized_keywords: list[str] = []
    seen: set[str] = set()
    for keyword in keywords or []:
        if not isinstance(keyword, str):
            continue
        normalized = " ".join(keyword.strip().split())
        if not normalized:
            continue
        dedupe_key = normalized.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized_keywords.append(normalized)
        if len(normalized_keywords) >= max_keywords:
            break
    return normalized_keywords


def escape_tantivy_term(value: str) -> str:
    return _TANTIVY_SPECIAL.sub(r"\\\g<0>", value)


def build_bm25_query(
    terms: list[str],
    field_boosts: dict[str, float],
    fallback: str | None = None,
) -> str:
    effective_terms = terms or ([fallback] if fallback else [])
    clauses: list[str] = []
    for term in effective_terms:
        for field, boost in field_boosts.items():
            if boost != 1.0:
                clauses.append(f"{field}:{term}^{boost}")
            else:
                clauses.append(f"{field}:{term}")
    return " OR ".join(clauses)


def format_products_services(row: dict) -> str:
    value = row.get("products_services")
    if not value:
        return "n/a"

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("Product category") or item.get("Product name") or ""
                desc = item.get("Description") or ""
                if name:
                    parts.append(f"{name}: {desc}" if desc else name)
            else:
                parts.append(str(item))
        return "; ".join(parts) if parts else "n/a"

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def build_hybrid_search_summary(results: dict, top_buyers: int = 10) -> str:
    shortlist = results.get("shortlist") or []
    target_shortlist = results.get("target_shortlist") or shortlist
    deals_shortlist = results.get("deals_shortlist") or []
    acquirers_shortlist = results.get("acquirers_shortlist") or []

    lines = [
        "Hybrid search completed.",
        f"Targets kept: {len(target_shortlist)}",
        f"Final shortlist entities: {len(shortlist)}",
        f"Deals found: {len(deals_shortlist)}",
        f"Potential buyers found: {len(acquirers_shortlist)}",
        "",
        f"Top {min(top_buyers, len(acquirers_shortlist))} buyers:",
    ]

    if not acquirers_shortlist:
        lines.append("- None")
        return "\n".join(lines)

    for index, buyer in enumerate(acquirers_shortlist[:top_buyers], start=1):
        buyer_name = str(buyer.get("company_name") or buyer.get("acquirer_name") or "Unknown buyer")
        lines.append(f"{index}. {buyer_name}")

        if buyer.get("fund_id") is not None:
            fund_info: list[str] = []
            if aum := _format_aum(buyer):
                fund_info.append(aum)
            if ticket := _extract_strategy_value(buyer, "ticket"):
                fund_info.append(f"Ticket: {ticket}")
            if ev := _extract_strategy_value(buyer, "ev"):
                fund_info.append(f"Target EV: {ev}")
            if fund_info:
                lines.append(f"   Fund info: {' | '.join(fund_info)}")

        deals = buyer.get("deals") or []
        if not deals:
            lines.append("   Deals: none")
            continue

        lines.append("   Deals:")
        for deal in deals[:5]:
            company_name = str(deal.get("company_name") or "Unknown target")
            deal_type = str(deal.get("deal_type") or "n/a")
            deal_year = str(deal.get("deal_year") or "n/a")
            lines.append(f"   - {company_name} | {deal_type} | {deal_year}")
        if len(deals) > 5:
            lines.append(f"   - ... {len(deals) - 5} more")

    return "\n".join(lines)


def _parse_json_like(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _format_aum(row: dict) -> str | None:
    amount = row.get("aum_amount")
    if amount is None:
        return None
    if isinstance(amount, Decimal):
        amount = float(amount)
    currency = row.get("aum_currency") or ""
    year = row.get("aum_year")
    amount_str = f"{amount:,.0f}" if isinstance(amount, (int, float)) else str(amount)
    label = f"AUM: {amount_str}"
    if currency:
        label += f" {currency}"
    if year:
        label += f" ({year})"
    return label


def _extract_strategy_value(row: dict, key: str) -> str | None:
    strategies = _parse_json_like(row.get("strategies"))
    if not isinstance(strategies, list) or not strategies:
        return None
    first = strategies[0]
    if not isinstance(first, dict):
        return None
    value = first.get(key)
    if value in (None, "", "n.a.", "n.d."):
        return None
    return str(value)
