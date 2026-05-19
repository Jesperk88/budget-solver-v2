"""
Terminal diagnostics for explaining forecast misses and market shifts.

Phase 4 focuses on CPC and click-efficiency trends. Phase 6 adds impression
share diagnostics. The output is intentionally print-only so it can be used
while iterating without creating report files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from budget_solver.constants import DATA_PATH
from budget_solver.data import load_data

IMPRESSION_SHARE_COLUMNS = [
    "search_impression_share",
    "search_budget_lost_impression_share",
    "search_rank_lost_impression_share",
]


def _fmt_eur(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}€{value / 1_000_000:.2f}M"
    return f"{sign}€{value:,.0f}"


def _fmt_float(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: float | int | None, signed: bool = True) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    sign = "+" if signed else ""
    return f"{float(value) * 100:{sign}.1f}%"


def _fmt_pp(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:+.1f}pp"


def _print_table(title: str, rows: list[dict], columns: list[tuple[str, str, str]]) -> None:
    print()
    print(title)
    print("-" * len(title))
    if not rows:
        print("No rows.")
        return

    def format_value(value, fmt):
        if fmt == "eur":
            return _fmt_eur(value)
        if fmt == "pct":
            return _fmt_pct(value, signed=True)
        if fmt == "pct_abs":
            return _fmt_pct(value, signed=False)
        if fmt == "pp":
            return _fmt_pp(value)
        if fmt == "float":
            return _fmt_float(value)
        if fmt == "float3":
            return _fmt_float(value, 3)
        if fmt == "int":
            return "n/a" if value is None or pd.isna(value) else f"{int(value):,}"
        return "" if value is None else str(value)

    formatted = []
    widths = [len(label) for _, label, _ in columns]
    for row in rows:
        values = []
        for i, (key, _, fmt) in enumerate(columns):
            text = format_value(row.get(key), fmt)
            values.append(text)
            widths[i] = max(widths[i], len(text))
        formatted.append(values)

    print("  ".join(label.ljust(widths[i]) for i, (_, label, _) in enumerate(columns)))
    print("  ".join("-" * width for width in widths))
    for values in formatted:
        cells = []
        for i, text in enumerate(values):
            align_left = columns[i][2] == "str"
            cells.append(text.ljust(widths[i]) if align_left else text.rjust(widths[i]))
        print("  ".join(cells))


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def _prepare_data(path: Path, target: str) -> pd.DataFrame:
    df = load_data(path)
    date_col = next((c for c in ("date", "week_start", "week") if c in df.columns), None)
    if not date_col:
        raise ValueError("data must include a date, week_start, or week column")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["_date"] = df[date_col]
    df["_month"] = df["_date"].dt.to_period("M")

    if "clicks" not in df.columns:
        raise ValueError("CPC diagnostics require a clicks column")
    if target == "conversion_value":
        pass
    elif target == "conversions":
        if "conversions_adj" in df.columns:
            df["conversion_value"] = pd.to_numeric(df["conversions_adj"], errors="coerce").fillna(0)
            print("  Using lag-adjusted conversions (conversions_adj).")
        elif "conversions" in df.columns:
            df["conversion_value"] = pd.to_numeric(df["conversions"], errors="coerce").fillna(0)
            print("  Using raw conversions (conversions); conversions_adj not found.")
        else:
            raise ValueError("--target conversions requires conversions or conversions_adj")
    else:
        raise ValueError(f"unsupported target: {target}")

    for column in ("cost", "conversion_value", "clicks", "impressions"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    if "conversions" in df.columns:
        df["conversions"] = pd.to_numeric(df["conversions"], errors="coerce").fillna(0)
    if "conversions_adj" in df.columns:
        df["conversions_adj"] = pd.to_numeric(df["conversions_adj"], errors="coerce").fillna(0)
    for column in [*IMPRESSION_SHARE_COLUMNS, "eligible_search_impressions"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    agg_map = {
        "spend": ("cost", "sum"),
        "revenue": ("conversion_value", "sum"),
        "clicks": ("clicks", "sum"),
    }
    if "conversions_adj" in df.columns:
        agg_map["conversions"] = ("conversions_adj", "sum")
    elif "conversions" in df.columns:
        agg_map["conversions"] = ("conversions", "sum")

    monthly = (
        df.groupby(["account_name", "_month"])
        .agg(**agg_map)
        .reset_index()
    )
    monthly["month"] = monthly["_month"].astype(str)
    monthly["cpc"] = _safe_div(monthly["spend"], monthly["clicks"])
    monthly["clicks_per_eur"] = _safe_div(monthly["clicks"], monthly["spend"])
    monthly["roas"] = _safe_div(monthly["revenue"], monthly["spend"])
    monthly["revenue_per_click"] = _safe_div(monthly["revenue"], monthly["clicks"])
    if "conversions" in monthly.columns:
        monthly["cvr"] = _safe_div(monthly["conversions"], monthly["clicks"])
        monthly["value_per_conversion"] = _safe_div(monthly["revenue"], monthly["conversions"])
    else:
        monthly["cvr"] = np.nan
        monthly["value_per_conversion"] = np.nan
    return monthly


def weekly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_week"] = df["_date"].dt.to_period("W")
    df["_iso_week"] = df["_date"].dt.isocalendar().week.astype(int)
    df["_year"] = df["_date"].dt.isocalendar().year.astype(int)

    agg_map = {
        "spend": ("cost", "sum"),
        "revenue": ("conversion_value", "sum"),
        "clicks": ("clicks", "sum"),
    }
    if "conversions_adj" in df.columns:
        agg_map["conversions"] = ("conversions_adj", "sum")
    elif "conversions" in df.columns:
        agg_map["conversions"] = ("conversions", "sum")

    weekly = (
        df.groupby(["account_name", "_week", "_iso_week", "_year"])
        .agg(**agg_map)
        .reset_index()
    )
    weekly["roas"] = _safe_div(weekly["revenue"], weekly["spend"])
    weekly["cvr"] = _safe_div(weekly["conversions"], weekly["clicks"])
    weekly["revenue_per_click"] = _safe_div(weekly["revenue"], weekly["clicks"])
    weekly["clicks_per_eur"] = _safe_div(weekly["clicks"], weekly["spend"])
    weekly["cpc"] = _safe_div(weekly["spend"], weekly["clicks"])
    return weekly


def portfolio_monthly_metrics(monthly: pd.DataFrame) -> pd.DataFrame:
    portfolio = (
        monthly.groupby("_month")
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
        )
        .reset_index()
    )
    portfolio["account_name"] = "TOTAL"
    portfolio["month"] = portfolio["_month"].astype(str)
    portfolio["cpc"] = _safe_div(portfolio["spend"], portfolio["clicks"])
    portfolio["clicks_per_eur"] = _safe_div(portfolio["clicks"], portfolio["spend"])
    portfolio["roas"] = _safe_div(portfolio["revenue"], portfolio["spend"])
    portfolio["revenue_per_click"] = _safe_div(portfolio["revenue"], portfolio["clicks"])
    portfolio["cvr"] = _safe_div(portfolio["conversions"], portfolio["clicks"])
    portfolio["value_per_conversion"] = _safe_div(portfolio["revenue"], portfolio["conversions"])
    return portfolio


def demand_index_by_proxy(
    weekly: pd.DataFrame,
    proxy: str,
    smoothing_weeks: int = 1,
) -> pd.DataFrame:
    if proxy not in weekly.columns:
        raise ValueError(f"proxy not available: {proxy}")

    proxy_rows = weekly.replace([np.inf, -np.inf], np.nan).dropna(subset=[proxy])
    proxy_rows = proxy_rows[proxy_rows[proxy] > 0]
    if proxy_rows.empty:
        return pd.DataFrame()

    index = (
        proxy_rows.groupby("_iso_week")[proxy]
        .median()
        .reindex(range(1, 54))
    )
    index = index.fillna(index.median())

    if smoothing_weeks > 1:
        # Circular smoothing: ISO week 1 neighbors week 53.
        radius = smoothing_weeks // 2
        smoothed = []
        for week in range(1, 54):
            neighbors = []
            for offset in range(-radius, radius + 1):
                neighbor = ((week + offset - 1) % 53) + 1
                neighbors.append(index.loc[neighbor])
            smoothed.append(float(np.median(neighbors)))
        index = pd.Series(smoothed, index=range(1, 54))

    normalized = index / index.mean()
    result = normalized.reset_index()
    result.columns = ["iso_week", "index"]
    result["proxy"] = proxy
    result["smoothing_weeks"] = smoothing_weeks
    return result


def demand_proxy_stability(weekly: pd.DataFrame, proxy: str) -> pd.DataFrame:
    if proxy not in weekly.columns:
        return pd.DataFrame()

    rows = weekly.replace([np.inf, -np.inf], np.nan).dropna(subset=[proxy])
    rows = rows[rows[proxy] > 0]
    if rows.empty:
        return pd.DataFrame()

    by_year = (
        rows.groupby(["_year", "_iso_week"])[proxy]
        .median()
        .reset_index()
    )
    pivot = by_year.pivot(index="_iso_week", columns="_year", values=proxy)
    if pivot.shape[1] < 2:
        return pd.DataFrame()

    pivot = pivot.dropna(thresh=2)
    years = sorted(pivot.columns)
    records = []
    for i, year in enumerate(years):
        prior_years = years[:i]
        if not prior_years:
            continue
        prior = pivot[prior_years].median(axis=1)
        current = pivot[year]
        aligned = pd.DataFrame({"current": current, "prior": prior}).dropna()
        aligned = aligned[(aligned["current"] > 0) & (aligned["prior"] > 0)]
        if aligned.empty:
            continue
        ratio = aligned["current"] / aligned["prior"] - 1.0
        records.append(
            {
                "proxy": proxy,
                "year": int(year),
                "weeks": len(aligned),
                "median_abs_yoy_change": float(ratio.abs().median()),
                "mean_abs_yoy_change": float(ratio.abs().mean()),
                "bias_yoy_change": float(ratio.mean()),
            }
        )
    return pd.DataFrame(records)


def demand_week_extremes(index_df: pd.DataFrame, n: int = 5) -> tuple[list[dict], list[dict]]:
    if index_df.empty:
        return [], []
    low = index_df.nsmallest(n, "index").to_dict("records")
    high = index_df.nlargest(n, "index").to_dict("records")
    return low, high


def _period_summary(monthly: pd.DataFrame, months: list[pd.Period], label: str) -> pd.DataFrame:
    subset = monthly[monthly["_month"].isin(months)]
    summary = (
        subset.groupby("account_name")
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
        )
        .reset_index()
    )
    summary["period"] = label
    summary["cpc"] = _safe_div(summary["spend"], summary["clicks"])
    summary["clicks_per_eur"] = _safe_div(summary["clicks"], summary["spend"])
    summary["roas"] = _safe_div(summary["revenue"], summary["spend"])
    summary["revenue_per_click"] = _safe_div(summary["revenue"], summary["clicks"])
    summary["cvr"] = _safe_div(summary["conversions"], summary["clicks"])
    summary["value_per_conversion"] = _safe_div(summary["revenue"], summary["conversions"])
    return summary


def period_comparison(
    monthly: pd.DataFrame,
    months: int,
    compare: str,
) -> tuple[pd.DataFrame, str, str]:
    complete_months = sorted(monthly["_month"].dropna().unique())
    if not complete_months:
        return pd.DataFrame(), "", ""

    current_months = complete_months[-months:]
    current_label = f"{current_months[0]}..{current_months[-1]}"

    if compare == "previous":
        previous_months = complete_months[-(months * 2):-months]
        compare_label = f"{previous_months[0]}..{previous_months[-1]}" if previous_months else ""
    elif compare == "yoy":
        previous_months = [month - 12 for month in current_months]
        available = set(complete_months)
        previous_months = [month for month in previous_months if month in available]
        compare_label = f"{previous_months[0]}..{previous_months[-1]}" if previous_months else ""
    else:
        raise ValueError("--compare must be previous or yoy")

    if not previous_months:
        return pd.DataFrame(), current_label, compare_label

    current = _period_summary(monthly, current_months, "current")
    previous = _period_summary(monthly, previous_months, "comparison")

    merged = current.merge(previous, on="account_name", suffixes=("_current", "_prev"))
    for metric in [
        "spend",
        "revenue",
        "clicks",
        "conversions",
        "cpc",
        "clicks_per_eur",
        "roas",
        "revenue_per_click",
        "cvr",
        "value_per_conversion",
    ]:
        merged[f"{metric}_change"] = (
            merged[f"{metric}_current"] - merged[f"{metric}_prev"]
        ) / merged[f"{metric}_prev"].replace(0, np.nan)

    return merged, current_label, compare_label


def latest_month_table(monthly: pd.DataFrame) -> list[dict]:
    latest = monthly["_month"].max()
    rows = monthly[monthly["_month"] == latest].copy()
    return rows.sort_values("spend", ascending=False).to_dict("records")


def diagnostic_flags(comparison: pd.DataFrame) -> list[dict]:
    rows = []
    for row in comparison.to_dict("records"):
        flags = []
        if row.get("cpc_change", 0) > 0.15:
            flags.append("CPC up")
        if row.get("clicks_per_eur_change", 0) < -0.10:
            flags.append("fewer clicks/€")
        if row.get("cvr_change", 0) < -0.10:
            flags.append("CVR down")
        if row.get("revenue_per_click_change", 0) < -0.10:
            flags.append("rev/click down")
        if row.get("roas_change", 0) < -0.10:
            flags.append("ROAS down")

        rows.append(
            {
                "account_name": row["account_name"],
                "spend": row["spend_current"],
                "cpc_change": row["cpc_change"],
                "clicks_per_eur_change": row["clicks_per_eur_change"],
                "cvr_change": row["cvr_change"],
                "revenue_per_click_change": row["revenue_per_click_change"],
                "roas_change": row["roas_change"],
                "flags": ", ".join(flags) if flags else "ok",
            }
        )
    return sorted(rows, key=lambda r: r["spend"], reverse=True)


def _aggregate_impression_share(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = df.copy()
    if "impressions" not in rows.columns:
        raise ValueError("impression-share diagnostics require an impressions column")

    use_eligible = (
        "eligible_search_impressions" in rows.columns
        and rows["eligible_search_impressions"].notna().any()
    )

    if use_eligible:
        rows["_is_weight"] = rows["eligible_search_impressions"].where(
            rows["eligible_search_impressions"] > 0
        )
        rows["_search_is_weighted"] = rows["impressions"]
        rows["_budget_lost_weighted"] = (
            rows["search_budget_lost_impression_share"] * rows["_is_weight"]
        )
        rows["_rank_lost_weighted"] = (
            rows["search_rank_lost_impression_share"] * rows["_is_weight"]
        )
    else:
        rows["_is_weight"] = rows["impressions"].where(rows["impressions"] > 0)
        rows["_search_is_weighted"] = rows["search_impression_share"] * rows["_is_weight"]
        rows["_budget_lost_weighted"] = (
            rows["search_budget_lost_impression_share"] * rows["_is_weight"]
        )
        rows["_rank_lost_weighted"] = (
            rows["search_rank_lost_impression_share"] * rows["_is_weight"]
        )

    grouped = (
        rows.groupby(group_cols, dropna=False)
        .agg(
            spend=("cost", "sum"),
            revenue=("conversion_value", "sum"),
            impressions=("impressions", "sum"),
            eligible_search_impressions=("_is_weight", "sum"),
            search_is_weighted=("_search_is_weighted", "sum"),
            search_budget_lost_impressions=("_budget_lost_weighted", "sum"),
            search_rank_lost_impressions=("_rank_lost_weighted", "sum"),
        )
        .reset_index()
    )

    weight = grouped["eligible_search_impressions"].where(
        grouped["eligible_search_impressions"] > 0
    )
    grouped["search_impression_share"] = grouped["search_is_weighted"] / weight
    grouped["search_budget_lost_impression_share"] = (
        grouped["search_budget_lost_impressions"] / weight
    )
    grouped["search_rank_lost_impression_share"] = (
        grouped["search_rank_lost_impressions"] / weight
    )
    grouped["roas"] = _safe_div(grouped["revenue"], grouped["spend"])
    return grouped


def impression_share_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in IMPRESSION_SHARE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "impression-share diagnostics require columns: " + ", ".join(missing)
        )

    account_monthly = _aggregate_impression_share(df, ["account_name", "_month"])
    portfolio_monthly = _aggregate_impression_share(df, ["_month"])
    portfolio_monthly["account_name"] = "TOTAL"

    monthly = pd.concat([account_monthly, portfolio_monthly], ignore_index=True)
    monthly["month"] = monthly["_month"].astype(str)
    return monthly


def _impression_share_period_summary(
    monthly: pd.DataFrame,
    months: list[pd.Period],
    label: str,
) -> pd.DataFrame:
    subset = monthly[monthly["_month"].isin(months)]
    summary = (
        subset.groupby("account_name", dropna=False)
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            impressions=("impressions", "sum"),
            eligible_search_impressions=("eligible_search_impressions", "sum"),
            search_is_weighted=("search_is_weighted", "sum"),
            search_budget_lost_impressions=("search_budget_lost_impressions", "sum"),
            search_rank_lost_impressions=("search_rank_lost_impressions", "sum"),
        )
        .reset_index()
    )
    weight = summary["eligible_search_impressions"].where(
        summary["eligible_search_impressions"] > 0
    )
    summary["period"] = label
    summary["search_impression_share"] = summary["search_is_weighted"] / weight
    summary["search_budget_lost_impression_share"] = (
        summary["search_budget_lost_impressions"] / weight
    )
    summary["search_rank_lost_impression_share"] = (
        summary["search_rank_lost_impressions"] / weight
    )
    summary["roas"] = _safe_div(summary["revenue"], summary["spend"])
    return summary


def impression_share_period_comparison(
    monthly: pd.DataFrame,
    months: int,
    compare: str,
    available_months: list[pd.Period] | None = None,
) -> tuple[pd.DataFrame, str, str]:
    complete_months = (
        sorted(available_months)
        if available_months is not None
        else sorted(monthly["_month"].dropna().unique())
    )
    if not complete_months:
        return pd.DataFrame(), "", ""

    current_months = complete_months[-months:]
    current_label = f"{current_months[0]}..{current_months[-1]}"

    if compare == "previous":
        previous_months = complete_months[-(months * 2):-months]
        compare_label = f"{previous_months[0]}..{previous_months[-1]}" if previous_months else ""
    elif compare == "yoy":
        previous_months = [month - 12 for month in current_months]
        available = set(complete_months)
        previous_months = [month for month in previous_months if month in available]
        compare_label = f"{previous_months[0]}..{previous_months[-1]}" if previous_months else ""
    else:
        raise ValueError("--compare must be previous or yoy")

    if not previous_months:
        return pd.DataFrame(), current_label, compare_label

    current = _impression_share_period_summary(monthly, current_months, "current")
    previous = _impression_share_period_summary(monthly, previous_months, "comparison")
    merged = current.merge(previous, on="account_name", suffixes=("_current", "_prev"))

    for metric in [
        "search_impression_share",
        "search_budget_lost_impression_share",
        "search_rank_lost_impression_share",
    ]:
        merged[f"{metric}_delta"] = (
            merged[f"{metric}_current"] - merged[f"{metric}_prev"]
        )
    merged["spend_change"] = (
        merged["spend_current"] - merged["spend_prev"]
    ) / merged["spend_prev"].replace(0, np.nan)
    merged["roas_change"] = (
        merged["roas_current"] - merged["roas_prev"]
    ) / merged["roas_prev"].replace(0, np.nan)
    return merged, current_label, compare_label


def complete_months_from_daily(df: pd.DataFrame) -> list[pd.Period]:
    months = sorted(df["_month"].dropna().unique())
    if not months:
        return []

    latest_date = df["_date"].max()
    latest_month = latest_date.to_period("M")
    month_end = latest_month.to_timestamp(how="end").date()
    if latest_date.date() < month_end:
        months = [month for month in months if month != latest_month]
    return months


def impression_share_flags(comparison: pd.DataFrame) -> list[dict]:
    rows = []
    for row in comparison.to_dict("records"):
        budget_lost = row.get("search_budget_lost_impression_share_current")
        rank_lost = row.get("search_rank_lost_impression_share_current")
        search_is = row.get("search_impression_share_current")
        budget_delta = row.get("search_budget_lost_impression_share_delta")
        rank_delta = row.get("search_rank_lost_impression_share_delta")

        flags = []
        if pd.notna(budget_lost) and budget_lost >= 0.10:
            flags.append("budget constrained")
        if pd.notna(rank_lost) and rank_lost >= 0.20:
            flags.append("rank constrained")
        if pd.notna(search_is) and search_is < 0.50:
            flags.append("low search IS")
        if pd.notna(budget_delta) and budget_delta >= 0.05:
            flags.append("budget pressure up")
        if pd.notna(rank_delta) and rank_delta >= 0.05:
            flags.append("rank pressure up")
        if not flags:
            flags.append("ok")

        rows.append(
            {
                "account_name": row["account_name"],
                "spend": row["spend_current"],
                "search_impression_share": search_is,
                "search_budget_lost_impression_share": budget_lost,
                "search_rank_lost_impression_share": rank_lost,
                "flags": ", ".join(flags),
            }
        )
    return sorted(rows, key=lambda r: r["spend"], reverse=True)


def run_diagnostics(df: pd.DataFrame, months: int, compare: str) -> None:
    monthly = monthly_metrics(df)
    portfolio = portfolio_monthly_metrics(monthly)
    combined = pd.concat([monthly, portfolio], ignore_index=True)

    latest_rows = latest_month_table(combined)
    latest_label = latest_rows[0]["month"] if latest_rows else "n/a"
    _print_table(
        f"Latest Month CPC Diagnostics ({latest_label})",
        latest_rows,
        [
            ("account_name", "Account", "str"),
            ("spend", "Spend", "eur"),
            ("clicks", "Clicks", "int"),
            ("cpc", "CPC", "float"),
            ("clicks_per_eur", "Clicks/€", "float3"),
            ("cvr", "CVR", "pct_abs"),
            ("revenue_per_click", "Rev/Click", "float"),
            ("roas", "ROAS", "float"),
        ],
    )

    comparison, current_label, comparison_label = period_comparison(combined, months, compare)
    if comparison.empty:
        print()
        print("Not enough comparison history for the requested period.")
        return

    _print_table(
        f"Trend Comparison ({current_label} vs {comparison_label}, {compare})",
        comparison.sort_values("spend_current", ascending=False).to_dict("records"),
        [
            ("account_name", "Account", "str"),
            ("spend_current", "Spend", "eur"),
            ("cpc_change", "CPC Δ", "pct"),
            ("clicks_per_eur_change", "Clicks/€ Δ", "pct"),
            ("cvr_change", "CVR Δ", "pct"),
            ("revenue_per_click_change", "Rev/Click Δ", "pct"),
            ("roas_change", "ROAS Δ", "pct"),
        ],
    )

    _print_table(
        "Diagnostic Flags",
        diagnostic_flags(comparison),
        [
            ("account_name", "Account", "str"),
            ("spend", "Spend", "eur"),
            ("cpc_change", "CPC Δ", "pct"),
            ("clicks_per_eur_change", "Clicks/€ Δ", "pct"),
            ("cvr_change", "CVR Δ", "pct"),
            ("revenue_per_click_change", "Rev/Click Δ", "pct"),
            ("roas_change", "ROAS Δ", "pct"),
            ("flags", "Flags", "str"),
        ],
    )


def run_demand_diagnostics(df: pd.DataFrame, smoothing_weeks: int) -> None:
    weekly = weekly_metrics(df)
    proxies = ["roas", "cvr", "revenue_per_click", "clicks_per_eur"]

    stability_rows = []
    index_summaries = []
    for proxy in proxies:
        stability = demand_proxy_stability(weekly, proxy)
        if not stability.empty:
            stability_rows.extend(stability.to_dict("records"))

        raw_index = demand_index_by_proxy(weekly, proxy, smoothing_weeks=1)
        smooth_index = demand_index_by_proxy(weekly, proxy, smoothing_weeks=smoothing_weeks)
        if not raw_index.empty:
            index_summaries.append(
                {
                    "proxy": proxy,
                    "smoothing_weeks": 1,
                    "min_index": raw_index["index"].min(),
                    "max_index": raw_index["index"].max(),
                    "range": raw_index["index"].max() - raw_index["index"].min(),
                    "std": raw_index["index"].std(),
                }
            )
        if not smooth_index.empty and smoothing_weeks > 1:
            index_summaries.append(
                {
                    "proxy": proxy,
                    "smoothing_weeks": smoothing_weeks,
                    "min_index": smooth_index["index"].min(),
                    "max_index": smooth_index["index"].max(),
                    "range": smooth_index["index"].max() - smooth_index["index"].min(),
                    "std": smooth_index["index"].std(),
                }
            )

    _print_table(
        "Demand Proxy Year-to-Year Stability",
        sorted(stability_rows, key=lambda r: (r["proxy"], r["year"])),
        [
            ("proxy", "Proxy", "str"),
            ("year", "Year", "int"),
            ("weeks", "Weeks", "int"),
            ("median_abs_yoy_change", "Median |YoY|", "pct_abs"),
            ("mean_abs_yoy_change", "Mean |YoY|", "pct_abs"),
            ("bias_yoy_change", "Bias", "pct"),
        ],
    )

    _print_table(
        "Demand Index Volatility By Proxy",
        sorted(index_summaries, key=lambda r: (r["proxy"], r["smoothing_weeks"])),
        [
            ("proxy", "Proxy", "str"),
            ("smoothing_weeks", "Smooth", "int"),
            ("min_index", "Min", "float"),
            ("max_index", "Max", "float"),
            ("range", "Range", "float"),
            ("std", "Std", "float"),
        ],
    )

    for proxy in proxies:
        index_df = demand_index_by_proxy(weekly, proxy, smoothing_weeks=smoothing_weeks)
        low, high = demand_week_extremes(index_df)
        _print_table(
            f"Lowest Demand Weeks ({proxy}, smooth={smoothing_weeks})",
            low,
            [
                ("iso_week", "Week", "int"),
                ("index", "Index", "float"),
            ],
        )
        _print_table(
            f"Highest Demand Weeks ({proxy}, smooth={smoothing_weeks})",
            high,
            [
                ("iso_week", "Week", "int"),
                ("index", "Index", "float"),
            ],
        )


def run_impression_share_diagnostics(df: pd.DataFrame, months: int, compare: str) -> None:
    try:
        monthly = impression_share_monthly_metrics(df)
    except ValueError as exc:
        print()
        print(f"Impression share diagnostics unavailable: {exc}")
        print("Rerun the updated data pull so output/core_markets.csv includes Search IS fields.")
        return

    if monthly.empty:
        print()
        print("No impression-share rows available.")
        return

    latest = monthly["_month"].max()
    latest_rows = (
        monthly[monthly["_month"] == latest]
        .sort_values("spend", ascending=False)
        .to_dict("records")
    )
    _print_table(
        f"Latest Month Impression Share Diagnostics ({latest})",
        latest_rows,
        [
            ("account_name", "Account", "str"),
            ("spend", "Spend", "eur"),
            ("search_impression_share", "Search IS", "pct_abs"),
            ("search_budget_lost_impression_share", "Lost Budget IS", "pct_abs"),
            ("search_rank_lost_impression_share", "Lost Rank IS", "pct_abs"),
            ("roas", "ROAS", "float"),
        ],
    )

    complete_months = complete_months_from_daily(df)
    if latest not in complete_months:
        max_date = df["_date"].max().date()
        print()
        print(f"Trend comparisons exclude incomplete {latest} data through {max_date}.")

    comparison, current_label, comparison_label = impression_share_period_comparison(
        monthly, months, compare, available_months=complete_months
    )
    if comparison.empty:
        print()
        print("Not enough comparison history for the requested impression-share period.")
        return

    rows = comparison.sort_values("spend_current", ascending=False).to_dict("records")
    _print_table(
        f"Impression Share Trend ({current_label} vs {comparison_label}, {compare})",
        rows,
        [
            ("account_name", "Account", "str"),
            ("spend_current", "Spend", "eur"),
            ("search_impression_share_current", "Search IS", "pct_abs"),
            ("search_impression_share_delta", "Search IS d", "pp"),
            ("search_budget_lost_impression_share_current", "Lost Budget", "pct_abs"),
            ("search_budget_lost_impression_share_delta", "Budget d", "pp"),
            ("search_rank_lost_impression_share_current", "Lost Rank", "pct_abs"),
            ("search_rank_lost_impression_share_delta", "Rank d", "pp"),
            ("roas_change", "ROAS d", "pct"),
        ],
    )

    _print_table(
        "Impression Share Constraint Flags",
        impression_share_flags(comparison),
        [
            ("account_name", "Account", "str"),
            ("spend", "Spend", "eur"),
            ("search_impression_share", "Search IS", "pct_abs"),
            ("search_budget_lost_impression_share", "Lost Budget", "pct_abs"),
            ("search_rank_lost_impression_share", "Lost Rank", "pct_abs"),
            ("flags", "Flags", "str"),
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print model diagnostics from solver daily data."
    )
    parser.add_argument(
        "--data",
        default=str(DATA_PATH),
        help=f"Daily data CSV/XLSX path (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=3,
        help="Number of recent months to compare as a period (default: 3).",
    )
    parser.add_argument(
        "--compare",
        choices=["previous", "yoy"],
        default="yoy",
        help="Compare recent period to previous period or same months last year (default: yoy).",
    )
    parser.add_argument(
        "--target",
        choices=["conversion_value", "conversions"],
        default="conversion_value",
        help="Metric basis for value diagnostics (default: conversion_value).",
    )
    parser.add_argument(
        "--section",
        choices=["cpc", "demand", "impression-share", "all"],
        default="cpc",
        help="Which diagnostics to print (default: cpc).",
    )
    parser.add_argument(
        "--smoothing-weeks",
        type=int,
        default=5,
        help="Odd-numbered smoothing window for demand index diagnostics (default: 5).",
    )
    args = parser.parse_args(argv)

    if args.months <= 0:
        parser.error("--months must be greater than 0")
    if args.smoothing_weeks <= 0 or args.smoothing_weeks % 2 == 0:
        parser.error("--smoothing-weeks must be a positive odd number")

    path = Path(args.data)
    if not path.exists():
        print(f"ERROR: data file not found: {path}", file=sys.stderr)
        return 1

    print(f"Loading data from: {path}")
    try:
        df = _prepare_data(path, args.target)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print()
    print("Diagnostics configuration")
    print("-------------------------")
    print(f"Date range : {df['_date'].min().date()} to {df['_date'].max().date()}")
    print(f"Accounts   : {df['account_name'].nunique()}")
    print(f"Compare    : latest {args.months} month(s) vs {args.compare}")
    print(f"Target     : {args.target}")
    print(f"Section    : {args.section}")

    if args.section in {"cpc", "all"}:
        run_diagnostics(df, args.months, args.compare)
    if args.section in {"demand", "all"}:
        run_demand_diagnostics(df, args.smoothing_weeks)
    if args.section in {"impression-share", "all"}:
        run_impression_share_diagnostics(df, args.months, args.compare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
