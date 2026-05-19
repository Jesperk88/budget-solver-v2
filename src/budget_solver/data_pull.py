#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
Budget Solver — Google Ads API Data Pull (v25)
───────────────────────────────────────────────
Pulls 24 months of daily account-level performance data from Google Ads API
for the 6 core markets. Queries at campaign level so filters can be applied,
then aggregates to account + date before writing to CSV.

FILTERS:
  - Campaign type = SEARCH only
  - Exclude campaigns containing "| BR" or "| PK" in the name
  - Core markets only: Landal BE/NL/DE, Roompot BE/NL/DE (see CORE_ACCOUNTS)

CONVERSION LAG CORRECTION:
  Conversions are attributed back to the click date, so recent days have
  incomplete data — not all conversions have arrived yet. Rather than
  excluding the last 30 days, we apply a daily multiplier derived from
  the observed conversion lag profile. Days older than 30 days are fully
  settled (multiplier = 1.0). The adjusted columns are:
    conversion_value_adj  — lag-corrected revenue (used by optimizer)
    conversions_adj       — lag-corrected conversions
    lag_factor            — the multiplier applied (1/cumulative_pct)

OUTPUT:
  output/core_markets.csv  — 6 core markets, one row per account per day (feeds into optimizer)

SETUP:
  1. pip install -e .
  2. Create google-ads.yaml (see README or bottom of this file)
  3. Run: budget-solver-pull
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

# Micros conversion constant
MICROS_PER_UNIT = 1_000_000

# ─────────────────────────────────────────────────────────────
# CONVERSION LAG PROFILE
# ─────────────────────────────────────────────────────────────
# Daily incremental % of total conversions that arrive on each day
# after the click. Derived from conversion lag deep-dive report.
# Buckets 14-21 and 21-30 are distributed evenly within the bucket.
#
#   days_elapsed 0  = same-day conversions  (38.31%)
#   days_elapsed 1  = conversions 1 day later (7.81%)
#   ...
#   days_elapsed 29 = last day of 30-day window
#
_DAILY_INCREMENTS_PCT = [
    38.31,          # day 0  (<1 day)
     7.81,          # day 1  (1-2 days)
     4.70,          # day 2  (2-3 days)
     3.82,          # day 3  (3-4 days)
     3.05,          # day 4  (4-5 days)
     2.63,          # day 5  (5-6 days)
     2.71,          # day 6  (6-7 days)
     2.58,          # day 7  (7-8 days)
     2.28,          # day 8  (8-9 days)
     2.22,          # day 9  (9-10 days)
     1.93,          # day 10 (10-11 days)
     1.81,          # day 11 (11-12 days)
     1.66,          # day 12 (12-13 days)
     1.74,          # day 13 (13-14 days)
    *([10.61 / 7] * 7),   # days 14-20  (14-21 day bucket, spread evenly)
    *([12.14 / 9] * 9),   # days 21-29  (21-30 day bucket, spread evenly)
]

# Build cumulative table: LAG_CUMULATIVE[d] = fraction of conversions
# expected to have arrived after d days have elapsed (0.0 – 1.0).
# Days >= 30 are fully settled → factor 1.0.
_cumulative = np.cumsum(_DAILY_INCREMENTS_PCT) / 100.0
LAG_CUMULATIVE = {d: min(float(_cumulative[d]), 1.0) for d in range(30)}


def lag_factor(days_elapsed: int) -> float:
    """
    Return the multiplier to apply to observed conversion_value so it
    represents the estimated fully-settled total.

    days_elapsed = (pull_date - click_date).days

    Examples:
      0 days → only 38.3% arrived → multiply by 2.61
      7 days → 65.6% arrived      → multiply by 1.52
     14 days → 78.8% arrived      → multiply by 1.27
     30+ days → 100% arrived      → multiply by 1.00
    """
    if days_elapsed >= 30:
        return 1.0
    cum = LAG_CUMULATIVE.get(days_elapsed, 1.0)
    return 1.0 / cum if cum > 0 else 1.0


def apply_lag_correction(df: pd.DataFrame, pull_date: datetime) -> pd.DataFrame:
    """
    Add lag_factor, conversion_value_adj, and conversions_adj columns.
    Spend and clicks are NOT adjusted (they are fully attributed same-day).
    """
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    pull_day = pd.Timestamp(pull_date.date())
    df["days_elapsed"] = (pull_day - df["date_dt"]).dt.days
    df["lag_factor"] = df["days_elapsed"].apply(lag_factor)
    df["conversion_value_adj"] = df["conversion_value"] * df["lag_factor"]
    df["conversions_adj"]      = df["conversions"]      * df["lag_factor"]
    df.drop(columns=["date_dt"], inplace=True)
    return df

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Query child accounts from each of these MCCs
CHILD_MCCS = {
    "8265762094": "Landal MCC",
    "6917028372": "Roompot MCC",
}

# Core markets to include in output (all others filtered out)
# These are the 6 accounts used for budget optimization
CORE_ACCOUNTS = [
    "Landal BE",
    "Landal NL",
    "Landal DE",
    "Roompot BE",
    "Roompot NL",
    "Roompot DE",
]

OUTPUT_CSV      = "output/core_markets.csv"
YAML_PATH       = Path(os.getenv(
    "GOOGLE_ADS_YAML_PATH",
    Path.home() / ".config" / "landal" / "google-ads.yaml"
))

# ─────────────────────────────────────────────────────────────
# GAQL — campaign-level daily data (1 year)
# Filters:
#   • SEARCH campaigns only
#   • Exclude "| BR" and "| PK" in campaign name
#   • Only days with spend > 0
# ─────────────────────────────────────────────────────────────

def build_query(start_date: str, end_date: str) -> str:
    return f"""
        SELECT
            customer.descriptive_name,
            customer.id,
            customer.currency_code,
            segments.date,
            metrics.cost_micros,
            metrics.conversions,
            metrics.conversions_value,
            metrics.clicks,
            metrics.impressions,
            metrics.search_impression_share,
            metrics.search_budget_lost_impression_share,
            metrics.search_rank_lost_impression_share
        FROM campaign
        WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
          AND campaign.advertising_channel_type = 'SEARCH'
          AND campaign.name NOT LIKE '%| BR%'
          AND campaign.name NOT LIKE '%| PK%'
          AND metrics.cost_micros > 0
        ORDER BY segments.date DESC
    """


# ─────────────────────────────────────────────────────────────
# LIST CHILD ACCOUNTS UNDER MCC
# ─────────────────────────────────────────────────────────────

def get_child_accounts(client: GoogleAdsClient, mcc_id: str) -> list[dict]:
    """Return list of {id, name} for all non-manager accounts under the MCC."""
    ga_service = client.get_service("GoogleAdsService")

    query = """
        SELECT
            customer_client.id,
            customer_client.descriptive_name,
            customer_client.manager,
            customer_client.status
        FROM customer_client
        WHERE customer_client.manager = FALSE
          AND customer_client.status = 'ENABLED'
    """

    accounts = []
    try:
        response = ga_service.search(customer_id=mcc_id, query=query)
        for row in response:
            accounts.append({
                "id":   str(row.customer_client.id),
                "name": row.customer_client.descriptive_name
            })
    except GoogleAdsException as ex:
        print(f"ERROR listing child accounts: {ex.error.code().name}")
        for error in ex.failure.errors:
            print(f"  {error.message}")
        sys.exit(1)

    return accounts


# ─────────────────────────────────────────────────────────────
# PULL DATA FOR ONE ACCOUNT
# ─────────────────────────────────────────────────────────────

def pull_account_data(
    client: GoogleAdsClient,
    account_id: str,
    account_name: str,
    query: str
) -> list[dict]:
    """Run GAQL query against one account and return rows as dicts."""
    ga_service = client.get_service("GoogleAdsService")
    rows = []

    try:
        response = ga_service.search(customer_id=account_id, query=query)
        for row in response:
            rows.append({
                "account_id":        account_id,
                "account_name":      account_name,
                "date":              row.segments.date,
                "cost":              row.metrics.cost_micros / MICROS_PER_UNIT,
                "conversions":       row.metrics.conversions,
                "conversion_value":  row.metrics.conversions_value,
                "clicks":            row.metrics.clicks,
                "impressions":       row.metrics.impressions,
                "search_impression_share": row.metrics.search_impression_share,
                "search_budget_lost_impression_share": row.metrics.search_budget_lost_impression_share,
                "search_rank_lost_impression_share": row.metrics.search_rank_lost_impression_share,
                "currency":          row.customer.currency_code,
            })
    except GoogleAdsException as ex:
        code = ex.error.code().name
        print(f"  WARNING: skipping {account_name} ({account_id}) — {code}")
        for error in ex.failure.errors:
            print(f"    {error.message}")

    return rows


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)

    # Date range: 24 months lookback ending today.
    # Conversion lag is handled by per-day multipliers (lag_factor) rather
    # than a hard cutoff, so we pull through to today and correct each day.
    pull_date  = datetime.today()
    end_date   = pull_date
    start_date = end_date - relativedelta(months=24)
    start_str  = start_date.strftime("%Y-%m-%d")
    end_str    = end_date.strftime("%Y-%m-%d")

    print(f"Budget Solver — Data Pull")
    print(f"Date range : {start_str} to {end_str}  (24-month lookback, lag-corrected)")
    print(f"Filters    : SEARCH campaigns only | excluding '| BR' and '| PK'")
    print()

    # Authenticate
    if not Path(YAML_PATH).exists():
        print(f"ERROR: credentials file not found at '{YAML_PATH}'")
        print("See the CREDENTIALS SETUP section at the bottom of this file.")
        sys.exit(1)

    client = GoogleAdsClient.load_from_storage(YAML_PATH, version="v23")

    # List child accounts across all MCCs
    accounts = []
    for mcc_id, mcc_name in CHILD_MCCS.items():
        print(f"Fetching child accounts under {mcc_name} ({mcc_id})...")
        found = get_child_accounts(client, mcc_id)
        print(f"  Found {len(found)} account(s)")
        accounts.extend(found)
    print(f"\nTotal accounts: {len(accounts)}\n")

    query = build_query(start_str, end_str)

    # Pull data for each account
    all_rows = []
    for acc in accounts:
        print(f"  Pulling: {acc['name']} ({acc['id']})", end="", flush=True)
        rows = pull_account_data(client, acc["id"], acc["name"], query)
        all_rows.extend(rows)
        print(f" → {len(rows):,} campaign-day rows")

    if not all_rows:
        print("\nNo data returned. Check your filters and date range.")
        sys.exit(1)

    # Build DataFrame
    df = pd.DataFrame(all_rows)
    for col in [
        "search_impression_share",
        "search_budget_lost_impression_share",
        "search_rank_lost_impression_share",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_search_is = (
        df["search_impression_share"].notna()
        & (df["search_impression_share"] > 0)
        & (df["impressions"] > 0)
    )
    df["eligible_search_impressions"] = np.where(
        valid_search_is,
        df["impressions"] / df["search_impression_share"],
        np.nan,
    )
    df["search_budget_lost_impressions"] = (
        df["search_budget_lost_impression_share"] * df["eligible_search_impressions"]
    )
    df["search_rank_lost_impressions"] = (
        df["search_rank_lost_impression_share"] * df["eligible_search_impressions"]
    )

    # Aggregate campaign-level rows → account + date level
    # (multiple campaigns per account per day collapse into one row)
    df_daily = (
        df.groupby(["account_id", "account_name", "date", "currency"])
        .agg(
            cost=("cost", "sum"),
            conversions=("conversions", "sum"),
            conversion_value=("conversion_value", "sum"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            eligible_search_impressions=("eligible_search_impressions", "sum"),
            search_budget_lost_impressions=("search_budget_lost_impressions", "sum"),
            search_rank_lost_impressions=("search_rank_lost_impressions", "sum"),
        )
        .reset_index()
        .sort_values(["account_name", "date"])
    )

    eligible = df_daily["eligible_search_impressions"].where(
        df_daily["eligible_search_impressions"] > 0
    )
    df_daily["search_impression_share"] = df_daily["impressions"] / eligible
    df_daily["search_budget_lost_impression_share"] = (
        df_daily["search_budget_lost_impressions"] / eligible
    )
    df_daily["search_rank_lost_impression_share"] = (
        df_daily["search_rank_lost_impressions"] / eligible
    )
    df_daily.drop(
        columns=["search_budget_lost_impressions", "search_rank_lost_impressions"],
        inplace=True,
    )

    # Remove any residual zero-spend rows post-aggregation
    df_daily = df_daily[df_daily["cost"] > 0]

    # Apply conversion lag correction
    df_daily = apply_lag_correction(df_daily, pull_date)

    # Filter to core markets only
    df_daily = df_daily[df_daily["account_name"].isin(CORE_ACCOUNTS)]
    if len(df_daily) == 0:
        print("\nERROR: No data found for core markets after filtering.")
        print(f"Core markets configured: {CORE_ACCOUNTS}")
        print("\nCheck account names in the output above and update CORE_ACCOUNTS list if needed.")
        sys.exit(1)

    # Write CSV (already filtered to core markets)
    df_daily.to_csv(OUTPUT_CSV, index=False)

    # Summary — show both raw and lag-adjusted ROAS for the last 30 days
    print()
    print("── FULL PERIOD SUMMARY ──────────────────────────────────────────")
    summary = (
        df_daily.groupby("account_name")
        .agg(
            days=("date", "nunique"),
            total_spend=("cost", "sum"),
            total_revenue_adj=("conversion_value_adj", "sum"),
        )
        .reset_index()
    )
    for _, r in summary.iterrows():
        roas = r.total_revenue_adj / r.total_spend if r.total_spend > 0 else 0
        print(f"  {r.account_name:<35} {r.days:>3} days  "
              f"spend=€{r.total_spend:>10,.0f}  "
              f"rev_adj=€{r.total_revenue_adj:>10,.0f}  "
              f"ROAS={roas:.2f}")

    # Trailing 30-day summary: lag-adjusted vs raw (shows lag impact)
    recent_start = (pull_date - timedelta(days=29)).strftime("%Y-%m-%d")
    recent_end = pull_date.strftime("%Y-%m-%d")
    recent = df_daily[(df_daily["date"] >= recent_start) & (df_daily["date"] <= recent_end)]
    if len(recent):
        print()
        print(f"── TRAILING 30 DAYS ({recent_start} to {recent_end}, lag-adjusted vs raw) ──")
        rec_sum = (
            recent.groupby("account_name")
            .agg(
                spend=("cost", "sum"),
                rev_raw=("conversion_value", "sum"),
                rev_adj=("conversion_value_adj", "sum"),
            )
            .reset_index()
        )
        for _, r in rec_sum.iterrows():
            roas_raw = r.rev_raw / r.spend if r.spend > 0 else 0
            roas_adj = r.rev_adj / r.spend if r.spend > 0 else 0
            print(f"  {r.account_name:<35}  "
                  f"ROAS raw={roas_raw:.2f}x  adj={roas_adj:.2f}x  "
                  f"(lag uplift: +{roas_adj - roas_raw:.2f}x)")

    print()
    print(f"Total rows : {len(df_daily):,} (filtered to {len(CORE_ACCOUNTS)} core markets)")
    print(f"Saved to   : {Path(OUTPUT_CSV).resolve()}")
    print()
    print("Lag correction applied. 'conversion_value_adj' is the column used by optimizer")
    print("Impression share diagnostics fields included for Search budget/rank constraints")
    print("Next step  : budget-solver --budget <amount> --data output/core_markets.csv --scenarios")


if __name__ == "__main__":
    main()


# ═════════════════════════════════════════════════════════════
# CREDENTIALS SETUP
# ═════════════════════════════════════════════════════════════
#
# Create a file called google-ads.yaml in the same folder as this script:
#
#   developer_token: YOUR_DEVELOPER_TOKEN
#   client_id:       YOUR_OAUTH_CLIENT_ID
#   client_secret:   YOUR_OAUTH_CLIENT_SECRET
#   refresh_token:   YOUR_REFRESH_TOKEN
#   login_customer_id: YOUR_MCC_CUSTOMER_ID   # digits only, no dashes
#   use_proto_plus:  True
#
# To get these credentials:
#   1. Developer token — Google Ads UI → Admin → API Center
#   2. OAuth credentials — Google Cloud Console → APIs & Services → Credentials
#      Create an OAuth 2.0 Client ID (Desktop app)
#   3. Refresh token — run the OAuth flow once using google-auth-oauthlib:
#      https://developers.google.com/google-ads/api/docs/oauth/overview
#      Or use the generate_user_credentials.py helper in the google-ads-python examples repo
#
# IMPORTANT: add google-ads.yaml to .gitignore — never commit credentials
