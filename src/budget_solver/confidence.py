"""
Bootstrap confidence intervals for budget-solver scenarios.

This is intentionally print-only while Phase 7 is being validated. It resamples
weekly training observations, refits response curves, recalibrates them, reruns
the scenario framework, and reports risk ranges for revenue, ROAS, spend, and
instantaneous mROAS.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from budget_solver.constants import DATA_PATH, TRAILING_WINDOW_DAYS, WEEKS_PER_MONTH
from budget_solver.curves import fit_portfolio_curves
from budget_solver.data import aggregate_weekly, load_data, remove_outliers
from budget_solver.scenarios import ScenarioSet, build_scenarios


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _calibration_blend(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("calibration blend must be between 0 and 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _fmt_eur(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value >= 1_000_000:
        return f"{sign}EUR {value / 1_000_000:.2f}M"
    return f"{sign}EUR {value:,.0f}"


def _fmt_roas(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}x"


def _fmt_pct(value: float | int | None, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    sign = "+" if signed else ""
    return f"{float(value) * 100:{sign}.1f}%"


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
        if fmt == "roas":
            return _fmt_roas(value)
        if fmt == "pct":
            return _fmt_pct(value, signed=False)
        if fmt == "pct_signed":
            return _fmt_pct(value, signed=True)
        if fmt == "int":
            return "n/a" if value is None or pd.isna(value) else f"{int(value):,}"
        if fmt == "score":
            return "n/a" if value is None or pd.isna(value) else f"{float(value):.0f}"
        if fmt == "float":
            return "n/a" if value is None or pd.isna(value) else f"{float(value):.3f}"
        return "" if value is None else str(value)

    widths = [len(label) for _, label, _ in columns]
    formatted = []
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


def _prepare_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()
    if target == "conversion_value":
        return df
    if target != "conversions":
        raise ValueError(f"unsupported target: {target}")

    if "conversions_adj" in df.columns:
        df["conversion_value"] = pd.to_numeric(df["conversions_adj"], errors="coerce").fillna(0)
        print("  Using lag-adjusted conversions (conversions_adj).")
    elif "conversions" in df.columns:
        df["conversion_value"] = pd.to_numeric(df["conversions"], errors="coerce").fillna(0)
        print("  Using raw conversions (conversions); conversions_adj not found.")
    else:
        raise ValueError("--target conversions requires conversions or conversions_adj")
    return df


def _date_col(df: pd.DataFrame) -> str:
    date_col = next((c for c in ("date", "week_start", "week") if c in df.columns), None)
    if not date_col:
        raise ValueError("data must include a date, week_start, or week column")
    return date_col


def _latest_complete_month_budget(df: pd.DataFrame, date_col: str) -> tuple[float, str]:
    dated = df.copy()
    dated["_month"] = dated[date_col].dt.to_period("M")
    months = sorted(dated["_month"].dropna().unique())
    if not months:
        raise ValueError("cannot infer budget because no valid dated rows were found")

    latest_date = dated[date_col].max().normalize()
    latest_month = latest_date.to_period("M")
    if latest_date.date() < latest_month.to_timestamp(how="end").date():
        months = [month for month in months if month != latest_month]
    if not months:
        raise ValueError("cannot infer budget because there is no complete month in the data")

    month = months[-1]
    budget = float(dated.loc[dated["_month"] == month, "cost"].sum())
    return budget, f"latest complete month ({month})"


def _training_slice(
    df: pd.DataFrame,
    date_col: str,
    training_months: int,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    latest = df[date_col].max().normalize()
    if training_months == 0:
        start = df[date_col].min().normalize()
    else:
        start = latest - pd.DateOffset(months=training_months)
    return df[df[date_col] >= start].copy(), start, latest


def _calibration_slice(
    df: pd.DataFrame,
    date_col: str,
    calibration_days: int,
) -> pd.DataFrame:
    latest = df[date_col].max().normalize()
    start = latest - pd.Timedelta(days=calibration_days - 1)
    return df[(df[date_col] >= start) & (df[date_col] <= latest)].copy()


def _scaled_params(params, model_name: str, scale: float) -> list[float]:
    base = model_name.replace("+cal", "")
    params = list(params)
    if not params:
        return params
    if base in {"log", "linear_fallback"}:
        return [float(p) * scale for p in params]
    if base == "power":
        scaled = params.copy()
        scaled[0] = float(scaled[0]) * scale
        return scaled
    return params


def _fit_monthly_predictors(
    account_data: dict,
    calibration_df: pd.DataFrame,
    calibrate: bool,
    calibration_blend: float,
    calibration_min: float | None,
    calibration_max: float | None,
) -> tuple[dict, dict, dict]:
    with contextlib.redirect_stdout(io.StringIO()):
        portfolio_results = fit_portfolio_curves(account_data, preferred_model="log")

    predict_fns = {}
    model_info = {}
    calibration_factors = {}

    for account, (fn, params, r2, model_name) in portfolio_results.items():
        predict_fns[account] = (
            lambda x, fn=fn, wpm=WEEKS_PER_MONTH: wpm * fn(x / wpm)
        )
        model_info[account] = (fn, params, r2, model_name)

    if calibrate and len(calibration_df):
        actual_spend = calibration_df.groupby("account_name")["cost"].sum().to_dict()
        actual_revenue = calibration_df.groupby("account_name")["conversion_value"].sum().to_dict()

        for account in list(predict_fns):
            spend = float(actual_spend.get(account, 0.0))
            revenue = float(actual_revenue.get(account, 0.0))
            if spend <= 0 or revenue <= 0:
                continue

            predicted = float(predict_fns[account](spend))
            if predicted <= 0:
                continue

            raw_scale = (revenue / spend) / (predicted / spend)
            capped_scale = raw_scale
            if calibration_min is not None:
                capped_scale = max(capped_scale, calibration_min)
            if calibration_max is not None:
                capped_scale = min(capped_scale, calibration_max)
            scale = 1.0 + calibration_blend * (capped_scale - 1.0)

            original_fn = predict_fns[account]
            predict_fns[account] = lambda x, fn=original_fn, s=scale: fn(x) * s

            fn_i, params_i, r2_i, model_name_i = model_info[account]
            model_info[account] = (
                fn_i,
                _scaled_params(params_i, model_name_i, scale),
                r2_i,
                f"{model_name_i}+cal",
            )
            calibration_factors[account] = scale

    return predict_fns, model_info, calibration_factors


def _bootstrap_account_data(account_data: dict, rng: np.random.Generator) -> dict:
    sampled = {}
    for account, data in account_data.items():
        spend = np.asarray(data["spend"], dtype=float)
        revenue = np.asarray(data["revenue"], dtype=float)
        weeks = np.asarray(data["_week"])
        if len(spend) == 0:
            continue
        sample_idx = rng.integers(0, len(spend), size=len(spend))
        sampled[account] = {
            "spend": spend[sample_idx],
            "revenue": revenue[sample_idx],
            "_week": weeks[sample_idx],
        }
    return sampled


def _scenario_rows(
    scenario_set: ScenarioSet,
    iteration: int,
    sample_type: str,
) -> tuple[list[dict], list[dict]]:
    portfolio_rows = []
    account_rows = []
    for scenario in scenario_set.scenarios:
        portfolio_rows.append(
            {
                "iteration": iteration,
                "sample_type": sample_type,
                "scenario_id": scenario.id,
                "scenario_name": scenario.name,
                "budget": scenario.budget_monthly,
                "revenue": scenario.revenue_monthly,
                "roas": scenario.blended_roas,
                "portfolio_discrete_mroas": scenario.portfolio_discrete_mroas,
                "recommended": scenario.recommended,
            }
        )
        for allocation in scenario.allocations:
            account_rows.append(
                {
                    "iteration": iteration,
                    "sample_type": sample_type,
                    "scenario_id": scenario.id,
                    "scenario_name": scenario.name,
                    "recommended": scenario.recommended,
                    "account": allocation.account,
                    "spend": allocation.monthly_spend,
                    "revenue": allocation.monthly_revenue,
                    "roas": allocation.roas,
                    "inst_mroas": allocation.inst_mroas,
                    "discrete_mroas": allocation.discrete_mroas_vs_prev,
                    "change_vs_prev": allocation.change_vs_prev,
                }
            )
    return portfolio_rows, account_rows


def _run_scenario_sample(
    df: pd.DataFrame,
    account_data: dict,
    calibration_df: pd.DataFrame,
    budget: float,
    min_mroas: float,
    baseline_window: int,
    max_account_changes: int,
    wow_cap: float,
    apply_stability: bool,
    calibrate: bool,
    calibration_blend: float,
    calibration_min: float | None,
    calibration_max: float | None,
    iteration: int,
    sample_type: str,
) -> tuple[list[dict], list[dict]]:
    predict_fns, model_info, _ = _fit_monthly_predictors(
        account_data=account_data,
        calibration_df=calibration_df,
        calibrate=calibrate,
        calibration_blend=calibration_blend,
        calibration_min=calibration_min,
        calibration_max=calibration_max,
    )
    scenario_set = build_scenarios(
        df=df,
        predict_fns=predict_fns,
        model_info=model_info,
        target_budget=budget,
        min_mroas=min_mroas,
        baseline_window_days=baseline_window,
        max_account_changes=max_account_changes,
        wow_cap=wow_cap,
        apply_stability=apply_stability,
    )
    return _scenario_rows(scenario_set, iteration=iteration, sample_type=sample_type)


def _account_quality_metadata(
    account_data: dict,
    model_info: dict,
    calibration_factors: dict,
) -> dict:
    quality = {}
    for account, data in account_data.items():
        _, _, r2, model_name = model_info.get(account, (None, None, np.nan, "unknown"))
        quality[account] = {
            "clean_training_weeks": len(data.get("spend", [])),
            "r2": float(r2) if pd.notna(r2) else np.nan,
            "calibration_factor": float(calibration_factors.get(account, 1.0)),
            "model": model_name,
        }
    return quality


def run_bootstrap(
    df: pd.DataFrame,
    date_col: str,
    budget: float,
    iterations: int,
    seed: int,
    training_months: int,
    outlier_removal: bool,
    calibrate: bool,
    calibration_days: int,
    calibration_blend: float,
    calibration_min: float | None,
    calibration_max: float | None,
    min_mroas: float,
    baseline_window: int,
    max_account_changes: int,
    wow_cap: float,
    apply_stability: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    training_df, train_start, train_end = _training_slice(df, date_col, training_months)
    calibration_df = _calibration_slice(df, date_col, calibration_days)

    account_data = aggregate_weekly(training_df)
    removal_log = []
    if outlier_removal:
        account_data, removal_log = remove_outliers(account_data)

    _, point_model_info, point_calibration_factors = _fit_monthly_predictors(
        account_data=account_data,
        calibration_df=calibration_df,
        calibrate=calibrate,
        calibration_blend=calibration_blend,
        calibration_min=calibration_min,
        calibration_max=calibration_max,
    )
    account_quality = _account_quality_metadata(
        account_data,
        point_model_info,
        point_calibration_factors,
    )

    portfolio_rows = []
    account_rows = []
    point_portfolio, point_accounts = _run_scenario_sample(
        df=df,
        account_data=account_data,
        calibration_df=calibration_df,
        budget=budget,
        min_mroas=min_mroas,
        baseline_window=baseline_window,
        max_account_changes=max_account_changes,
        wow_cap=wow_cap,
        apply_stability=apply_stability,
        calibrate=calibrate,
        calibration_blend=calibration_blend,
        calibration_min=calibration_min,
        calibration_max=calibration_max,
        iteration=0,
        sample_type="point",
    )
    portfolio_rows.extend(point_portfolio)
    account_rows.extend(point_accounts)

    rng = np.random.default_rng(seed)
    failures = []
    for iteration in range(1, iterations + 1):
        sampled_data = _bootstrap_account_data(account_data, rng)
        try:
            boot_portfolio, boot_accounts = _run_scenario_sample(
                df=df,
                account_data=sampled_data,
                calibration_df=calibration_df,
                budget=budget,
                min_mroas=min_mroas,
                baseline_window=baseline_window,
                max_account_changes=max_account_changes,
                wow_cap=wow_cap,
                apply_stability=apply_stability,
                calibrate=calibrate,
                calibration_blend=calibration_blend,
                calibration_min=calibration_min,
                calibration_max=calibration_max,
                iteration=iteration,
                sample_type="bootstrap",
            )
        except Exception as exc:
            failures.append({"iteration": iteration, "error": str(exc)})
            continue
        portfolio_rows.extend(boot_portfolio)
        account_rows.extend(boot_accounts)

    metadata = {
        "train_start": train_start,
        "train_end": train_end,
        "training_rows": len(training_df),
        "training_weeks": training_df[date_col].dt.to_period("W").nunique(),
        "calibration_start": calibration_df[date_col].min() if len(calibration_df) else None,
        "calibration_end": calibration_df[date_col].max() if len(calibration_df) else None,
        "outliers_removed": len(removal_log),
        "account_quality": account_quality,
        "failures": failures,
        "successful_iterations": len(
            {row["iteration"] for row in portfolio_rows if row["sample_type"] == "bootstrap"}
        ),
    }
    return pd.DataFrame(portfolio_rows), pd.DataFrame(account_rows), metadata


def _interval(series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"p10": np.nan, "p50": np.nan, "p90": np.nan}
    q10, q50, q90 = np.percentile(clean, [10, 50, 90])
    return {"p10": float(q10), "p50": float(q50), "p90": float(q90)}


def _point_lookup(df: pd.DataFrame, keys: list[str], value_cols: list[str]) -> pd.DataFrame:
    point = df[df["sample_type"] == "point"][keys + value_cols].copy()
    return point.rename(columns={col: f"{col}_point" for col in value_cols})


def portfolio_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    boot = portfolio[portfolio["sample_type"] == "bootstrap"]
    value_cols = ["budget", "revenue", "roas", "portfolio_discrete_mroas"]
    rows = []
    for (scenario_id, scenario_name), grp in boot.groupby(["scenario_id", "scenario_name"]):
        row = {"scenario_id": scenario_id, "scenario_name": scenario_name}
        for col in value_cols:
            vals = _interval(grp[col])
            row[f"{col}_p10"] = vals["p10"]
            row[f"{col}_p50"] = vals["p50"]
            row[f"{col}_p90"] = vals["p90"]
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    point = _point_lookup(portfolio, ["scenario_id", "scenario_name"], value_cols)
    return summary.merge(point, on=["scenario_id", "scenario_name"], how="left")


def _component_score(value: float | int | None) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(np.clip(value, 0.0, 100.0))


def _reliability_band(score: float | int | None) -> str:
    if score is None or pd.isna(score):
        return "n/a"
    if score >= 80:
        return "High"
    if score >= 60:
        return "Medium"
    return "Low"


def add_reliability_scores(
    account_summary: pd.DataFrame,
    account_quality: dict | None,
    min_mroas: float,
) -> pd.DataFrame:
    if account_summary.empty:
        return account_summary

    result = account_summary.copy()
    account_quality = account_quality or {}
    result["r2"] = result["account"].map(
        lambda account: account_quality.get(account, {}).get("r2", np.nan)
    )
    result["clean_training_weeks"] = result["account"].map(
        lambda account: account_quality.get(account, {}).get("clean_training_weeks", np.nan)
    )
    result["calibration_factor"] = result["account"].map(
        lambda account: account_quality.get(account, {}).get("calibration_factor", np.nan)
    )

    interval_width = result["inst_mroas_p90"] - result["inst_mroas_p10"]
    interval_base = result["inst_mroas_p50"].abs().clip(lower=min_mroas)
    result["mroas_interval_width"] = interval_width
    result["mroas_interval_width_pct"] = interval_width / interval_base.replace(0, np.nan)

    result["curve_quality_score"] = result["r2"].clip(lower=0, upper=1) * 100
    result["data_volume_score"] = (result["clean_training_weeks"] / 24).clip(upper=1) * 100
    result["calibration_stability_score"] = (
        100 - (result["calibration_factor"] - 1.0).abs() / 0.30 * 50
    ).clip(lower=0, upper=100)
    result["interval_tightness_score"] = (
        100 - result["mroas_interval_width_pct"] / 1.0 * 100
    ).clip(lower=0, upper=100)
    result["floor_safety_score"] = (
        100 - result["prob_mroas_below_floor"] / 0.25 * 100
    ).clip(lower=0, upper=100)

    components = [
        ("curve_quality_score", 0.25),
        ("data_volume_score", 0.20),
        ("calibration_stability_score", 0.20),
        ("interval_tightness_score", 0.20),
        ("floor_safety_score", 0.15),
    ]
    weighted = sum(result[col].fillna(0) * weight for col, weight in components)
    available_weight = sum(
        result[col].notna().astype(float) * weight for col, weight in components
    )
    result["reliability_score"] = weighted / available_weight.replace(0, np.nan)
    result["reliability_score"] = result["reliability_score"].apply(_component_score)
    result["reliability_band"] = result["reliability_score"].apply(_reliability_band)
    return result


def recommended_account_summary(
    accounts: pd.DataFrame,
    min_mroas: float,
    account_quality: dict | None = None,
) -> pd.DataFrame:
    boot = accounts[(accounts["sample_type"] == "bootstrap") & (accounts["recommended"])]
    value_cols = ["spend", "revenue", "roas", "inst_mroas"]
    rows = []
    for account, grp in boot.groupby("account"):
        row = {"account": account, "samples": len(grp)}
        for col in value_cols:
            vals = _interval(grp[col])
            row[f"{col}_p10"] = vals["p10"]
            row[f"{col}_p50"] = vals["p50"]
            row[f"{col}_p90"] = vals["p90"]
        row["prob_mroas_below_floor"] = float((grp["inst_mroas"] < min_mroas).mean())
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    point = _point_lookup(
        accounts[accounts["recommended"]],
        ["account"],
        value_cols + ["scenario_id"],
    )
    summary = summary.merge(point, on="account", how="left")
    return add_reliability_scores(summary, account_quality, min_mroas)


def display_results(
    portfolio: pd.DataFrame,
    accounts: pd.DataFrame,
    metadata: dict,
    min_mroas: float,
) -> None:
    successful = metadata["successful_iterations"]
    failures = metadata["failures"]
    print()
    print("Bootstrap run")
    print("-------------")
    print(f"Successful bootstrap iterations : {successful}")
    print(f"Failed bootstrap iterations     : {len(failures)}")
    print(f"Training window                 : {metadata['train_start'].date()} to {metadata['train_end'].date()}")
    print(f"Training rows / weeks           : {metadata['training_rows']:,} / {metadata['training_weeks']}")
    if metadata["calibration_start"] is not None:
        print(
            "Calibration window              : "
            f"{metadata['calibration_start'].date()} to {metadata['calibration_end'].date()}"
        )
    print(f"Outlier weeks removed           : {metadata['outliers_removed']}")

    if successful == 0:
        print("No successful bootstrap samples. Cannot build confidence intervals.")
        return

    portfolio_ci = portfolio_summary(portfolio)
    scenario_order = {"A": 0, "B": 1, "C": 2, "C1": 2, "D": 3}
    portfolio_ci["_order"] = portfolio_ci["scenario_id"].map(scenario_order).fillna(99)
    portfolio_ci = portfolio_ci.sort_values(["_order", "scenario_id"])

    _print_table(
        "Portfolio Scenario Confidence Intervals",
        portfolio_ci.to_dict("records"),
        [
            ("scenario_id", "ID", "str"),
            ("scenario_name", "Scenario", "str"),
            ("budget_point", "Budget Point", "eur"),
            ("revenue_point", "Rev Point", "eur"),
            ("revenue_p10", "Rev P10", "eur"),
            ("revenue_p50", "Rev P50", "eur"),
            ("revenue_p90", "Rev P90", "eur"),
            ("roas_point", "ROAS Point", "roas"),
            ("roas_p10", "ROAS P10", "roas"),
            ("roas_p50", "ROAS P50", "roas"),
            ("roas_p90", "ROAS P90", "roas"),
        ],
    )

    account_ci = recommended_account_summary(
        accounts,
        min_mroas,
        metadata.get("account_quality"),
    )
    if account_ci.empty:
        print()
        print("No recommended scenario account rows found.")
        return

    account_ci = account_ci.sort_values(
        ["reliability_score", "prob_mroas_below_floor", "spend_point"],
        ascending=[True, False, False],
        na_position="last",
    )
    _print_table(
        f"Recommended Scenario Account Risk (mROAS floor {min_mroas:.2f}x)",
        account_ci.to_dict("records"),
        [
            ("account", "Account", "str"),
            ("reliability_score", "Reliability", "score"),
            ("reliability_band", "Band", "str"),
            ("scenario_id_point", "Scen", "str"),
            ("spend_point", "Spend Point", "eur"),
            ("spend_p10", "Spend P10", "eur"),
            ("spend_p50", "Spend P50", "eur"),
            ("spend_p90", "Spend P90", "eur"),
            ("inst_mroas_point", "mROAS Point", "roas"),
            ("inst_mroas_p10", "mROAS P10", "roas"),
            ("inst_mroas_p50", "mROAS P50", "roas"),
            ("inst_mroas_p90", "mROAS P90", "roas"),
            ("prob_mroas_below_floor", "Prob < Floor", "pct"),
        ],
    )

    risk_rows = []
    for row in account_ci.to_dict("records"):
        probability = row["prob_mroas_below_floor"]
        if probability >= 0.25:
            level = "high"
        elif probability >= 0.10:
            level = "medium"
        else:
            level = "low"
        risk_rows.append(
            {
                "account": row["account"],
                "risk": level,
                "reliability_score": row.get("reliability_score"),
                "reliability_band": row.get("reliability_band"),
                "prob_mroas_below_floor": probability,
                "mroas_p10": row["inst_mroas_p10"],
                "mroas_p50": row["inst_mroas_p50"],
                "mroas_p90": row["inst_mroas_p90"],
            }
        )

    _print_table(
        "mROAS Reliability Flags",
        risk_rows,
        [
            ("account", "Account", "str"),
            ("reliability_score", "Reliability", "score"),
            ("reliability_band", "Band", "str"),
            ("risk", "Risk", "str"),
            ("prob_mroas_below_floor", "Prob < Floor", "pct"),
            ("mroas_p10", "P10", "roas"),
            ("mroas_p50", "P50", "roas"),
            ("mroas_p90", "P90", "roas"),
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for budget-solver scenarios."
    )
    parser.add_argument(
        "--data",
        default=str(DATA_PATH),
        help=f"Daily data CSV/XLSX path (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Monthly budget for Scenario B/C. Defaults to latest complete month spend.",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=200,
        help="Number of bootstrap iterations (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--training-months",
        type=_non_negative_int,
        default=6,
        help="Months of history for curve fitting; 0 = full history (default: 6).",
    )
    parser.add_argument(
        "--target",
        choices=["conversion_value", "conversions"],
        default="conversion_value",
        help="Metric to forecast (default: conversion_value).",
    )
    parser.add_argument(
        "--no-outlier-removal",
        action="store_true",
        help="Disable the current solver outlier removal before bootstrapping.",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable trailing-window calibration.",
    )
    parser.add_argument(
        "--calibration-days",
        type=_positive_int,
        default=TRAILING_WINDOW_DAYS,
        help=f"Trailing days used for calibration (default: {TRAILING_WINDOW_DAYS}).",
    )
    parser.add_argument(
        "--calibration-blend",
        type=_calibration_blend,
        default=0.5,
        help="Calibration strength from 0 to 1 (default: 0.5, best backtest candidate).",
    )
    parser.add_argument(
        "--calibration-min",
        type=_positive_float,
        default=None,
        help="Optional lower cap for calibration factor, e.g. 0.7.",
    )
    parser.add_argument(
        "--calibration-max",
        type=_positive_float,
        default=None,
        help="Optional upper cap for calibration factor, e.g. 1.3.",
    )
    parser.add_argument(
        "--min-mroas",
        type=_positive_float,
        default=2.5,
        help="Minimum instantaneous mROAS floor (default: 2.5).",
    )
    parser.add_argument(
        "--baseline-window",
        type=_positive_int,
        default=7,
        help="Days for Scenario A baseline run rate (default: 7).",
    )
    parser.add_argument(
        "--max-account-changes",
        type=_non_negative_int,
        default=0,
        help="Limit optional Scenario C changes; 0 = no limit (default: 0).",
    )
    parser.add_argument(
        "--wow-cap",
        type=_positive_float,
        default=0.20,
        help="Week-over-week phasing warning cap (default: 0.20).",
    )
    parser.add_argument(
        "--no-stability-rules",
        dest="apply_stability",
        action="store_false",
        help="Disable Scenario C stability rules.",
    )
    args = parser.parse_args(argv)

    if (
        args.calibration_min is not None
        and args.calibration_max is not None
        and args.calibration_min > args.calibration_max
    ):
        parser.error("--calibration-min cannot be greater than --calibration-max")

    path = Path(args.data)
    if not path.exists():
        print(f"ERROR: data file not found: {path}", file=sys.stderr)
        return 1

    print(f"Loading data from: {path}")
    try:
        df = _prepare_target(load_data(path), args.target)
        date_col = _date_col(df)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        print("ERROR: no valid dated rows found.", file=sys.stderr)
        return 1

    if args.budget is None:
        try:
            budget, budget_source = _latest_complete_month_budget(df, date_col)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        budget = args.budget
        budget_source = "command line"

    print()
    print("Confidence configuration")
    print("------------------------")
    print(f"Date range          : {df[date_col].min().date()} to {df[date_col].max().date()}")
    print(f"Accounts            : {df['account_name'].nunique()}")
    print(f"Budget              : {_fmt_eur(budget)} ({budget_source})")
    print(f"Iterations          : {args.iterations}")
    print(f"Training months     : {args.training_months}")
    print(f"Outlier removal     : {'off' if args.no_outlier_removal else 'on'}")
    print(f"Calibration         : {'off' if args.no_calibrate else 'on'}")
    if not args.no_calibrate:
        print(f"Calibration days    : {args.calibration_days}")
        print(f"Calibration blend   : {args.calibration_blend:.2f}")
    print(f"mROAS floor         : {args.min_mroas:.2f}x")
    print()
    print("Metric note: intervals are P10/P50/P90 across bootstrap scenario reruns.")

    portfolio, accounts, metadata = run_bootstrap(
        df=df,
        date_col=date_col,
        budget=budget,
        iterations=args.iterations,
        seed=args.seed,
        training_months=args.training_months,
        outlier_removal=not args.no_outlier_removal,
        calibrate=not args.no_calibrate,
        calibration_days=args.calibration_days,
        calibration_blend=args.calibration_blend,
        calibration_min=args.calibration_min,
        calibration_max=args.calibration_max,
        min_mroas=args.min_mroas,
        baseline_window=args.baseline_window,
        max_account_changes=args.max_account_changes,
        wow_cap=args.wow_cap,
        apply_stability=args.apply_stability,
    )
    display_results(portfolio, accounts, metadata, args.min_mroas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
