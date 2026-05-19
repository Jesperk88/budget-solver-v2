"""
Rolling monthly backtests for the budget solver forecast model.

This module evaluates prediction accuracy without exporting results. It trains
the same response-curve model used by the solver on historical data available
before each forecast month, then predicts revenue at the spend that actually
happened in that month.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

from budget_solver.constants import DATA_PATH, TRAILING_WINDOW_DAYS, WEEKS_PER_MONTH
from budget_solver.curves import (
    fit_portfolio_curves,
    log_curve,
    make_safe_predictor,
    power_curve,
    suppress_curve_fit_warnings,
)
from budget_solver.data import (
    aggregate_weekly,
    apply_demand_normalization,
    build_demand_index,
    load_data,
    remove_outliers,
)
from budget_solver.diagnostics import demand_index_by_proxy, weekly_metrics
from budget_solver.scenarios import build_scenarios

MODEL_FAMILIES = [
    "current_log",
    "power",
    "saturating",
    "piecewise",
    "recency_log",
    "robust_log",
]

ALLOCATION_MOVE_THRESHOLD_EUR = 1000.0
ALLOCATION_GROUP_KEYS = ["model_family", "training_months", "eval_training_months"]


def _parse_training_windows(raw: str) -> list[int]:
    """Parse '3,6,12' into a sorted list of unique month windows."""
    windows = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "--training-months must be a comma-separated list of integers"
            ) from None
        if value < 0:
            raise argparse.ArgumentTypeError("training windows must be >= 0")
        windows.append(value)
    if not windows:
        raise argparse.ArgumentTypeError("at least one training window is required")
    return sorted(set(windows))


def _parse_model_families(raw: str) -> list[str]:
    """Parse 'current_log,power' or 'all' into model family names."""
    if raw.strip().lower() == "all":
        return MODEL_FAMILIES.copy()

    families = []
    for part in raw.split(","):
        family = part.strip()
        if not family:
            continue
        if family not in MODEL_FAMILIES:
            raise argparse.ArgumentTypeError(
                f"unknown model family: {family}. Choices: {', '.join(MODEL_FAMILIES)}"
            )
        families.append(family)
    if not families:
        raise argparse.ArgumentTypeError("at least one model family is required")
    return list(dict.fromkeys(families))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
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
        return f"{sign}€{value / 1_000_000:.2f}M"
    return f"{sign}€{value:,.0f}"


def _fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:+.1f}%"


def _fmt_pct_abs(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _fmt_roas(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}x"


def _print_table(title: str, rows: list[dict], columns: list[tuple[str, str, str]]) -> None:
    """
    Print a compact fixed-width table.

    columns: list of (key, label, format_name), where format_name is one of
    'str', 'int', 'eur', 'pct', 'roas', 'float'.
    """
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
            return _fmt_pct(value)
        if fmt == "pct_abs":
            return _fmt_pct_abs(value)
        if fmt == "roas":
            return _fmt_roas(value)
        if fmt == "int":
            return "n/a" if value is None or pd.isna(value) else f"{int(value):,}"
        if fmt == "float":
            return "n/a" if value is None or pd.isna(value) else f"{float(value):.3f}"
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

    header = "  ".join(label.ljust(widths[i]) for i, (_, label, _) in enumerate(columns))
    print(header)
    print("  ".join("-" * width for width in widths))
    for values in formatted:
        cells = []
        for i, text in enumerate(values):
            align_left = columns[i][2] == "str"
            cells.append(text.ljust(widths[i]) if align_left else text.rjust(widths[i]))
        print("  ".join(cells))


def _month_midpoint_iso_week(month: pd.Period) -> int:
    start = month.to_timestamp(how="start")
    end = month.to_timestamp(how="end").normalize()
    midpoint = start + pd.Timedelta(days=(end.day - 1) // 2)
    return int(midpoint.isocalendar().week)


def _complete_months(df: pd.DataFrame, date_col: str, include_partial: bool) -> list[pd.Period]:
    months = sorted(df[date_col].dt.to_period("M").dropna().unique())
    if include_partial or not months:
        return months

    latest_date = df[date_col].max().normalize()
    return [
        month
        for month in months
        if month.to_timestamp(how="end").normalize() <= latest_date
    ]


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
        raise ValueError("--target conversions requires a conversions or conversions_adj column")
    return df


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _clean_curve_data(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spend = np.asarray(data["spend"], dtype=float)
    revenue = np.asarray(data["revenue"], dtype=float)
    weeks = np.asarray(data["_week"])
    mask = (spend > 0) & (revenue >= 0) & np.isfinite(spend) & np.isfinite(revenue)
    return spend[mask], revenue[mask], weeks[mask]


def _linear_fallback(spend: np.ndarray, revenue: np.ndarray) -> tuple:
    avg_roas = float((revenue / spend).mean()) if len(spend) and spend.mean() > 0 else 0.0
    raw_fn = lambda x, r=avg_roas: r * np.asarray(x, dtype=float)
    min_spend = float(spend.min()) if len(spend) else 0.0
    anchor = float(raw_fn(min_spend)) if len(spend) else 0.0
    return make_safe_predictor(raw_fn, min_spend, anchor), [avg_roas, 1.0], 0.0, "linear_fallback"


def _saturating_curve(x, vmax, k):
    x = np.asarray(x, dtype=float)
    return vmax * x / (k + np.maximum(x, 1e-9))


def _fit_saturating(data: dict) -> tuple:
    spend, revenue, _ = _clean_curve_data(data)
    if len(spend) < 3:
        return _linear_fallback(spend, revenue)
    try:
        p0 = [max(float(revenue.max()) * 1.5, 1.0), max(float(np.median(spend)), 1.0)]
        with suppress_curve_fit_warnings():
            params, _ = curve_fit(
                _saturating_curve,
                spend,
                revenue,
                p0=p0,
                bounds=([0.0, 1e-9], [np.inf, np.inf]),
                maxfev=20000,
            )
        raw_fn = lambda x, p=params: _saturating_curve(x, *p)
        min_spend = float(spend.min())
        anchor = float(np.maximum(raw_fn(min_spend), 0.0))
        fn = make_safe_predictor(raw_fn, min_spend, anchor)
        return fn, params, _r_squared(revenue, raw_fn(spend)), "saturating"
    except Exception:
        return _linear_fallback(spend, revenue)


def _week_ordinal(week) -> int:
    if hasattr(week, "ordinal"):
        return int(week.ordinal)
    try:
        return int(pd.Period(str(week), freq="W").ordinal)
    except Exception:
        return 0


def _fit_recency_weighted_log(data: dict, half_life_weeks: float = 8.0) -> tuple:
    spend, revenue, weeks = _clean_curve_data(data)
    if len(spend) < 3:
        return _linear_fallback(spend, revenue)

    ordinals = np.array([_week_ordinal(week) for week in weeks], dtype=float)
    age = ordinals.max() - ordinals
    weights = np.power(0.5, age / half_life_weeks)
    sigma = 1 / np.sqrt(np.maximum(weights, 1e-6))

    try:
        with suppress_curve_fit_warnings():
            params, _ = curve_fit(
                log_curve,
                spend,
                revenue,
                p0=[revenue.mean(), 0],
                sigma=sigma,
                absolute_sigma=False,
                maxfev=20000,
            )
        if params[0] <= 0:
            return _linear_fallback(spend, revenue)
        raw_fn = lambda x, p=params: log_curve(x, *p)
        min_spend = float(spend.min())
        anchor = float(np.maximum(raw_fn(min_spend), 0.0))
        fn = make_safe_predictor(raw_fn, min_spend, anchor)
        return fn, params, _r_squared(revenue, raw_fn(spend)), "recency_log"
    except Exception:
        return _linear_fallback(spend, revenue)


def _fit_robust_log(data: dict) -> tuple:
    spend, revenue, _ = _clean_curve_data(data)
    if len(spend) < 3:
        return _linear_fallback(spend, revenue)

    scale = float(np.median(np.abs(revenue - np.median(revenue))))
    scale = max(scale, float(np.std(revenue)), 1.0)
    try:
        result = least_squares(
            lambda p: log_curve(spend, p[0], p[1]) - revenue,
            x0=np.array([max(revenue.mean(), 1.0), 0.0]),
            bounds=([1e-9, -np.inf], [np.inf, np.inf]),
            loss="soft_l1",
            f_scale=scale,
            max_nfev=20000,
        )
        if not result.success or result.x[0] <= 0:
            return _linear_fallback(spend, revenue)
        params = result.x
        raw_fn = lambda x, p=params: log_curve(x, *p)
        min_spend = float(spend.min())
        anchor = float(np.maximum(raw_fn(min_spend), 0.0))
        fn = make_safe_predictor(raw_fn, min_spend, anchor)
        return fn, params, _r_squared(revenue, raw_fn(spend)), "robust_log"
    except Exception:
        return _linear_fallback(spend, revenue)


def _fit_piecewise(data: dict, max_bins: int = 6) -> tuple:
    spend, revenue, _ = _clean_curve_data(data)
    if len(spend) < 3:
        return _linear_fallback(spend, revenue)

    try:
        frame = pd.DataFrame({"spend": spend, "revenue": revenue}).sort_values("spend")
        n_bins = min(max_bins, max(3, len(frame) // 4))
        if frame["spend"].nunique() > n_bins:
            frame["_bin"] = pd.qcut(frame["spend"], q=n_bins, duplicates="drop")
            points = (
                frame.groupby("_bin", observed=True)
                .agg(spend=("spend", "median"), revenue=("revenue", "median"))
                .reset_index(drop=True)
                .sort_values("spend")
            )
        else:
            points = (
                frame.groupby("spend", as_index=False)
                .agg(revenue=("revenue", "mean"))
                .sort_values("spend")
            )

        xs = points["spend"].to_numpy(dtype=float)
        ys = points["revenue"].to_numpy(dtype=float)
        if len(xs) < 2 or np.any(np.diff(xs) <= 0):
            return _linear_fallback(spend, revenue)

        def raw_fn(x):
            scalar_input = np.isscalar(x)
            arr = np.atleast_1d(np.asarray(x, dtype=float))
            pred = np.interp(arr, xs, ys)

            below = arr < xs[0]
            if np.any(below):
                pred[below] = (arr[below] / xs[0]) * max(ys[0], 0.0)

            above = arr > xs[-1]
            if np.any(above):
                slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
                slope = max(float(slope), 0.0)
                pred[above] = ys[-1] + slope * (arr[above] - xs[-1])

            pred = np.maximum(pred, 0.0)
            return float(pred[0]) if scalar_input else pred

        min_spend = float(xs[0])
        anchor = float(max(ys[0], 0.0))
        fn = make_safe_predictor(raw_fn, min_spend, anchor)
        return fn, [float(v) for pair in zip(xs, ys) for v in pair], _r_squared(revenue, raw_fn(spend)), "piecewise"
    except Exception:
        return _linear_fallback(spend, revenue)


def _fit_experimental_family(account_data: dict, model_family: str) -> dict:
    results = {}
    for account, data in account_data.items():
        if model_family == "saturating":
            results[account] = _fit_saturating(data)
        elif model_family == "piecewise":
            results[account] = _fit_piecewise(data)
        elif model_family == "recency_log":
            results[account] = _fit_recency_weighted_log(data)
        elif model_family == "robust_log":
            results[account] = _fit_robust_log(data)
        else:
            raise ValueError(f"unsupported experimental model family: {model_family}")
    return results


def _fit_model_family(account_data: dict, model_family: str, verbose: bool) -> dict:
    if model_family == "current_log":
        if verbose:
            return fit_portfolio_curves(account_data, preferred_model="log")
        with contextlib.redirect_stdout(io.StringIO()):
            return fit_portfolio_curves(account_data, preferred_model="log")

    if model_family == "power":
        if verbose:
            return fit_portfolio_curves(account_data, preferred_model="power")
        with contextlib.redirect_stdout(io.StringIO()):
            return fit_portfolio_curves(account_data, preferred_model="power")

    return _fit_experimental_family(account_data, model_family)


def _scaled_params_for_calibration(params, model_name: str, scale: float) -> list[float]:
    """Scale fitted curve parameters so Scenario C/D mROAS matches calibrated revenue."""
    scaled = [float(p) for p in params]
    if not scaled:
        return scaled

    base_model = model_name.replace("+cal", "")
    if base_model == "power":
        scaled[0] *= scale
    elif base_model in {"log", "linear_fallback"}:
        scaled = [p * scale for p in scaled]
    return scaled


def _iqr_bounds(values: np.ndarray, multiplier: float) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return None
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def _log_outlier(
    account: str,
    week,
    spend: float,
    revenue: float,
    roas: float,
    reason: str,
) -> dict:
    return {
        "account": account,
        "week": str(week),
        "spend": round(float(spend), 2),
        "revenue": round(float(revenue), 2),
        "roas": round(float(roas), 3),
        "reason": reason,
    }


def _apply_backtest_outlier_strategy(
    account_data: dict,
    method: str,
    min_spend_pct: float = 0.20,
    roas_iqr_mult: float = 2.0,
) -> tuple[dict, list[dict]]:
    """
    Apply experimental outlier strategies for backtest comparison.

    current: delegate to production remove_outliers().
    none: keep all observations.
    low-spend: drop only weeks below min_spend_pct x median spend.
    roas-iqr: drop only ROAS IQR outliers.
    winsorize-roas: keep all weeks, cap ROAS outliers to IQR bounds.
    low-spend-winsorize: drop low-spend weeks, then cap ROAS outliers.
    """
    if method == "current":
        return remove_outliers(account_data, min_spend_pct=min_spend_pct, roas_iqr_mult=roas_iqr_mult)
    if method == "none":
        return account_data, []

    cleaned = {}
    log = []

    for account, data in account_data.items():
        spend = np.asarray(data["spend"], dtype=float)
        revenue = np.asarray(data["revenue"], dtype=float)
        weeks = np.asarray(data["_week"])
        keep = np.ones(len(spend), dtype=bool)
        adjusted_revenue = revenue.copy()

        if method in {"low-spend", "low-spend-winsorize"}:
            median_spend = np.median(spend[spend > 0]) if np.any(spend > 0) else 1.0
            threshold = median_spend * min_spend_pct
            low_mask = spend < threshold
            with np.errstate(divide="ignore", invalid="ignore"):
                roas = np.where(spend > 0, revenue / spend, 0.0)
            for i in np.where(low_mask)[0]:
                log.append(
                    _log_outlier(
                        account,
                        weeks[i],
                        spend[i],
                        revenue[i],
                        roas[i],
                        f"low spend (<{min_spend_pct:.0%} of median)",
                    )
                )
            keep &= ~low_mask

        if method == "roas-iqr":
            with np.errstate(divide="ignore", invalid="ignore"):
                roas = np.where(spend > 0, revenue / spend, np.nan)
            bounds = _iqr_bounds(roas[spend > 0], roas_iqr_mult)
            if bounds:
                lo, hi = bounds
                outlier_mask = (spend > 0) & ((roas < lo) | (roas > hi))
                for i in np.where(outlier_mask)[0]:
                    log.append(
                        _log_outlier(
                            account,
                            weeks[i],
                            spend[i],
                            revenue[i],
                            roas[i],
                            f"ROAS outlier dropped (IQRx{roas_iqr_mult}: bounds {lo:.2f}-{hi:.2f})",
                        )
                    )
                keep &= ~outlier_mask

        if method in {"winsorize-roas", "low-spend-winsorize"}:
            with np.errstate(divide="ignore", invalid="ignore"):
                roas = np.where(spend > 0, adjusted_revenue / spend, np.nan)
            bounds = _iqr_bounds(roas[(spend > 0) & keep], roas_iqr_mult)
            if bounds:
                lo, hi = bounds
                clip_mask = keep & (spend > 0) & ((roas < lo) | (roas > hi))
                for i in np.where(clip_mask)[0]:
                    clipped_roas = min(max(float(roas[i]), lo), hi)
                    adjusted_revenue[i] = clipped_roas * spend[i]
                    log.append(
                        _log_outlier(
                            account,
                            weeks[i],
                            spend[i],
                            revenue[i],
                            roas[i],
                            f"ROAS winsorized to {clipped_roas:.2f} (bounds {lo:.2f}-{hi:.2f})",
                        )
                    )

        if method not in {
            "low-spend",
            "roas-iqr",
            "winsorize-roas",
            "low-spend-winsorize",
        }:
            raise ValueError(f"unknown outlier method: {method}")

        cleaned[account] = {
            "spend": spend[keep],
            "revenue": adjusted_revenue[keep],
            "_week": weeks[keep],
        }

    return cleaned, log


def _build_backtest_demand_index(
    train_df: pd.DataFrame,
    account_data: dict,
    proxy: str,
    smoothing_weeks: int,
) -> dict:
    """
    Build demand index for backtesting.

    roas_legacy preserves the production solver's current ROAS-based index.
    Other proxies use weekly metrics from diagnostics.py and can be smoothed.
    """
    if proxy == "roas_legacy":
        return build_demand_index(account_data)

    weekly_input = train_df.copy()
    if "_date" not in weekly_input.columns:
        date_col = next((c for c in ("date", "week_start", "week") if c in weekly_input.columns), None)
        if not date_col:
            return {}
        weekly_input["_date"] = pd.to_datetime(weekly_input[date_col], errors="coerce")
        weekly_input = weekly_input.dropna(subset=["_date"])
    weekly = weekly_metrics(weekly_input)
    index_df = demand_index_by_proxy(
        weekly,
        proxy=proxy,
        smoothing_weeks=smoothing_weeks,
    )
    if index_df.empty:
        return {}
    return dict(zip(index_df["iso_week"].astype(int), index_df["index"].astype(float)))


def _build_monthly_predictors(
    train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    forecast_week: int,
    training_months: int,
    normalize_demand: bool,
    outlier_removal: bool,
    calibrate: bool,
    calibration_blend: float,
    calibration_min: float | None,
    calibration_max: float | None,
    verbose: bool,
    outlier_method: str = "current",
    demand_proxy: str = "roas_legacy",
    demand_smoothing_weeks: int = 1,
    model_family: str = "current_log",
) -> tuple[dict, dict, list[dict], dict]:
    """
    Fit monthly-scale prediction functions using only data before the forecast month.
    """
    train_account_data = aggregate_weekly(train_df)

    demand_index = _build_backtest_demand_index(
        train_df,
        train_account_data,
        proxy=demand_proxy,
        smoothing_weeks=demand_smoothing_weeks,
    )
    fitting_data = train_account_data

    removal_log: list[dict] = []
    active_outlier_method = outlier_method if outlier_removal else "none"
    fitting_data, removal_log = _apply_backtest_outlier_strategy(
        fitting_data,
        active_outlier_method,
    )

    if normalize_demand:
        fitting_data = apply_demand_normalization(fitting_data, demand_index)

    portfolio_results = _fit_model_family(fitting_data, model_family, verbose)

    predict_fns = {}
    model_info = {}
    for account, (fn, params, r2, model_name) in portfolio_results.items():
        base_fn = fn
        if normalize_demand:
            demand = demand_index.get(forecast_week, 1.0)
            base_fn = lambda x, fn=fn, d=demand: fn(x) * d

        predict_fns[account] = (
            lambda x, fn=base_fn, wpm=WEEKS_PER_MONTH: wpm * fn(x / wpm)
        )
        model_info[account] = (fn, params, r2, model_name)

    calibration_factors = {}
    if calibrate and len(calibration_df):
        actual_spend = calibration_df.groupby("account_name")["cost"].sum().to_dict()
        actual_revenue = calibration_df.groupby("account_name")["conversion_value"].sum().to_dict()

        for account in list(predict_fns):
            spend = actual_spend.get(account, 0.0)
            revenue = actual_revenue.get(account, 0.0)
            if spend <= 0 or revenue <= 0:
                continue

            actual_roas = revenue / spend
            model_pred = predict_fns[account](spend)
            model_roas = model_pred / spend if spend > 0 else 0.0
            if model_roas <= 0:
                continue

            raw_scale = actual_roas / model_roas
            capped_scale = raw_scale
            if calibration_min is not None:
                capped_scale = max(capped_scale, calibration_min)
            if calibration_max is not None:
                capped_scale = min(capped_scale, calibration_max)

            # Blend toward neutral 1.0 so noisy trailing windows do not fully
            # override the fitted curve. blend=1.0 preserves the old behavior.
            scale = 1.0 + calibration_blend * (capped_scale - 1.0)
            calibration_factors[account] = scale
            predict_fns[account] = (
                lambda x, fn=predict_fns[account], s=scale: fn(x) * s
            )

            fn_i, params_i, r2_i, model_name_i = model_info[account]
            model_info[account] = (
                fn_i,
                _scaled_params_for_calibration(params_i, model_name_i, scale),
                r2_i,
                f"{model_name_i}+cal",
            )

    metadata = {
        "training_months": training_months,
        "demand_index": demand_index,
        "demand_proxy": demand_proxy,
        "demand_smoothing_weeks": demand_smoothing_weeks,
        "model_family": model_family,
        "calibration_factors": calibration_factors,
    }
    return predict_fns, model_info, removal_log, metadata


def _run_one_backtest(
    df: pd.DataFrame,
    date_col: str,
    forecast_month: pd.Period,
    training_months: int,
    normalize_demand: bool,
    outlier_removal: bool,
    calibrate: bool,
    calibration_days: int,
    calibration_blend: float,
    calibration_min: float | None,
    calibration_max: float | None,
    min_train_weeks: int,
    verbose: bool,
    outlier_method: str = "current",
    demand_proxy: str = "roas_legacy",
    demand_smoothing_weeks: int = 1,
    model_family: str = "current_log",
) -> tuple[list[dict], dict | None]:
    forecast_start = forecast_month.to_timestamp(how="start")
    forecast_end = forecast_month.to_timestamp(how="end").normalize()

    if training_months == 0:
        train_start = df[date_col].min().normalize()
    else:
        train_start = forecast_start - pd.DateOffset(months=training_months)

    train_df = df[(df[date_col] >= train_start) & (df[date_col] < forecast_start)].copy()
    actual_df = df[(df[date_col] >= forecast_start) & (df[date_col] <= forecast_end)].copy()

    calibration_start = forecast_start - pd.Timedelta(days=calibration_days)
    calibration_df = df[
        (df[date_col] >= calibration_start)
        & (df[date_col] < forecast_start)
    ].copy()

    if train_df.empty or actual_df.empty:
        return [], None

    train_weeks = train_df[date_col].dt.to_period("W").nunique()
    if train_weeks < min_train_weeks:
        return [], {
            "month": str(forecast_month),
            "training_months": training_months,
            "skipped": True,
            "reason": f"only {train_weeks} training weeks",
        }

    forecast_week = _month_midpoint_iso_week(forecast_month)
    predict_fns, model_info, removal_log, metadata = _build_monthly_predictors(
        train_df=train_df,
        calibration_df=calibration_df,
        forecast_week=forecast_week,
        training_months=training_months,
        normalize_demand=normalize_demand,
        outlier_removal=outlier_removal,
        calibrate=calibrate,
        calibration_blend=calibration_blend,
        calibration_min=calibration_min,
        calibration_max=calibration_max,
        verbose=verbose,
        outlier_method=outlier_method,
        demand_proxy=demand_proxy,
        demand_smoothing_weeks=demand_smoothing_weeks,
        model_family=model_family,
    )

    actual_by_account = (
        actual_df.groupby("account_name")
        .agg(actual_spend=("cost", "sum"), actual_revenue=("conversion_value", "sum"))
        .reset_index()
    )

    rows = []
    for record in actual_by_account.to_dict("records"):
        account = record["account_name"]
        if account not in predict_fns:
            continue

        actual_spend = float(record["actual_spend"])
        actual_revenue = float(record["actual_revenue"])
        predicted_revenue = float(predict_fns[account](actual_spend))
        error = actual_revenue - predicted_revenue

        _, _, r2, model_name = model_info[account]
        rows.append(
            {
                "month": str(forecast_month),
                "model_family": model_family,
                "training_months": training_months,
                "account": account,
                "actual_spend": actual_spend,
                "predicted_revenue": predicted_revenue,
                "actual_revenue": actual_revenue,
                "error": error,
                "error_pct_vs_pred": error / predicted_revenue if predicted_revenue else np.nan,
                "abs_error": abs(error),
                "actual_roas": actual_revenue / actual_spend if actual_spend else np.nan,
                "predicted_roas": predicted_revenue / actual_spend if actual_spend else np.nan,
                "r2": r2,
                "model": model_name,
                "calibration_factor": metadata["calibration_factors"].get(account, 1.0),
                "outliers_removed": sum(1 for row in removal_log if row["account"] == account),
            }
        )

    month_summary = {
        "month": str(forecast_month),
        "model_family": model_family,
        "training_months": training_months,
        "skipped": False,
        "train_start": train_start.date().isoformat(),
        "train_end": (forecast_start - pd.Timedelta(days=1)).date().isoformat(),
        "forecast_week": forecast_week,
        "train_weeks": train_weeks,
        "calibration_days": calibration_days,
        "calibration_blend": calibration_blend,
        "calibration_min": calibration_min,
        "calibration_max": calibration_max,
        "outlier_method": outlier_method if outlier_removal else "none",
        "demand_proxy": demand_proxy,
        "demand_smoothing_weeks": demand_smoothing_weeks,
    }
    return rows, month_summary


def run_backtests(
    df: pd.DataFrame,
    date_col: str,
    training_windows: list[int],
    months_back: int,
    normalize_demand: bool,
    outlier_removal: bool,
    calibrate: bool,
    calibration_days: int,
    calibration_blend: float,
    calibration_min: float | None,
    calibration_max: float | None,
    include_partial_months: bool,
    min_train_weeks: int,
    verbose: bool,
    outlier_method: str = "current",
    demand_proxy: str = "roas_legacy",
    demand_smoothing_weeks: int = 1,
    model_families: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    model_families = model_families or ["current_log"]
    months = _complete_months(df, date_col, include_partial_months)
    if not months:
        return pd.DataFrame(), []

    max_window = max(training_windows)
    first_allowed = (
        df[date_col].min().normalize() + pd.DateOffset(months=max_window)
        if max_window > 0
        else df[date_col].min().normalize()
    ).to_period("M")
    months = [month for month in months if month >= first_allowed]

    if months_back > 0:
        months = months[-months_back:]

    all_rows = []
    month_summaries = []
    for model_family in model_families:
        for training_months in training_windows:
            for month in months:
                rows, summary = _run_one_backtest(
                    df=df,
                    date_col=date_col,
                    forecast_month=month,
                    training_months=training_months,
                    normalize_demand=normalize_demand,
                    outlier_removal=outlier_removal,
                    calibrate=calibrate,
                    calibration_days=calibration_days,
                    calibration_blend=calibration_blend,
                    calibration_min=calibration_min,
                    calibration_max=calibration_max,
                    min_train_weeks=min_train_weeks,
                    verbose=verbose,
                    outlier_method=outlier_method,
                    demand_proxy=demand_proxy,
                    demand_smoothing_weeks=demand_smoothing_weeks,
                    model_family=model_family,
                )
                all_rows.extend(rows)
                if summary:
                    month_summaries.append(summary)

    return pd.DataFrame(all_rows), month_summaries


def _scenario_by_id(scenario_set, scenario_id: str):
    for scenario in scenario_set.scenarios:
        if scenario.id == scenario_id:
            return scenario
    return None


def _recommended_scenario(scenario_set):
    for scenario in scenario_set.scenarios:
        if scenario.recommended:
            return scenario
    return _scenario_by_id(scenario_set, "C")


def _allocation_dict(scenario) -> dict:
    return {alloc.account: alloc for alloc in scenario.allocations}


def _run_one_allocation_quality_backtest(
    df: pd.DataFrame,
    date_col: str,
    forecast_month: pd.Period,
    training_months: int,
    normalize_demand: bool,
    outlier_removal: bool,
    calibrate: bool,
    calibration_days: int,
    calibration_blend: float,
    calibration_min: float | None,
    calibration_max: float | None,
    min_train_weeks: int,
    verbose: bool,
    outlier_method: str,
    demand_proxy: str,
    demand_smoothing_weeks: int,
    model_family: str,
    eval_training_months: int | None,
    min_mroas: float,
    baseline_window: int,
    max_account_changes: int,
    wow_cap: float,
    apply_stability: bool,
) -> tuple[dict | None, list[dict], dict | None]:
    forecast_start = forecast_month.to_timestamp(how="start")
    forecast_end = forecast_month.to_timestamp(how="end").normalize()

    if training_months == 0:
        train_start = df[date_col].min().normalize()
    else:
        train_start = forecast_start - pd.DateOffset(months=training_months)

    train_df = df[(df[date_col] >= train_start) & (df[date_col] < forecast_start)].copy()
    actual_df = df[(df[date_col] >= forecast_start) & (df[date_col] <= forecast_end)].copy()

    calibration_start = forecast_start - pd.Timedelta(days=calibration_days)
    calibration_df = df[
        (df[date_col] >= calibration_start)
        & (df[date_col] < forecast_start)
    ].copy()

    if train_df.empty or actual_df.empty:
        return None, [], None

    train_weeks = train_df[date_col].dt.to_period("W").nunique()
    if train_weeks < min_train_weeks:
        return None, [], {
            "month": str(forecast_month),
            "model_family": model_family,
            "training_months": training_months,
            "skipped": True,
            "reason": f"only {train_weeks} training weeks",
        }

    eval_training_months = training_months if eval_training_months is None else eval_training_months
    if eval_training_months == 0:
        eval_train_start = df[date_col].min().normalize()
    else:
        eval_train_start = forecast_start - pd.DateOffset(months=eval_training_months)
    eval_train_df = df[
        (df[date_col] >= eval_train_start) & (df[date_col] < forecast_start)
    ].copy()
    eval_train_weeks = eval_train_df[date_col].dt.to_period("W").nunique()
    if eval_train_weeks < min_train_weeks:
        return None, [], {
            "month": str(forecast_month),
            "model_family": model_family,
            "training_months": training_months,
            "eval_training_months": eval_training_months,
            "skipped": True,
            "reason": f"only {eval_train_weeks} evaluation training weeks",
        }

    forecast_week = _month_midpoint_iso_week(forecast_month)
    predict_fns, model_info, removal_log, metadata = _build_monthly_predictors(
        train_df=train_df,
        calibration_df=calibration_df,
        forecast_week=forecast_week,
        training_months=training_months,
        normalize_demand=normalize_demand,
        outlier_removal=outlier_removal,
        calibrate=calibrate,
        calibration_blend=calibration_blend,
        calibration_min=calibration_min,
        calibration_max=calibration_max,
        verbose=verbose,
        outlier_method=outlier_method,
        demand_proxy=demand_proxy,
        demand_smoothing_weeks=demand_smoothing_weeks,
        model_family=model_family,
    )
    if eval_training_months == training_months:
        eval_predict_fns = predict_fns
        eval_model_info = model_info
        eval_removal_log = removal_log
        eval_metadata = metadata
    else:
        eval_predict_fns, eval_model_info, eval_removal_log, eval_metadata = _build_monthly_predictors(
            train_df=eval_train_df,
            calibration_df=calibration_df,
            forecast_week=forecast_week,
            training_months=eval_training_months,
            normalize_demand=normalize_demand,
            outlier_removal=outlier_removal,
            calibrate=calibrate,
            calibration_blend=calibration_blend,
            calibration_min=calibration_min,
            calibration_max=calibration_max,
            verbose=verbose,
            outlier_method=outlier_method,
            demand_proxy=demand_proxy,
            demand_smoothing_weeks=demand_smoothing_weeks,
            model_family=model_family,
        )

    actual_by_account = (
        actual_df.groupby("account_name")
        .agg(actual_spend=("cost", "sum"), actual_revenue=("conversion_value", "sum"))
        .reset_index()
    )
    scored_accounts = set(predict_fns).intersection(eval_predict_fns)
    actual_lookup = {
        row["account_name"]: row
        for row in actual_by_account.to_dict("records")
        if row["account_name"] in scored_accounts
    }
    if not actual_lookup:
        return None, [], {
            "month": str(forecast_month),
            "model_family": model_family,
            "training_months": training_months,
            "skipped": True,
            "reason": "no actual spend rows for modeled accounts",
        }

    target_budget = float(sum(row["actual_spend"] for row in actual_lookup.values()))
    if target_budget <= 0:
        return None, [], {
            "month": str(forecast_month),
            "model_family": model_family,
            "training_months": training_months,
            "skipped": True,
            "reason": "zero actual budget",
        }

    try:
        scenario_set = build_scenarios(
            df=train_df,
            predict_fns=predict_fns,
            model_info=model_info,
            target_budget=target_budget,
            min_mroas=min_mroas,
            baseline_window_days=baseline_window,
            max_account_changes=max_account_changes,
            wow_cap=wow_cap,
            apply_stability=apply_stability,
        )
    except Exception as exc:
        return None, [], {
            "month": str(forecast_month),
            "model_family": model_family,
            "training_months": training_months,
            "skipped": True,
            "reason": f"scenario build failed: {exc}",
        }

    scenario_b = _scenario_by_id(scenario_set, "B")
    scenario_c = _recommended_scenario(scenario_set)
    if scenario_b is None or scenario_c is None:
        return None, [], {
            "month": str(forecast_month),
            "model_family": model_family,
            "training_months": training_months,
            "skipped": True,
            "reason": "scenario B/C missing",
        }

    b_allocs = _allocation_dict(scenario_b)
    c_allocs = _allocation_dict(scenario_c)
    modeled_accounts = sorted(scored_accounts)

    account_rows = []
    actual_pred_revenue = 0.0
    actual_observed_revenue = 0.0
    for account in modeled_accounts:
        actual = actual_lookup.get(account, {})
        actual_spend = float(actual.get("actual_spend", 0.0))
        actual_revenue = float(actual.get("actual_revenue", 0.0))
        actual_pred = float(eval_predict_fns[account](actual_spend)) if actual_spend > 0 else 0.0

        b_alloc = b_allocs.get(account)
        c_alloc = c_allocs.get(account)
        if b_alloc is None or c_alloc is None:
            continue

        delta_c_vs_actual = c_alloc.monthly_spend - actual_spend
        delta_c_vs_b = c_alloc.monthly_spend - b_alloc.monthly_spend
        if delta_c_vs_b > ALLOCATION_MOVE_THRESHOLD_EUR:
            action_vs_b = "increase"
        elif delta_c_vs_b < -ALLOCATION_MOVE_THRESHOLD_EUR:
            action_vs_b = "cut"
        else:
            action_vs_b = "unchanged"

        b_pred_revenue = float(eval_predict_fns[account](b_alloc.monthly_spend))
        c_pred_revenue = float(eval_predict_fns[account](c_alloc.monthly_spend))
        _, _, r2, model_name = model_info[account]
        _, _, eval_r2, eval_model_name = eval_model_info[account]
        account_rows.append(
            {
                "month": str(forecast_month),
                "model_family": model_family,
                "training_months": training_months,
                "eval_training_months": eval_training_months,
                "account": account,
                "actual_spend": actual_spend,
                "actual_observed_revenue": actual_revenue,
                "actual_pred_revenue": actual_pred,
                "scenario_b_spend": b_alloc.monthly_spend,
                "scenario_b_pred_revenue": b_pred_revenue,
                "scenario_c_id": scenario_c.id,
                "scenario_c_spend": c_alloc.monthly_spend,
                "scenario_c_pred_revenue": c_pred_revenue,
                "c_vs_actual_spend": delta_c_vs_actual,
                "c_vs_b_spend": delta_c_vs_b,
                "c_vs_actual_revenue": c_pred_revenue - actual_pred,
                "c_vs_b_revenue": c_pred_revenue - b_pred_revenue,
                "action_vs_b": action_vs_b,
                "actual_roas": actual_revenue / actual_spend if actual_spend else np.nan,
                "actual_pred_roas": actual_pred / actual_spend if actual_spend else np.nan,
                "scenario_c_roas": (
                    c_pred_revenue / c_alloc.monthly_spend
                    if c_alloc.monthly_spend
                    else np.nan
                ),
                "scenario_c_inst_mroas": c_alloc.inst_mroas,
                "r2": r2,
                "model": model_name,
                "eval_r2": eval_r2,
                "eval_model": eval_model_name,
                "calibration_factor": metadata["calibration_factors"].get(account, 1.0),
                "eval_calibration_factor": eval_metadata["calibration_factors"].get(account, 1.0),
                "outliers_removed": sum(1 for row in removal_log if row["account"] == account),
                "eval_outliers_removed": sum(1 for row in eval_removal_log if row["account"] == account),
            }
        )
        actual_pred_revenue += actual_pred
        actual_observed_revenue += actual_revenue

    scenario_b_revenue = sum(row["scenario_b_pred_revenue"] for row in account_rows)
    scenario_c_revenue = sum(row["scenario_c_pred_revenue"] for row in account_rows)
    scenario_b_budget = sum(row["scenario_b_spend"] for row in account_rows)
    scenario_c_budget = sum(row["scenario_c_spend"] for row in account_rows)
    portfolio_row = {
        "month": str(forecast_month),
        "model_family": model_family,
        "training_months": training_months,
        "eval_training_months": eval_training_months,
        "budget": target_budget,
        "actual_observed_revenue": actual_observed_revenue,
        "actual_mix_pred_revenue": actual_pred_revenue,
        "scenario_b_budget": scenario_b_budget,
        "scenario_b_pred_revenue": scenario_b_revenue,
        "scenario_c_id": scenario_c.id,
        "scenario_c_budget": scenario_c_budget,
        "scenario_c_pred_revenue": scenario_c_revenue,
        "unspent_budget": target_budget - scenario_c_budget,
        "c_vs_actual_revenue": scenario_c_revenue - actual_pred_revenue,
        "c_vs_actual_pct": (
            (scenario_c_revenue - actual_pred_revenue) / actual_pred_revenue
            if actual_pred_revenue
            else np.nan
        ),
        "c_vs_b_revenue": scenario_c_revenue - scenario_b_revenue,
        "c_vs_b_pct": (
            (scenario_c_revenue - scenario_b_revenue) / scenario_b_revenue
            if scenario_b_revenue
            else np.nan
        ),
        "actual_observed_roas": actual_observed_revenue / target_budget if target_budget else np.nan,
        "actual_mix_pred_roas": actual_pred_revenue / target_budget if target_budget else np.nan,
        "scenario_b_roas": scenario_b_revenue / scenario_b_budget if scenario_b_budget else np.nan,
        "scenario_c_roas": scenario_c_revenue / scenario_c_budget if scenario_c_budget else np.nan,
        "train_start": train_start.date().isoformat(),
        "train_end": (forecast_start - pd.Timedelta(days=1)).date().isoformat(),
        "eval_train_start": eval_train_start.date().isoformat(),
        "eval_train_end": (forecast_start - pd.Timedelta(days=1)).date().isoformat(),
        "train_weeks": train_weeks,
        "eval_train_weeks": eval_train_weeks,
        "forecast_week": forecast_week,
        "outlier_method": outlier_method if outlier_removal else "none",
        "demand_proxy": demand_proxy,
        "demand_smoothing_weeks": demand_smoothing_weeks,
    }
    return portfolio_row, account_rows, {
        "month": str(forecast_month),
        "model_family": model_family,
        "training_months": training_months,
        "eval_training_months": eval_training_months,
        "skipped": False,
    }


def run_allocation_quality_backtests(
    df: pd.DataFrame,
    date_col: str,
    training_windows: list[int],
    months_back: int,
    normalize_demand: bool,
    outlier_removal: bool,
    calibrate: bool,
    calibration_days: int,
    calibration_blend: float,
    calibration_min: float | None,
    calibration_max: float | None,
    include_partial_months: bool,
    min_train_weeks: int,
    verbose: bool,
    outlier_method: str,
    demand_proxy: str,
    demand_smoothing_weeks: int,
    model_families: list[str],
    eval_training_months: int | None,
    min_mroas: float,
    baseline_window: int,
    max_account_changes: int,
    wow_cap: float,
    apply_stability: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    months = _complete_months(df, date_col, include_partial_months)
    if not months:
        return pd.DataFrame(), pd.DataFrame(), []

    max_window = max(training_windows)
    first_allowed = (
        df[date_col].min().normalize() + pd.DateOffset(months=max_window)
        if max_window > 0
        else df[date_col].min().normalize()
    ).to_period("M")
    months = [month for month in months if month >= first_allowed]

    if months_back > 0:
        months = months[-months_back:]

    portfolio_rows = []
    account_rows = []
    month_summaries = []
    for model_family in model_families:
        for training_months in training_windows:
            for month in months:
                portfolio_row, account_detail, summary = _run_one_allocation_quality_backtest(
                    df=df,
                    date_col=date_col,
                    forecast_month=month,
                    training_months=training_months,
                    normalize_demand=normalize_demand,
                    outlier_removal=outlier_removal,
                    calibrate=calibrate,
                    calibration_days=calibration_days,
                    calibration_blend=calibration_blend,
                    calibration_min=calibration_min,
                    calibration_max=calibration_max,
                    min_train_weeks=min_train_weeks,
                    verbose=verbose,
                    outlier_method=outlier_method,
                    demand_proxy=demand_proxy,
                    demand_smoothing_weeks=demand_smoothing_weeks,
                    model_family=model_family,
                    eval_training_months=eval_training_months,
                    min_mroas=min_mroas,
                    baseline_window=baseline_window,
                    max_account_changes=max_account_changes,
                    wow_cap=wow_cap,
                    apply_stability=apply_stability,
                )
                if portfolio_row:
                    portfolio_rows.append(portfolio_row)
                account_rows.extend(account_detail)
                if summary:
                    month_summaries.append(summary)

    return pd.DataFrame(portfolio_rows), pd.DataFrame(account_rows), month_summaries


def _portfolio_by_month(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby(["model_family", "training_months", "month"])
        .agg(
            accounts=("account", "nunique"),
            actual_spend=("actual_spend", "sum"),
            predicted_revenue=("predicted_revenue", "sum"),
            actual_revenue=("actual_revenue", "sum"),
            abs_error=("abs_error", "sum"),
        )
        .reset_index()
    )
    grouped["error"] = grouped["actual_revenue"] - grouped["predicted_revenue"]
    grouped["error_pct_vs_pred"] = grouped["error"] / grouped["predicted_revenue"]
    grouped["wape"] = grouped["abs_error"] / grouped["actual_revenue"].replace(0, np.nan)
    grouped["predicted_roas"] = grouped["predicted_revenue"] / grouped["actual_spend"]
    grouped["actual_roas"] = grouped["actual_revenue"] / grouped["actual_spend"]
    return grouped


def _account_summary(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby(["model_family", "training_months", "account"])
        .agg(
            months=("month", "nunique"),
            actual_spend=("actual_spend", "sum"),
            predicted_revenue=("predicted_revenue", "sum"),
            actual_revenue=("actual_revenue", "sum"),
            abs_error=("abs_error", "sum"),
            avg_r2=("r2", "mean"),
            avg_calibration=("calibration_factor", "mean"),
            outliers_removed=("outliers_removed", "sum"),
        )
        .reset_index()
    )
    grouped["error"] = grouped["actual_revenue"] - grouped["predicted_revenue"]
    grouped["error_pct_vs_pred"] = grouped["error"] / grouped["predicted_revenue"]
    grouped["wape"] = grouped["abs_error"] / grouped["actual_revenue"].replace(0, np.nan)
    grouped["predicted_roas"] = grouped["predicted_revenue"] / grouped["actual_spend"]
    grouped["actual_roas"] = grouped["actual_revenue"] / grouped["actual_spend"]
    return grouped


def _settings_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        portfolio.groupby(["model_family", "training_months"])
        .agg(
            months=("month", "nunique"),
            actual_spend=("actual_spend", "sum"),
            predicted_revenue=("predicted_revenue", "sum"),
            actual_revenue=("actual_revenue", "sum"),
            abs_error=("abs_error", "sum"),
            avg_monthly_abs_error_pct=("error_pct_vs_pred", lambda s: s.abs().mean()),
        )
        .reset_index()
    )
    grouped["error"] = grouped["actual_revenue"] - grouped["predicted_revenue"]
    grouped["error_pct_vs_pred"] = grouped["error"] / grouped["predicted_revenue"]
    grouped["wape"] = grouped["abs_error"] / grouped["actual_revenue"].replace(0, np.nan)
    grouped["predicted_roas"] = grouped["predicted_revenue"] / grouped["actual_spend"]
    grouped["actual_roas"] = grouped["actual_revenue"] / grouped["actual_spend"]
    return grouped


def _allocation_direction_by_month(accounts: pd.DataFrame) -> pd.DataFrame:
    if accounts.empty:
        return pd.DataFrame()

    rows = []
    for (model_family, training_months, eval_training_months, month), grp in accounts.groupby(
        ALLOCATION_GROUP_KEYS + ["month"]
    ):
        increased = grp[grp["action_vs_b"] == "increase"]
        cut = grp[grp["action_vs_b"] == "cut"]

        def weighted_roas(frame):
            spend = frame["actual_spend"].sum()
            revenue = frame["actual_observed_revenue"].sum()
            return revenue / spend if spend else np.nan

        increased_roas = weighted_roas(increased)
        cut_roas = weighted_roas(cut)
        spread = increased_roas - cut_roas if pd.notna(increased_roas) and pd.notna(cut_roas) else np.nan
        rows.append(
            {
                "model_family": model_family,
                "training_months": training_months,
                "eval_training_months": eval_training_months,
                "month": month,
                "increase_accounts": increased["account"].nunique(),
                "cut_accounts": cut["account"].nunique(),
                "increase_actual_roas": increased_roas,
                "cut_actual_roas": cut_roas,
                "directional_spread": spread,
                "directional_hit": bool(spread > 0) if pd.notna(spread) else np.nan,
                "hit": "yes" if pd.notna(spread) and spread > 0 else ("no" if pd.notna(spread) else "n/a"),
            }
        )
    return pd.DataFrame(rows)


def _allocation_settings_summary(
    portfolio: pd.DataFrame,
    direction: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        portfolio.groupby(ALLOCATION_GROUP_KEYS)
        .agg(
            months=("month", "nunique"),
            budget=("budget", "sum"),
            actual_observed_revenue=("actual_observed_revenue", "sum"),
            actual_mix_pred_revenue=("actual_mix_pred_revenue", "sum"),
            scenario_b_pred_revenue=("scenario_b_pred_revenue", "sum"),
            scenario_c_pred_revenue=("scenario_c_pred_revenue", "sum"),
            scenario_c_budget=("scenario_c_budget", "sum"),
            unspent_budget=("unspent_budget", "sum"),
            c_wins_vs_actual=("c_vs_actual_revenue", lambda s: int((s > 0).sum())),
            c_wins_vs_b=("c_vs_b_revenue", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    grouped["c_vs_actual_revenue"] = (
        grouped["scenario_c_pred_revenue"] - grouped["actual_mix_pred_revenue"]
    )
    grouped["c_vs_actual_pct"] = (
        grouped["c_vs_actual_revenue"] / grouped["actual_mix_pred_revenue"].replace(0, np.nan)
    )
    grouped["c_vs_b_revenue"] = grouped["scenario_c_pred_revenue"] - grouped["scenario_b_pred_revenue"]
    grouped["c_vs_b_pct"] = (
        grouped["c_vs_b_revenue"] / grouped["scenario_b_pred_revenue"].replace(0, np.nan)
    )
    grouped["actual_observed_roas"] = grouped["actual_observed_revenue"] / grouped["budget"].replace(0, np.nan)
    grouped["actual_mix_pred_roas"] = grouped["actual_mix_pred_revenue"] / grouped["budget"].replace(0, np.nan)
    grouped["scenario_c_roas"] = grouped["scenario_c_pred_revenue"] / grouped["scenario_c_budget"].replace(0, np.nan)

    if not direction.empty:
        hit_rates = (
            direction.dropna(subset=["directional_hit"])
            .groupby(ALLOCATION_GROUP_KEYS)
            .agg(
                directional_months=("month", "nunique"),
                directional_hit_rate=("directional_hit", "mean"),
            )
            .reset_index()
        )
        grouped = grouped.merge(hit_rates, on=ALLOCATION_GROUP_KEYS, how="left")
    else:
        grouped["directional_months"] = np.nan
        grouped["directional_hit_rate"] = np.nan

    return grouped


def _allocation_account_summary(accounts: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        accounts.groupby(ALLOCATION_GROUP_KEYS + ["account"])
        .agg(
            months=("month", "nunique"),
            actual_spend=("actual_spend", "sum"),
            actual_observed_revenue=("actual_observed_revenue", "sum"),
            actual_pred_revenue=("actual_pred_revenue", "sum"),
            scenario_b_spend=("scenario_b_spend", "sum"),
            scenario_b_pred_revenue=("scenario_b_pred_revenue", "sum"),
            scenario_c_spend=("scenario_c_spend", "sum"),
            scenario_c_pred_revenue=("scenario_c_pred_revenue", "sum"),
            increases_vs_b=("action_vs_b", lambda s: int((s == "increase").sum())),
            cuts_vs_b=("action_vs_b", lambda s: int((s == "cut").sum())),
            avg_r2=("r2", "mean"),
            avg_calibration=("calibration_factor", "mean"),
            avg_c_inst_mroas=("scenario_c_inst_mroas", "mean"),
        )
        .reset_index()
    )
    grouped["c_vs_actual_spend"] = grouped["scenario_c_spend"] - grouped["actual_spend"]
    grouped["c_vs_actual_revenue"] = grouped["scenario_c_pred_revenue"] - grouped["actual_pred_revenue"]
    grouped["c_vs_b_spend"] = grouped["scenario_c_spend"] - grouped["scenario_b_spend"]
    grouped["actual_observed_roas"] = (
        grouped["actual_observed_revenue"] / grouped["actual_spend"].replace(0, np.nan)
    )
    grouped["actual_pred_roas"] = grouped["actual_pred_revenue"] / grouped["actual_spend"].replace(0, np.nan)
    grouped["scenario_c_roas"] = (
        grouped["scenario_c_pred_revenue"] / grouped["scenario_c_spend"].replace(0, np.nan)
    )
    return grouped


def display_results(results: pd.DataFrame, month_summaries: list[dict]) -> None:
    if results.empty:
        print("No backtest results. Check that the data contains enough complete months.")
        if month_summaries:
            print()
            print("Skipped months:")
            for summary in month_summaries:
                if summary.get("skipped"):
                    print(f"  {summary['month']} ({summary['training_months']}m): {summary['reason']}")
        return

    portfolio = _portfolio_by_month(results)
    settings = _settings_summary(portfolio)
    settings["_abs_bias"] = settings["error_pct_vs_pred"].abs()
    settings = settings.sort_values(["wape", "_abs_bias", "avg_monthly_abs_error_pct"])
    multiple_models = results["model_family"].nunique() > 1

    _print_table(
        "Model Family Comparison" if multiple_models else "Training Window Comparison",
        settings.to_dict("records"),
        [
            ("model_family", "Model", "str"),
            ("training_months", "Train", "int"),
            ("months", "Months", "int"),
            ("actual_spend", "Actual Spend", "eur"),
            ("predicted_revenue", "Pred Rev", "eur"),
            ("actual_revenue", "Actual Rev", "eur"),
            ("error_pct_vs_pred", "Bias", "pct"),
            ("wape", "WAPE", "pct_abs"),
            ("avg_monthly_abs_error_pct", "Avg |Mo Error|", "pct_abs"),
            ("predicted_roas", "Pred ROAS", "roas"),
            ("actual_roas", "Actual ROAS", "roas"),
        ],
    )

    if multiple_models:
        best = settings.iloc[0]
        detail_pairs = [(best["model_family"], int(best["training_months"]))]
        print()
        print(
            "Detailed monthly/account tables below show the best candidate by WAPE: "
            f"{best['model_family']} ({int(best['training_months'])}m training)."
        )
    else:
        detail_pairs = [
            (model_family, int(training_months))
            for model_family in sorted(results["model_family"].unique())
            for training_months in sorted(results["training_months"].unique())
        ]

    for model_family, training_months in detail_pairs:
        month_rows = (
            portfolio[
                (portfolio["model_family"] == model_family)
                & (portfolio["training_months"] == training_months)
            ]
            .sort_values("month")
            .to_dict("records")
        )
        _print_table(
            f"Portfolio By Month ({model_family}, {training_months}m training)",
            month_rows,
            [
                ("month", "Month", "str"),
                ("accounts", "Accts", "int"),
                ("actual_spend", "Spend", "eur"),
                ("predicted_revenue", "Pred Rev", "eur"),
                ("actual_revenue", "Actual Rev", "eur"),
                ("error_pct_vs_pred", "Error", "pct"),
                ("wape", "WAPE", "pct_abs"),
                ("predicted_roas", "Pred ROAS", "roas"),
                ("actual_roas", "Actual ROAS", "roas"),
            ],
        )

        account_rows = (
            _account_summary(
                results[
                    (results["model_family"] == model_family)
                    & (results["training_months"] == training_months)
                ]
            )
            .sort_values("wape", ascending=False)
            .to_dict("records")
        )
        _print_table(
            f"Account Reliability ({model_family}, {training_months}m training, worst WAPE first)",
            account_rows,
            [
                ("account", "Account", "str"),
                ("months", "Months", "int"),
                ("actual_spend", "Spend", "eur"),
                ("predicted_revenue", "Pred Rev", "eur"),
                ("actual_revenue", "Actual Rev", "eur"),
                ("error_pct_vs_pred", "Bias", "pct"),
                ("wape", "WAPE", "pct_abs"),
                ("predicted_roas", "Pred ROAS", "roas"),
                ("actual_roas", "Actual ROAS", "roas"),
                ("avg_r2", "Avg R2", "float"),
                ("avg_calibration", "Avg Cal", "float"),
            ],
        )

    skipped = [summary for summary in month_summaries if summary.get("skipped")]
    if skipped:
        print()
        print("Skipped months")
        print("--------------")
        for summary in skipped:
            print(f"{summary['month']} ({summary['training_months']}m): {summary['reason']}")


def display_allocation_quality_results(
    portfolio: pd.DataFrame,
    accounts: pd.DataFrame,
    month_summaries: list[dict],
) -> None:
    if portfolio.empty:
        print("No allocation-quality results. Check that the data contains enough complete months.")
        skipped = [summary for summary in month_summaries if summary.get("skipped")]
        if skipped:
            print()
            print("Skipped months:")
            for summary in skipped:
                print(f"  {summary['month']} ({summary['training_months']}m): {summary['reason']}")
        return

    direction = _allocation_direction_by_month(accounts)
    settings = _allocation_settings_summary(portfolio, direction)
    settings = settings.sort_values(["c_vs_actual_pct", "c_vs_b_pct"], ascending=[False, False])
    multiple_settings = (
        portfolio["model_family"].nunique() > 1
        or portfolio["training_months"].nunique() > 1
        or portfolio["eval_training_months"].nunique() > 1
    )

    _print_table(
        "Allocation Quality Summary",
        settings.to_dict("records"),
        [
            ("model_family", "Model", "str"),
            ("training_months", "Alloc Train", "int"),
            ("eval_training_months", "Eval Train", "int"),
            ("months", "Months", "int"),
            ("budget", "Budget", "eur"),
            ("actual_mix_pred_revenue", "Actual Mix Rev", "eur"),
            ("scenario_c_pred_revenue", "C Rev", "eur"),
            ("c_vs_actual_pct", "C vs Actual", "pct"),
            ("c_vs_b_pct", "C vs B", "pct"),
            ("c_wins_vs_actual", "C Wins Actual", "int"),
            ("c_wins_vs_b", "C Wins B", "int"),
            ("directional_hit_rate", "Dir Hit", "pct_abs"),
        ],
    )

    if multiple_settings:
        best = settings.iloc[0]
        detail_pairs = [
            (
                best["model_family"],
                int(best["training_months"]),
                int(best["eval_training_months"]),
            )
        ]
        print()
        print(
            "Detailed tables below show the strongest candidate by C vs Actual: "
            f"{best['model_family']} "
            f"({int(best['training_months'])}m allocation / "
            f"{int(best['eval_training_months'])}m evaluation)."
        )
    else:
        detail_pairs = [
            (model_family, int(training_months), int(eval_training_months))
            for model_family in sorted(portfolio["model_family"].unique())
            for training_months in sorted(portfolio["training_months"].unique())
            for eval_training_months in sorted(portfolio["eval_training_months"].unique())
        ]

    for model_family, training_months, eval_training_months in detail_pairs:
        title_suffix = (
            f"{model_family}, {training_months}m allocation / "
            f"{eval_training_months}m evaluation"
        )
        month_rows = (
            portfolio[
                (portfolio["model_family"] == model_family)
                & (portfolio["training_months"] == training_months)
                & (portfolio["eval_training_months"] == eval_training_months)
            ]
            .sort_values("month")
            .to_dict("records")
        )
        _print_table(
            f"Portfolio Allocation Quality ({title_suffix})",
            month_rows,
            [
                ("month", "Month", "str"),
                ("budget", "Budget", "eur"),
                ("actual_observed_revenue", "Observed Rev", "eur"),
                ("actual_mix_pred_revenue", "Actual Mix Rev", "eur"),
                ("scenario_b_pred_revenue", "B Rev", "eur"),
                ("scenario_c_id", "C ID", "str"),
                ("scenario_c_budget", "C Budget", "eur"),
                ("scenario_c_pred_revenue", "C Rev", "eur"),
                ("unspent_budget", "Unspent", "eur"),
                ("c_vs_actual_pct", "C vs Actual", "pct"),
                ("c_vs_b_pct", "C vs B", "pct"),
            ],
        )

        account_rows = (
            _allocation_account_summary(
                accounts[
                    (accounts["model_family"] == model_family)
                    & (accounts["training_months"] == training_months)
                    & (accounts["eval_training_months"] == eval_training_months)
                ]
            )
            .assign(_abs_move=lambda frame: frame["c_vs_actual_spend"].abs())
            .sort_values("_abs_move", ascending=False)
            .to_dict("records")
        )
        _print_table(
            f"Account Allocation Movement ({title_suffix})",
            account_rows,
            [
                ("account", "Account", "str"),
                ("months", "Months", "int"),
                ("actual_spend", "Actual Spend", "eur"),
                ("scenario_c_spend", "C Spend", "eur"),
                ("c_vs_actual_spend", "C-Actual Spend", "eur"),
                ("actual_pred_revenue", "Actual Mix Rev", "eur"),
                ("scenario_c_pred_revenue", "C Rev", "eur"),
                ("c_vs_actual_revenue", "C-Actual Rev", "eur"),
                ("increases_vs_b", "Inc vs B", "int"),
                ("cuts_vs_b", "Cut vs B", "int"),
                ("actual_observed_roas", "Obs ROAS", "roas"),
                ("scenario_c_roas", "C ROAS", "roas"),
                ("avg_c_inst_mroas", "Avg C mROAS", "roas"),
                ("avg_r2", "Avg R2", "float"),
            ],
        )

        direction_rows = (
            direction[
                (direction["model_family"] == model_family)
                & (direction["training_months"] == training_months)
                & (direction["eval_training_months"] == eval_training_months)
            ]
            .sort_values("month")
            .to_dict("records")
        )
        _print_table(
            f"Directional Sanity Check ({title_suffix})",
            direction_rows,
            [
                ("month", "Month", "str"),
                ("increase_accounts", "Inc Accts", "int"),
                ("cut_accounts", "Cut Accts", "int"),
                ("increase_actual_roas", "Inc Obs ROAS", "roas"),
                ("cut_actual_roas", "Cut Obs ROAS", "roas"),
                ("directional_spread", "Spread", "roas"),
                ("hit", "Hit", "str"),
            ],
        )

    skipped = [summary for summary in month_summaries if summary.get("skipped")]
    if skipped:
        print()
        print("Skipped months")
        print("--------------")
        for summary in skipped:
            print(f"{summary['month']} ({summary['training_months']}m): {summary['reason']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest budget-solver forecast accuracy by training on historical "
            "data before each month and predicting revenue at actual spend."
        )
    )
    parser.add_argument(
        "--data",
        default=str(DATA_PATH),
        help=f"Daily data CSV/XLSX path (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--training-months",
        type=_parse_training_windows,
        default=[6, 12],
        help="Comma-separated training windows to compare, e.g. 3,6,12 (default: 6,12)",
    )
    parser.add_argument(
        "--model-family",
        type=_parse_model_families,
        default=["current_log"],
        help=(
            "Comma-separated model families to compare, or 'all'. "
            f"Choices: {', '.join(MODEL_FAMILIES)} (default: current_log)."
        ),
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Number of most recent complete months to backtest (0 = all eligible, default: 6)",
    )
    parser.add_argument(
        "--allocation-quality",
        action="store_true",
        help=(
            "Run counterfactual allocation-quality backtest instead of forecast-error "
            "backtest. Compares actual monthly spend mix vs Scenario B/C at the same budget."
        ),
    )
    parser.add_argument(
        "--allocation-eval-training-months",
        type=int,
        default=None,
        help=(
            "In --allocation-quality mode, score actual/B/C revenue with a different "
            "training window than the one used to choose Scenario C. Example: "
            "--training-months 9 --allocation-eval-training-months 6 tests a 9/6 hybrid."
        ),
    )
    parser.add_argument(
        "--target",
        choices=["conversion_value", "conversions"],
        default="conversion_value",
        help="Metric to forecast (default: conversion_value)",
    )
    parser.add_argument(
        "--min-train-weeks",
        type=int,
        default=8,
        help="Minimum weekly observations required before a month is tested (default: 8)",
    )
    parser.add_argument(
        "--min-mroas",
        type=_positive_float,
        default=2.5,
        help="Minimum instantaneous mROAS floor for Scenario C/D allocation tests (default: 2.5).",
    )
    parser.add_argument(
        "--baseline-window",
        type=_positive_int,
        default=7,
        help="Days to use for Scenario A baseline in allocation tests (default: 7).",
    )
    parser.add_argument(
        "--max-account-changes",
        type=int,
        default=0,
        help="Limit optional Scenario C reallocations in allocation tests (0 = no limit, default: 0).",
    )
    parser.add_argument(
        "--wow-cap",
        type=_positive_float,
        default=0.20,
        help="Week-over-week phasing warning cap used by Scenario C (default: 0.20).",
    )
    parser.add_argument(
        "--no-stability-rules",
        dest="apply_stability",
        action="store_false",
        help="Disable Scenario C stability rules in allocation tests.",
    )
    parser.set_defaults(apply_stability=True)
    parser.add_argument(
        "--no-outlier-removal",
        action="store_true",
        help="Disable the same outlier removal used by the solver.",
    )
    parser.add_argument(
        "--outlier-method",
        choices=[
            "current",
            "none",
            "low-spend",
            "roas-iqr",
            "winsorize-roas",
            "low-spend-winsorize",
        ],
        default="current",
        help=(
            "Outlier strategy to test (default: current). "
            "--no-outlier-removal is equivalent to --outlier-method none."
        ),
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable trailing-30-day calibration before each forecast month.",
    )
    parser.add_argument(
        "--calibration-days",
        type=_positive_int,
        default=TRAILING_WINDOW_DAYS,
        help=(
            "Trailing days before each forecast month used for calibration "
            f"(default: {TRAILING_WINDOW_DAYS})."
        ),
    )
    parser.add_argument(
        "--calibration-blend",
        type=_calibration_blend,
        default=1.0,
        help=(
            "Strength of calibration from 0 to 1. 1 = full calibration, "
            "0 = neutral/no effect (default: 1)."
        ),
    )
    parser.add_argument(
        "--calibration-min",
        type=_positive_float,
        default=None,
        help="Optional lower cap for each account calibration factor, e.g. 0.7.",
    )
    parser.add_argument(
        "--calibration-max",
        type=_positive_float,
        default=None,
        help="Optional upper cap for each account calibration factor, e.g. 1.3.",
    )
    parser.add_argument(
        "--normalize-demand",
        action="store_true",
        help="Apply demand normalization during each historical fit.",
    )
    parser.add_argument(
        "--demand-proxy",
        choices=["roas_legacy", "roas", "cvr", "revenue_per_click", "clicks_per_eur"],
        default="roas_legacy",
        help=(
            "Demand proxy used when --normalize-demand is on. "
            "roas_legacy matches the current solver method (default: roas_legacy)."
        ),
    )
    parser.add_argument(
        "--demand-smoothing-weeks",
        type=_positive_int,
        default=1,
        help=(
            "Odd-numbered smoothing window for non-legacy demand proxies "
            "(default: 1 = no smoothing)."
        ),
    )
    parser.add_argument(
        "--include-partial-months",
        action="store_true",
        help="Include the latest month even if the data does not reach month-end.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show curve fitting messages for each backtest run.",
    )

    args = parser.parse_args(argv)
    if (
        args.calibration_min is not None
        and args.calibration_max is not None
        and args.calibration_min > args.calibration_max
    ):
        parser.error("--calibration-min cannot be greater than --calibration-max")
    if args.demand_smoothing_weeks % 2 == 0:
        parser.error("--demand-smoothing-weeks must be an odd number")
    if args.max_account_changes < 0:
        parser.error("--max-account-changes must be >= 0")
    if args.allocation_eval_training_months is not None and args.allocation_eval_training_months < 0:
        parser.error("--allocation-eval-training-months must be >= 0")
    if args.allocation_eval_training_months is not None and not args.allocation_quality:
        parser.error("--allocation-eval-training-months can only be used with --allocation-quality")
    if args.allocation_eval_training_months is not None and len(args.training_months) != 1:
        parser.error("--allocation-eval-training-months requires a single --training-months value")
    if args.allocation_quality and any(family != "current_log" for family in args.model_family):
        parser.error(
            "--allocation-quality currently supports --model-family current_log only "
            "because Scenario C breakeven logic is log-curve based."
        )

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        print("Pass --data with the daily solver input CSV/XLSX.", file=sys.stderr)
        return 1

    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    df = _prepare_target(df, args.target)

    date_col = next((c for c in ("date", "week_start", "week") if c in df.columns), None)
    if not date_col:
        print("ERROR: data must include a date, week_start, or week column.", file=sys.stderr)
        return 1

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        print("ERROR: no valid dates found in data.", file=sys.stderr)
        return 1

    print()
    print("Backtest configuration")
    print("----------------------")
    print(f"Date range          : {df[date_col].min().date()} to {df[date_col].max().date()}")
    print(f"Accounts            : {df['account_name'].nunique()}")
    print(f"Target              : {args.target}")
    print(f"Mode                : {'allocation quality' if args.allocation_quality else 'forecast accuracy'}")
    print(f"Training windows    : {', '.join(str(v) for v in args.training_months)} months")
    print(f"Model families      : {', '.join(args.model_family)}")
    print(f"Forecast months     : {'all eligible' if args.months == 0 else args.months}")
    active_outlier_method = "none" if args.no_outlier_removal else args.outlier_method
    print(f"Outlier method      : {active_outlier_method}")
    print(f"Calibration         : {'off' if args.no_calibrate else 'on'}")
    if not args.no_calibrate:
        print(f"Calibration days    : {args.calibration_days}")
        print(f"Calibration blend   : {args.calibration_blend:.2f}")
        print(
            "Calibration caps    : "
            f"{args.calibration_min if args.calibration_min is not None else 'none'}"
            " to "
            f"{args.calibration_max if args.calibration_max is not None else 'none'}"
        )
    print(f"Demand normalization: {'on' if args.normalize_demand else 'off'}")
    if args.normalize_demand:
        print(f"Demand proxy        : {args.demand_proxy}")
        print(f"Demand smoothing    : {args.demand_smoothing_weeks} week(s)")
    if args.allocation_quality:
        print(f"mROAS floor         : {args.min_mroas:.2f}x")
        print(f"Baseline window     : {args.baseline_window} days")
        if args.allocation_eval_training_months is not None:
            print(
                "Allocation scoring  : "
                f"{args.training_months[0]}m allocation model, "
                f"{args.allocation_eval_training_months}m evaluation model"
            )
        print(
            "Stability rules     : "
            f"{'on' if args.apply_stability else 'off'}"
            f" (max changes {args.max_account_changes})"
        )
    print()
    if args.allocation_quality:
        print(
            "Metric note: this is a model-predicted counterfactual. "
            "Actual Mix Rev, B Rev, and C Rev are all evaluated with the fitted model "
            "at the same monthly budget."
        )
    else:
        print(
            "Metric note: Error/Bias is (actual - predicted) / predicted. "
            "WAPE is sum(abs error) / sum(actual revenue)."
        )

    if args.allocation_quality:
        portfolio, accounts, month_summaries = run_allocation_quality_backtests(
            df=df,
            date_col=date_col,
            training_windows=args.training_months,
            months_back=args.months,
            normalize_demand=args.normalize_demand,
            outlier_removal=not args.no_outlier_removal,
            calibrate=not args.no_calibrate,
            calibration_days=args.calibration_days,
            calibration_blend=args.calibration_blend,
            calibration_min=args.calibration_min,
            calibration_max=args.calibration_max,
            include_partial_months=args.include_partial_months,
            min_train_weeks=args.min_train_weeks,
            verbose=args.verbose,
            outlier_method=args.outlier_method,
            demand_proxy=args.demand_proxy,
            demand_smoothing_weeks=args.demand_smoothing_weeks,
            model_families=args.model_family,
            eval_training_months=args.allocation_eval_training_months,
            min_mroas=args.min_mroas,
            baseline_window=args.baseline_window,
            max_account_changes=args.max_account_changes,
            wow_cap=args.wow_cap,
            apply_stability=args.apply_stability,
        )
        display_allocation_quality_results(portfolio, accounts, month_summaries)
        return 0

    results, month_summaries = run_backtests(
        df=df,
        date_col=date_col,
        training_windows=args.training_months,
        months_back=args.months,
        normalize_demand=args.normalize_demand,
        outlier_removal=not args.no_outlier_removal,
        calibrate=not args.no_calibrate,
        calibration_days=args.calibration_days,
        calibration_blend=args.calibration_blend,
        calibration_min=args.calibration_min,
        calibration_max=args.calibration_max,
        include_partial_months=args.include_partial_months,
        min_train_weeks=args.min_train_weeks,
        verbose=args.verbose,
        outlier_method=args.outlier_method,
        demand_proxy=args.demand_proxy,
        demand_smoothing_weeks=args.demand_smoothing_weeks,
        model_families=args.model_family,
    )
    display_results(results, month_summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
