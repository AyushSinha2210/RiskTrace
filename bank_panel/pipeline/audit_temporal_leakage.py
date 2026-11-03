"""Temporal leakage auditor. Exits with code 1 if any check fails.

Usage: python audit_temporal_leakage.py --panel data/panel/panel_final.csv \
           --prices data/market/market_daily_prices.csv --edges data/market/market_correlation_edges.csv \
           --events data/reference/rbi_distress_events.csv --registry data/reference/indian_banks_registry.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ingest_and_clean_panel import FEATURES, LAG_DAYS_AUDITED, LAG_DAYS_UNAUDITED
from ratios import compute_ratios


def check_information_lag(panel: pd.DataFrame) -> list[str]:
    q4 = panel["quarter_end_date"].dt.month == 3
    minimum = pd.to_timedelta(np.where(q4, LAG_DAYS_AUDITED, LAG_DAYS_UNAUDITED), unit="D")
    bad = panel[panel["information_availability_date"] < panel["quarter_end_date"] + minimum]
    return [f"{len(bad)} rows have information_availability_date earlier than the disclosure lag"] if len(bad) else []


def check_exits(panel: pd.DataFrame, registry: pd.DataFrame) -> list[str]:
    merged = panel.merge(registry[["bank_id", "active_end_quarter"]], on="bank_id")
    bad = merged[merged["calendar_quarter"] > merged["active_end_quarter"]]
    return [f"{len(bad)} rows fall after a bank's exit, for example {bad.iloc[0]['bank_id']} {bad.iloc[0]['calendar_quarter']}"] if len(bad) else []


def check_imputation(panel: pd.DataFrame) -> list[str]:
    """Every imputed value must equal the bank's last observed value from at most two quarters earlier."""
    problems = []
    ordered = panel.sort_values(["bank_id", "quarter_end_date"])
    for feature in FEATURES:
        flag = f"imputed_{feature}"
        if flag not in ordered or not ordered[flag].any():
            continue
        observed = ordered[feature].where(ordered[flag] == 0)
        allowed = observed.groupby(ordered["bank_id"]).ffill(limit=2)
        wrong = (ordered[flag] == 1) & ~np.isclose(ordered[feature], allowed, atol=1e-3, equal_nan=False)
        if wrong.any():
            problems.append(f"{feature}: {int(wrong.sum())} imputed values do not match an earlier observation")
    return problems


def check_labels(panel: pd.DataFrame, events: pd.DataFrame) -> list[str]:
    """A positive onset label must have its event strictly after the row's quarter end and within the horizon."""
    problems = []
    onsets = events[events["is_onset"] == 1]
    for k in (1, 2, 4):
        column = f"y_event_onset_{k}"
        for bank, quarter_end in panel.loc[panel[column] == 1, ["bank_id", "quarter_end_date"]].itertuples(index=False):
            dates = onsets.loc[onsets["bank_id"] == bank, "event_date_used"]
            horizon = quarter_end + pd.offsets.QuarterEnd(k)
            if not ((dates > quarter_end) & (dates <= horizon)).any():
                problems.append(f"{column}=1 for {bank} {quarter_end.date()} has no event in its window")
    active_positive = panel[(panel["under_pca_or_moratorium_at_t"] == 1) & (panel["y_event_onset_4"] == 1)]
    if len(active_positive):
        problems.append(f"{len(active_positive)} rows are labelled onset while already under PCA")
    usable_blind = panel[(panel["mask_onset_usable"] == 1) & (panel["onset_in_blind_window"] == 1)]
    if len(usable_blind):
        problems.append(f"{len(usable_blind)} usable rows have an onset in the blind window")
    return problems


def check_ratios_recompute(panel: pd.DataFrame) -> list[str]:
    """Ratios must be pure functions of raw columns, so explanation perturbations stay consistent."""
    recomputed = compute_ratios(panel)
    problems = []
    for feature in ("gnpa_ratio", "cd_ratio", "casa_ratio", "crar_ratio"):
        original = panel[feature].where(panel[f"imputed_{feature}"] == 0)
        gap = (original - recomputed[feature]).abs().max()
        if gap > 1e-3:
            problems.append(f"{feature} differs from a raw recomputation by {gap:.4f}")
    return problems


def check_correlation_window(prices: pd.DataFrame, edges: pd.DataFrame, samples: int = 25) -> list[str]:
    """Recompute sampled edges from returns strictly before quarter end, then show future data would differ."""
    wide = prices.pivot(index="date", columns="bank_id", values="log_return").sort_index()
    problems = []
    for row in edges.sample(min(samples, len(edges)), random_state=0).itertuples(index=False):
        end = pd.Timestamp(row.quarter_end_date)
        window = wide[(wide.index < end) & (wide.index >= end - pd.Timedelta(days=90))]
        rho = window[row.bank_i].corr(window[row.bank_j], min_periods=30)
        if not np.isclose(rho, row.rho, atol=1e-3):
            problems.append(f"edge {row.bank_i}-{row.bank_j} {end.date()} is not reproducible from pre-quarter-end returns")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=Path("data/panel/panel_final.csv"))
    parser.add_argument("--prices", type=Path, default=Path("data/market/market_daily_prices.csv"))
    parser.add_argument("--edges", type=Path, default=Path("data/market/market_correlation_edges.csv"))
    parser.add_argument("--events", type=Path, default=Path("data/reference/rbi_distress_events.csv"))
    parser.add_argument("--registry", type=Path, default=Path("data/reference/indian_banks_registry.csv"))
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, parse_dates=["quarter_end_date", "information_availability_date"])
    prices = pd.read_csv(args.prices, parse_dates=["date"], low_memory=False)
    edges = pd.read_csv(args.edges)
    events = pd.read_csv(args.events, parse_dates=["event_date_used"])
    registry = pd.read_csv(args.registry, keep_default_na=False)
    checks = {
        "information lag": check_information_lag(panel),
        "no rows after exit": check_exits(panel, registry),
        "imputation looks backward only": check_imputation(panel),
        "labels sit in the future": check_labels(panel, events),
        "ratios recompute from raw": check_ratios_recompute(panel),
        "correlation windows": check_correlation_window(prices, edges),
    }
    failed = False
    for name, problems in checks.items():
        print(f"{'FAIL' if problems else 'PASS'}  {name}")
        for problem in problems[:10]:
            print(f"      {problem}")
        failed = failed or bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
