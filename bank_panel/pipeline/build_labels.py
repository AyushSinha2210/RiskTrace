"""Deliverable 3: attach supervisory and threshold labels to the cleaned panel.

Usage: python build_labels.py --panel data/panel/panel_clean.csv --events data/reference/rbi_distress_events.csv \
           --output data/panel/panel_final.csv

Labels
  y_event_onset_k (k = 1, 2, 4)  1 if a PCA entry or moratorium onset falls after quarter t and by the end of t+k.
  mask_onset_usable              1 only if the bank is at risk at t and no onset falls in the blind window (see below).
  y_active_regulatory_distress   1 if the bank was under PCA or moratorium at any time in the quarter.
  y_threshold_breach             the brief's static rule.
  y_threshold_breach_regime      the rule in force at that date.
  y_new_breach_k                 a breach within the next k quarters by a bank not in breach at t.

The blind window: quarter t's numbers are only public on information_availability_date, so an event between
quarter end and that date cannot be predicted from them and is often caused by them (RBI acts on March accounts).
Rows with an onset in that window keep their y value but get mask_onset_usable = 0. Train on the mask.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 2, 4)
LAST_QUARTER = pd.Timestamp("2026-03-31")


def regime_minimums(date: pd.Timestamp) -> dict:
    """Capital floors by date. The buffer stayed at 1.875% until 2021-10-01 after RBI deferred the last tranche."""
    if date < pd.Timestamp("2017-03-31"):
        return {"regime": "2002", "crar": 9.0, "cet1": None, "nnpa": 10.0, "roa_floor": 0.25}
    if date < pd.Timestamp("2018-03-31"):
        return {"regime": "2017", "crar": 10.25, "cet1": 6.75, "nnpa": 6.0, "roa_floor": None}
    if date < pd.Timestamp("2021-10-01"):
        return {"regime": "2017", "crar": 10.875, "cet1": 7.375, "nnpa": 6.0, "roa_floor": None}
    if date < pd.Timestamp("2022-01-01"):
        return {"regime": "2017", "crar": 11.5, "cet1": 8.0, "nnpa": 6.0, "roa_floor": None}
    return {"regime": "2021", "crar": 11.5, "cet1": 8.0, "nnpa": 6.0, "roa_floor": None}


def add_roa_streak(df: pd.DataFrame) -> pd.DataFrame:
    """roa_negative_2y: the last two completed financial years both had negative ROA.

    RBI's March ROA is April-to-March, so it is a full-year figure. Each value is carried forward
    to later quarters but never backward.
    """
    out = df.sort_values(["bank_id", "quarter_end_date"]).copy()
    march = out["quarter_end_date"].dt.month == 3
    annual = out["rbi_roa_annualized"].where(march)
    last = annual.groupby(out["bank_id"]).ffill()
    previous = annual.groupby(out["bank_id"]).transform(lambda s: s.dropna().shift(1).reindex(s.index).ffill())
    out["roa_negative_2y"] = ((last < 0) & (previous < 0)).astype(int)
    return out


def threshold_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = add_roa_streak(df)
    static = (out["nnpa_ratio"] >= 6.0) | (out["crar_ratio"] < 10.25) | (out["cet1_ratio"] < 6.75) | (out["roa_negative_2y"] == 1)
    out["y_threshold_breach"] = static.astype(int)
    regime_rows = pd.DataFrame([regime_minimums(d) for d in out["quarter_end_date"]], index=out.index)
    breach = (out["nnpa_ratio"] >= regime_rows["nnpa"]) | (out["crar_ratio"] < regime_rows["crar"])
    breach |= (out["cet1_ratio"] < pd.to_numeric(regime_rows["cet1"])).fillna(False)
    old = regime_rows["regime"] == "2002"
    breach |= old & (out["rbi_roa_annualized"] < 0.25)
    breach |= (regime_rows["regime"] == "2017") & (out["roa_negative_2y"] == 1)
    out["pca_regime"] = regime_rows["regime"]
    out["y_threshold_breach_regime"] = breach.astype(int)
    for k in HORIZONS:
        future = pd.concat([out.groupby("bank_id")["y_threshold_breach_regime"].shift(-i) for i in range(1, k + 1)], axis=1)
        complete = future.notna().all(axis=1)
        out[f"y_new_breach_{k}"] = (future.max(axis=1).where(complete))
        out.loc[out["y_threshold_breach_regime"] == 1, f"y_new_breach_{k}"] = 0
        out.loc[~complete & (out["y_threshold_breach_regime"] == 0), f"y_new_breach_{k}"] = np.nan
    return out


def supervisory_labels(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    intervals = events[events["event_type"].isin(["PCA_Entry", "Moratorium_S45"])].copy()
    intervals["end"] = pd.to_datetime(intervals["exit_date"]).fillna(pd.Timestamp("2100-01-01"))
    onsets = events[events["is_onset"] == 1]
    last_row = out.groupby("bank_id")["quarter_end_date"].transform("max")
    active, under, blind = [], [], []
    labels = {k: [] for k in HORIZONS}
    for bank, quarter_end, available, final in zip(out["bank_id"], out["quarter_end_date"], out["information_availability_date"], last_row):
        quarter_start = quarter_end - pd.offsets.QuarterEnd(1) + pd.Timedelta(days=1)
        own = intervals[intervals["bank_id"] == bank]
        active.append(int(((own["event_date_used"] <= quarter_end) & (own["end"] >= quarter_start)).any()))
        under.append(int(((own["event_date_used"] <= quarter_end) & (own["end"] > quarter_end)).any()))
        dates = onsets.loc[onsets["bank_id"] == bank, "event_date_used"]
        blind.append(int(((dates > quarter_end) & (dates <= available)).any()))
        for k in HORIZONS:
            horizon_end = quarter_end + pd.offsets.QuarterEnd(k)
            hit = ((dates > quarter_end) & (dates <= horizon_end)).any()
            labels[k].append(1.0 if hit else (np.nan if horizon_end > min(final, LAST_QUARTER) else 0.0))
    out["y_active_regulatory_distress"] = active
    out["under_pca_or_moratorium_at_t"] = under
    out["onset_in_blind_window"] = blind
    for k in HORIZONS:
        out[f"y_event_onset_{k}"] = labels[k]
        out.loc[out["under_pca_or_moratorium_at_t"] == 1, f"y_event_onset_{k}"] = 0.0
    out["mask_onset_usable"] = ((out["under_pca_or_moratorium_at_t"] == 0) & (out["onset_in_blind_window"] == 0)).astype(int)
    return out


def threshold_level(value: float, cuts: list[float], higher_is_worse: bool) -> int:
    """0 if clear of threshold 1, otherwise the highest threshold breached (1 to 3)."""
    if pd.isna(value):
        return 0
    return sum(value >= c if higher_is_worse else value < c for c in cuts)


def annotate_events(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Derive each PCA entry's risk threshold from the last March accounts before it.

    RBI does not publish the threshold a bank was placed under, so these columns are calculated, not reported.
    Pre-2017 entries are left blank because that scheme used different tests.
    """
    out = events.copy()
    for column in ("derived_basis_quarter", "derived_nnpa_pct", "derived_crar_pct", "derived_cet1_pct", "derived_negative_roa_years", "derived_threshold"):
        out[column] = pd.Series([None] * len(out), index=out.index, dtype=object)
    march = panel[panel["quarter_end_date"].dt.month == 3]
    for index, event in out.iterrows():
        if event["event_type"] != "PCA_Entry" or not str(event["framework"]).startswith("PCA 2017"):
            continue
        rows = march[(march["bank_id"] == event["bank_id"]) & (march["quarter_end_date"] < event["event_date_used"])].sort_values("quarter_end_date")
        if rows.empty:
            continue
        row = rows.iloc[-1]
        floors = regime_minimums(row["quarter_end_date"])
        nnpa = threshold_level(row["nnpa_ratio"], [6.0, 9.0, 12.0], True)
        crar = threshold_level(row["crar_ratio"], [floors["crar"], floors["crar"] - 2.5, floors["crar"] - 4.0], False)
        cet1 = threshold_level(row["cet1_ratio"], [floors["cet1"], floors["cet1"] - 1.625, floors["cet1"] - 3.125], False)
        streak = 0
        for value in rows["rbi_roa_annualized"].iloc[::-1]:
            if value < 0:
                streak += 1
            else:
                break
        roa = 3 if streak >= 4 else 2 if streak == 3 else 1 if streak == 2 else 0
        out.loc[index, "derived_basis_quarter"] = row["calendar_quarter"]
        out.loc[index, "derived_nnpa_pct"] = round(row["nnpa_ratio"], 2)
        out.loc[index, "derived_crar_pct"] = round(row["crar_ratio"], 2)
        out.loc[index, "derived_cet1_pct"] = round(row["cet1_ratio"], 2)
        out.loc[index, "derived_negative_roa_years"] = streak
        out.loc[index, "derived_threshold"] = max(nnpa, crar, cet1, roa)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=Path("data/panel/panel_clean.csv"))
    parser.add_argument("--events", type=Path, default=Path("data/reference/rbi_distress_events.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/panel/panel_final.csv"))
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, parse_dates=["quarter_end_date", "information_availability_date"])
    events = pd.read_csv(args.events, parse_dates=["event_date_used"])
    final = supervisory_labels(threshold_labels(panel), events).sort_values(["bank_id", "quarter_end_date"])
    final.to_csv(args.output, index=False, float_format="%.4f", date_format="%Y-%m-%d")
    annotate_events(events, final).to_csv(args.events, index=False, date_format="%Y-%m-%d")
    print(f"{len(final)} rows written to {args.output}")
    usable = final[final["mask_onset_usable"] == 1]
    print("active distress rows:", int(final["y_active_regulatory_distress"].sum()))
    for k in HORIZONS:
        print(f"k={k}: usable rows {usable[f'y_event_onset_{k}'].notna().sum()}, positives {int(usable[f'y_event_onset_{k}'].sum())}, "
              f"banks {usable.loc[usable[f'y_event_onset_{k}'] == 1, 'bank_id'].nunique()}")
    print("static breach rows:", int(final["y_threshold_breach"].sum()), "| regime breach rows:", int(final["y_threshold_breach_regime"].sum()))


if __name__ == "__main__":
    main()
