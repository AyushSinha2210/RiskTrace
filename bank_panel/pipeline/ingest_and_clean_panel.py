"""Clean a raw bank-quarter panel: fiscal dates, information lag, identity checks, safe imputation.

Input: any CSV with the raw_* schema. data/panel/panel_raw.csv from build_rbi_panel.py is ready to use.
To load a ProwessIQ or NSE export, pass --column-map, a JSON file of {"their column": "raw_total_assets", ...}.
A file may give dates as quarter_end_date (YYYY-MM-DD) or as fiscal_year + fiscal_quarter
(Indian fiscal year, so FY2018 Q3 is the quarter ended 2017-12-31).

Usage:
  python ingest_and_clean_panel.py --input data/panel/panel_raw.csv --output data/panel/panel_clean.csv \
      --coverage data/validation/coverage_report.csv --train-end 2021Q4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ratios import compute_ratios

Q4_MONTH = 3
LAG_DAYS_UNAUDITED = 45  # Q1 to Q3, SEBI Reg 33
LAG_DAYS_AUDITED = 60    # Q4 annual audited results and Pillar 3
# SEBI moved these deadlines during COVID. A fixed +45/+60 would treat results as public before they had to be.
SEBI_DEADLINE_OVERRIDES = {
    "2020-03-31": "2020-07-31",  # SEBI circular of 2020-06-24
    "2020-06-30": "2020-09-15",  # SEBI circular of 2020-07-29
    "2021-03-31": "2021-06-30",  # SEBI circular of 2021-04-29
}
FEATURES = ["gnpa_ratio", "nnpa_ratio", "pcr_ratio", "crar_ratio", "cet1_ratio", "casa_ratio", "cd_ratio",
            "slr_liquidity_ratio", "roa_annualized", "nim_annualized"]
NET_NPA_RATIOS = ["nnpa_ratio", "pcr_ratio"]


def fiscal_to_quarter_end(fiscal_year: int, fiscal_quarter: int) -> pd.Timestamp:
    """Indian fiscal quarters: Q1=Jun, Q2=Sep, Q3=Dec, Q4=Mar. FY2018 runs Apr 2017 to Mar 2018."""
    month = {1: 6, 2: 9, 3: 12, 4: 3}[int(fiscal_quarter)]
    year = int(fiscal_year) if month == 3 else int(fiscal_year) - 1
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def standardise_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "quarter_end_date" in out:
        out["quarter_end_date"] = pd.to_datetime(out["quarter_end_date"]) + pd.offsets.MonthEnd(0)
    else:
        out["quarter_end_date"] = [fiscal_to_quarter_end(y, q) for y, q in zip(out["fiscal_year"], out["fiscal_quarter"])]
    dates = out["quarter_end_date"]
    out["calendar_quarter"] = dates.dt.to_period("Q").astype(str)
    out["fiscal_year"] = np.where(dates.dt.month <= Q4_MONTH, dates.dt.year, dates.dt.year + 1)
    out["fiscal_quarter"] = dates.dt.month.map({6: 1, 9: 2, 12: 3, 3: 4})
    return out


def add_information_dates(df: pd.DataFrame, filing_dates: Path | None = None) -> pd.DataFrame:
    """information_availability_date: when the quarter's numbers can be assumed public.

    Uses the regulatory deadline, with SEBI's COVID extensions. If a CSV of actual filing dates is
    supplied (bank_id, quarter_end_date, filing_date) the later of filing date and quarter end wins.
    """
    out = df.copy()
    is_q4 = out["quarter_end_date"].dt.month == Q4_MONTH
    lag = np.where(is_q4, LAG_DAYS_AUDITED, LAG_DAYS_UNAUDITED)
    out["information_availability_date"] = out["quarter_end_date"] + pd.to_timedelta(lag, unit="D")
    keys = out["quarter_end_date"].dt.strftime("%Y-%m-%d")
    for quarter_end, deadline in SEBI_DEADLINE_OVERRIDES.items():
        out.loc[keys == quarter_end, "information_availability_date"] = pd.Timestamp(deadline)
    if filing_dates is not None:
        actual = pd.read_csv(filing_dates, parse_dates=["quarter_end_date", "filing_date"])
        out = out.merge(actual, on=["bank_id", "quarter_end_date"], how="left")
        out["information_availability_date"] = out["filing_date"].fillna(out["information_availability_date"])
        out = out.drop(columns="filing_date")
    return out


def check_identities(df: pd.DataFrame, tolerance_share: float = 1e-4, npa_tolerance_share: float = 0.25) -> pd.DataFrame:
    """Flag accounting breaks. Tolerance is the larger of ₹1 crore and a share of total assets.

    Gross NPA - Net NPA = NPA provisions is only approximate: net NPA also deducts items such as
    DICGC/ECGC claims and floating provisions. So that test uses a wide tolerance and is a soft flag.
    """
    out = df.copy()
    tolerance = (out["raw_total_assets"].abs() * tolerance_share).clip(lower=1.0)
    balance_gap = out["raw_total_assets"] - (out["raw_total_liabilities_ex_equity"] + out["raw_total_equity"])
    out["flag_balance_sheet_break"] = (balance_gap.abs() >= tolerance).astype(int)
    held = out["raw_provisions_held"]
    npa_gap = out["raw_gross_minus_net_npa"] - held
    npa_tolerance = (out["raw_gross_npa"].abs() * npa_tolerance_share).clip(lower=1.0)
    out["flag_npa_provision_gap"] = ((npa_gap.abs() >= npa_tolerance) & held.notna()).astype(int)
    # In early global returns some banks reported net NPA close to gross NPA, far from their audited figure.
    share = out["raw_net_npa"] / out["raw_gross_npa"].where(out["raw_gross_npa"] > 50)
    typical = share.groupby(out["bank_id"]).transform(lambda s: s.rolling(9, center=True, min_periods=4).median())
    out["flag_net_npa_suspect"] = ((share >= 0.9) & (typical < 0.85)).astype(int)
    return out


def forward_fill_features(df: pd.DataFrame, train_end: str, limit: int = 2, scope: str = "train") -> pd.DataFrame:
    """Forward-fill each feature within a bank, at most `limit` quarters, never from the future.

    scope="train" fills only rows up to train_end, as the brief asks. scope="all" also fills test rows,
    which is still causal because forward-filling looks backwards only.
    """
    out = df.sort_values(["bank_id", "quarter_end_date"]).copy()
    for feature in FEATURES:
        filled = out.groupby("bank_id")[feature].ffill(limit=limit)
        eligible = out["calendar_quarter"] <= train_end if scope == "train" else pd.Series(True, index=out.index)
        take = out[feature].isna() & filled.notna() & eligible
        out[f"imputed_{feature}"] = take.astype(int)
        out.loc[take, feature] = filled[take]
    return out


def coverage_report(df: pd.DataFrame, train_end: str) -> pd.DataFrame:
    columns = FEATURES + [c for c in df.columns if c.startswith("raw_") and c not in ("raw_flow_scope",)]
    rows = []
    for name, part in (("train", df[df["calendar_quarter"] <= train_end]), ("test", df[df["calendar_quarter"] > train_end])):
        for column in columns:
            rows.append({"split": name, "column": column, "rows": len(part), "missing_pct": round(100 * part[column].isna().mean(), 2),
                         "imputed_rows": int(part.get(f"imputed_{column}", pd.Series(dtype=int)).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/panel/panel_raw.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/panel/panel_clean.csv"))
    parser.add_argument("--coverage", type=Path, default=Path("data/validation/coverage_report.csv"))
    parser.add_argument("--column-map", type=Path)
    parser.add_argument("--filing-dates", type=Path)
    parser.add_argument("--train-end", default="2021Q4")
    parser.add_argument("--fill-scope", choices=["train", "all"], default="train")
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    if args.column_map:
        raw = raw.rename(columns=json.loads(args.column_map.read_text()))
    panel = add_information_dates(standardise_dates(raw), args.filing_dates)
    panel = check_identities(panel)
    ratios = compute_ratios(panel)  # the Dec 2013 lag row is still here, so 2014Q1 ROA and NIM exist
    panel = pd.concat([panel, ratios], axis=1)
    suspect = panel["flag_net_npa_suspect"] == 1
    panel.loc[suspect, NET_NPA_RATIOS] = np.nan  # raw_net_npa itself stays as RBI published it
    if "is_lag_only_row" in panel:
        panel = panel[~panel["is_lag_only_row"].astype(bool)].drop(columns="is_lag_only_row")
    panel = forward_fill_features(panel, args.train_end, scope=args.fill_scope)
    report = coverage_report(panel, args.train_end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.coverage.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False, float_format="%.4f", date_format="%Y-%m-%d")
    report.to_csv(args.coverage, index=False)
    print(f"{len(panel)} rows, {panel['bank_id'].nunique()} banks")
    print("balance sheet breaks:", int(panel["flag_balance_sheet_break"].sum()),
          "| NPA vs provision gaps:", int(panel["flag_npa_provision_gap"].sum()),
          "| suspect net NPA rows:", int(panel["flag_net_npa_suspect"].sum()))


if __name__ == "__main__":
    main()
