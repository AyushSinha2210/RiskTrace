"""Part B: derived ratios and balanced shocks. All ratios are pure functions of raw_* columns,
so a perturbed balance sheet can be recomputed deterministically.

Rupee amounts are in crore; ratios are in per cent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(numerator, denominator, scale: float = 100.0):
    """numerator / denominator * scale. Missing or non-positive denominators give NaN, never 0 or inf."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    den = den.where(den > 0)
    return scale * num / den


def average_with_lag(df: pd.DataFrame, column: str) -> pd.Series:
    """Two-point average of a stock with its previous quarter, within each bank.

    NaN when the previous quarter is absent or not exactly one quarter back, so a gap or a bank's
    first quarter can never borrow a value from elsewhere.
    """
    ordered = df.sort_values(["bank_id", "quarter_end_date"])
    dates = pd.to_datetime(ordered["quarter_end_date"])
    previous = ordered.groupby("bank_id")[column].shift(1)
    months = dates.dt.year * 12 + dates.dt.month
    consecutive = months.groupby(ordered["bank_id"]).diff() == 3
    average = (ordered[column] + previous.where(consecutive)) / 2
    return average.reindex(df.index)


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Return the ten node-feature ratios for a panel with bank_id and quarter_end_date columns."""
    out = pd.DataFrame(index=df.index)
    out["gnpa_ratio"] = safe_div(df["raw_gross_npa"], df["raw_gross_advances"])
    out["nnpa_ratio"] = safe_div(df["raw_net_npa"], df["raw_net_advances"])
    out["pcr_ratio"] = safe_div(df["raw_gross_npa"] - df["raw_net_npa"], df["raw_gross_npa"])
    total_capital = df["raw_tier1_capital"] + df["raw_tier2_capital"]
    out["crar_ratio"] = safe_div(total_capital, df["raw_rwa"])
    out["cet1_ratio"] = safe_div(df["raw_cet1_capital"], df["raw_rwa"])
    out["casa_ratio"] = safe_div(df["raw_casa_deposits"], df["raw_total_deposits"])
    out["cd_ratio"] = safe_div(df["raw_net_advances"], df["raw_total_deposits"])
    liquid = df["raw_cash_and_rbi_balances"] + df["raw_slr_gsec_investments"]
    out["slr_liquidity_ratio"] = safe_div(liquid, df["raw_total_assets"])
    average_assets = average_with_lag(df, "raw_total_assets")
    out["roa_annualized"] = safe_div(df["raw_net_profit"], average_assets, scale=400.0)
    out["nim_annualized"] = safe_div(df["raw_net_interest_income"], average_assets, scale=400.0)
    return out


def apply_npa_shock(row: pd.Series, delta_gross_npa: float, provision_share: float | None = None) -> pd.Series:
    """Shock gross NPAs by delta (₹ crore) and keep the accounting identities intact.

    A share c of the new NPA is provided for, c defaults to the bank's current coverage.
      gross NPA +delta, net NPA +(1-c)delta, provisions +c delta,
      net advances and total assets -c delta, equity and CET1/Tier 1 -c delta (pre-tax, so a slight overstatement).
    Assets = Liabilities + Equity and Gross NPA - Net NPA = NPA provisions both still hold afterwards.
    Risk-weighted assets are left unchanged, which is a simplification.
    """
    shocked = row.copy()
    gross, net = row["raw_gross_npa"], row["raw_net_npa"]
    coverage = provision_share if provision_share is not None else ((gross - net) / gross if gross > 0 else 0.0)
    hit = coverage * delta_gross_npa
    shocked["raw_gross_npa"] = gross + delta_gross_npa
    shocked["raw_net_npa"] = net + delta_gross_npa - hit
    shocked["raw_provisions_held"] = row["raw_provisions_held"] + hit
    for column in ("raw_net_advances", "raw_total_assets", "raw_total_equity", "raw_cet1_capital", "raw_tier1_capital"):
        shocked[column] = row[column] - hit
    return shocked


def balance_sheet_similarity(df_quarter: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cosine similarity of asset-scaled balance-sheet vectors for one quarter (for edge rebuilding)."""
    scaled = df_quarter.set_index("bank_id")[columns].div(df_quarter.set_index("bank_id")["raw_total_assets"], axis=0)
    scaled = scaled.dropna()
    matrix = scaled.to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = np.nan
    unit = matrix / norms
    return pd.DataFrame(unit @ unit.T, index=scaled.index, columns=scaled.index)
