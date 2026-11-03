"""Write indian_banks_registry.csv from the curated seed plus ISINs seen in NSE exchange files.

Usage: python build_registry.py --prices data/market/market_daily_prices.csv --panel data/panel/panel_raw.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bank_registry_seed import BANKS

RENAMES = {"IDFCFIRSTB": ["IDFCBANK", "IDFCFIRSTB"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=Path("data/market/market_daily_prices.csv"))
    parser.add_argument("--panel", type=Path, default=Path("data/panel/panel_raw.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/reference/indian_banks_registry.csv"))
    args = parser.parse_args()
    prices = pd.read_csv(args.prices, parse_dates=["date"], dtype={"isin": str})
    panel = pd.read_csv(args.panel)
    panel = panel[~panel["is_lag_only_row"]]
    all_quarters = list(pd.period_range("2014Q1", "2026Q1", freq="Q").astype(str))
    rows = []
    for bank_id, name, tier, nse, bse, start, end, reason, acquirer, merged, slugs, note in BANKS:
        symbols = RENAMES.get(nse, [nse]) if nse else []
        quotes = prices[prices["symbol"].isin(symbols + [nse])] if nse else prices.iloc[0:0]
        with_isin = quotes[quotes["isin"].notna() & (quotes["isin"] != "")].sort_values("date")
        history = with_isin.groupby("isin", sort=False)["date"].min().sort_values()
        have = set(panel.loc[panel["bank_id"] == bank_id, "calendar_quarter"])
        expected = [q for q in all_quarters if start <= q <= end]
        rows.append({
            "bank_id": bank_id, "bank_name": name, "ownership_tier": tier,
            "ownership_tier_before_change": "PSB" if bank_id == "IDBI" else "",
            "ownership_change_date": "2019-01-21" if bank_id == "IDBI" else "",
            "nse_symbol": nse, "nse_symbol_history": "IDFCBANK until 2019-01-15, then IDFCFIRSTB" if bank_id == "IDFCFIRST" else "",
            "bse_code": {"INGVYSYA": "531807"}.get(bank_id, bse),
            "isin": history.index[-1] if len(history) else "",
            "isin_history": "; ".join(f"{i} from {d.date()}" for i, d in history.items()),
            "isin_last_seen_date": with_isin["date"].max().date() if len(with_isin) else "",
            "first_price_date": quotes["date"].min().date() if len(quotes) else "",
            "last_price_date": quotes["date"].max().date() if len(quotes) else "",
            "active_start_quarter": start, "active_end_quarter": end, "exit_reason": reason,
            "acquiring_bank_id": acquirer, "merger_effective_date": merged,
            "first_data_quarter": min(have) if have else "", "last_data_quarter": max(have) if have else "",
            "missing_quarters_in_rbi_data": ";".join(q for q in expected if q not in have),
            "rbi_series_slug": ";".join(slugs), "notes": note,
        })
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} banks to {args.output}")


if __name__ == "__main__":
    main()
