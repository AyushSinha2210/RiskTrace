"""Build daily returns, quarterly return-correlation edges and the quarterly macro table.

Price sources (combine them freely):
  --prices CSV          already extracted rows (data/market/market_daily_prices.csv is supplied).
  --bhavcopy-dir DIR    a folder of NSE bhavcopy CSVs; the only free source that still has delisted banks.
  --yfinance            pull listed banks from Yahoo (nse_symbol.NS) to fill recent dates. Needs internet.
Yahoo drops delisted tickers, so using it alone removes every merged bank from the graph.

Usage:
  python fetch_market_and_macro.py --prices data/market/market_daily_prices.csv \
      --registry data/reference/indian_banks_registry.csv --out-dir data
Add --yfinance --yf-start 2025-05-01 on a machine with internet to close the 2025-26 gap.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAX_GAP_DAYS = 5  # a return across a longer gap is not a daily return
# Split and large-bonus ex-dates, found from price ratios and confirmed by an ISIN change or a matching ratio.
CORPORATE_ACTION_DAYS = {
    ("AXISBANK", "2014-07-28"), ("BANKBARODA", "2015-01-22"), ("CANBK", "2024-05-15"), ("CORPBANK", "2015-01-22"),
    ("FEDERALBNK", "2013-10-17"), ("FEDERALBNK", "2015-07-08"), ("HDFCBANK", "2019-09-19"), ("ICICIBANK", "2014-12-04"),
    ("J&KBANK", "2014-09-04"), ("KARURVYSYA", "2016-11-17"), ("KOTAKBANK", "2015-07-08"), ("PNB", "2014-12-18"),
    ("SBIN", "2014-11-20"), ("YESBANK", "2017-09-21"),
}
RENAMED_SYMBOLS = {"IDFCBANK": "IDFCFIRSTB"}


def read_bhavcopy_folder(folder: Path, symbols: set[str]) -> pd.DataFrame:
    """Read NSE bhavcopies in either the old (CLOSE, PREVCLOSE) or the sec_bhavdata (CLOSE_PRICE) layout."""
    rows = []
    for path in sorted(folder.glob("*.csv")):
        try:
            day = pd.to_datetime(path.name[:9], format="%d%b%Y")
        except ValueError:
            continue
        with open(path, newline="", errors="replace") as handle:
            reader = csv.reader(handle)
            header = [c.strip().upper() for c in next(reader)]
            index = {name: i for i, name in enumerate(header)}
            close = index.get("CLOSE", index.get("CLOSE_PRICE"))
            volume = index.get("TOTTRDQTY", index.get("TTL_TRD_QNTY"))
            for row in reader:
                if row and row[0].strip() in symbols and row[index["SERIES"]].strip() in ("EQ", "BE", "BZ"):
                    isin = row[index["ISIN"]].strip() if "ISIN" in index and index["ISIN"] < len(row) else ""
                    rows.append((day, row[0].strip(), row[index["SERIES"]].strip(), float(row[close]), float(row[volume]), isin))
    return pd.DataFrame(rows, columns=["date", "symbol", "series", "close", "volume", "isin"])


def fetch_yfinance(symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Dividend- and split-adjusted closes, so corporate action days need no special handling."""
    import yfinance as yf  # imported here so offline runs do not need it

    frames = []
    for symbol in symbols:
        data = yf.download(f"{symbol}.NS", start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            print(f"  no Yahoo data for {symbol}")
            continue
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        part = data.reset_index()[["Date", "Close", "Volume"]]
        part.columns = ["date", "close", "volume"]
        part["symbol"], part["series"], part["isin"] = symbol, "YF", ""
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "symbol", "series", "close", "volume", "isin"])


def daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns per symbol, blank across data gaps and on corporate action ex-dates."""
    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].replace(RENAMED_SYMBOLS)
    out = out.sort_values(["symbol", "date", "series"]).drop_duplicates(["symbol", "date"], keep="first")
    previous = out.groupby("symbol")["close"].shift(1)
    gap = out.groupby("symbol")["date"].diff().dt.days
    out["log_return"] = np.log(out["close"] / previous).where(gap <= MAX_GAP_DAYS)
    action = [(s, d.strftime("%Y-%m-%d")) in CORPORATE_ACTION_DAYS for s, d in zip(out["symbol"], out["date"])]
    out["corporate_action_day"] = np.array(action, dtype=int)
    out.loc[out["corporate_action_day"] == 1, "log_return"] = np.nan
    return out


def quarterly_correlations(returns: pd.DataFrame, quarter_ends: pd.DatetimeIndex, window_days: int = 90, min_obs: int = 30) -> pd.DataFrame:
    """Pearson correlation R_ij^t over the window_days calendar days strictly before each quarter end."""
    wide = returns.pivot(index="date", columns="bank_id", values="log_return").sort_index()
    rows = []
    for quarter_end in quarter_ends:
        window = wide[(wide.index < quarter_end) & (wide.index >= quarter_end - pd.Timedelta(days=window_days))]
        if window.empty:
            continue
        window = window.dropna(axis=1, thresh=min_obs)
        correlation = window.corr(min_periods=min_obs)
        counts = window.notna().astype(int).T @ window.notna().astype(int)
        columns = list(correlation.columns)
        for a in range(len(columns)):
            for b in range(a + 1, len(columns)):
                value = correlation.iat[a, b]
                if pd.notna(value):
                    rows.append((quarter_end.date(), columns[a], columns[b], round(float(value), 5), int(counts.iat[a, b])))
    return pd.DataFrame(rows, columns=["quarter_end_date", "bank_i", "bank_j", "rho", "n_obs"])


def to_quarter_end(month_start) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(month_start)) + pd.offsets.QuarterEnd(0)


def build_macro(source: Path, tbill_csv: Path | None = None) -> pd.DataFrame:
    quarters = pd.date_range("2014-03-31", "2026-03-31", freq="QE")
    macro = pd.DataFrame({"quarter_end_date": quarters})
    repo = pd.read_csv(source / "repo_rate_changes.csv", parse_dates=["effective_date"]).sort_values("effective_date")
    macro = pd.merge_asof(macro, repo, left_on="quarter_end_date", right_on="effective_date").rename(columns={"repo_rate_pct": "rbi_repo_rate"}).drop(columns="effective_date")
    yields = pd.read_csv(source / "fred_INDIRLTLT01STQ.csv", parse_dates=["observation_date"])
    yields["quarter_end_date"] = to_quarter_end(yields["observation_date"])
    macro = macro.merge(yields.rename(columns={"value": "india_10y_gsec_yield"})[["quarter_end_date", "india_10y_gsec_yield"]], how="left")
    macro["india_10y_minus_repo"] = macro["india_10y_gsec_yield"] - macro["rbi_repo_rate"]
    if tbill_csv is not None:  # columns: quarter_end_date, tbill_91d
        tbill = pd.read_csv(tbill_csv, parse_dates=["quarter_end_date"])
        macro = macro.merge(tbill, how="left")
        macro["india_yield_curve_slope"] = macro["india_10y_gsec_yield"] - macro["tbill_91d"]
    else:
        macro["india_yield_curve_slope"] = np.nan
    old = json.loads((source / "gdp-quarterly-levels.json").read_text())
    new = json.loads((source / "gdp-2022-quarterly-levels.json").read_text())
    growth = {}
    for table, keep_from, keep_to in ((old, "2013-01-01", "2023-03-31"), (new, "2023-04-01", "2026-12-31")):
        # YoY is computed within one base year. Comparing levels across bases would be wrong.
        level = pd.Series(table["series"]["gdp_constant"]["data"], index=to_quarter_end(table["dates"]), dtype=float)
        yoy = (100 * (level / level.shift(4) - 1)).dropna()
        growth.update(yoy[(yoy.index >= keep_from) & (yoy.index <= keep_to)].to_dict())
    macro["india_real_gdp_yoy"] = macro["quarter_end_date"].map(growth)
    macro["gdp_base"] = np.where(macro["quarter_end_date"] <= "2023-03-31", "2011-12", "2022-23")
    cpi = json.loads((source / "cpi-2024-linked.json").read_text())
    monthly = pd.Series(cpi["series"]["headline"]["data"], index=pd.to_datetime(cpi["dates"]), dtype=float).dropna()
    # The source series is already year-on-year inflation; average the three months of each quarter.
    macro["india_cpi_inflation"] = macro["quarter_end_date"].map(monthly.groupby(monthly.index + pd.offsets.QuarterEnd(0)).mean().round(3))
    macro["india_real_gdp_yoy_lag1"] = macro["india_real_gdp_yoy"].shift(1)  # GDP is released about two months after quarter end
    return macro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--bhavcopy-dir", type=Path)
    parser.add_argument("--yfinance", action="store_true")
    parser.add_argument("--yf-start", default="2025-05-01")
    parser.add_argument("--yf-end")
    parser.add_argument("--registry", type=Path, default=Path("data/reference/indian_banks_registry.csv"))
    parser.add_argument("--macro-source", type=Path, default=Path("data/source_macro"))
    parser.add_argument("--tbill-csv", type=Path)
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--min-obs", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    registry = pd.read_csv(args.registry, dtype=str, keep_default_na=False)
    listed = registry[registry["nse_symbol"] != ""]
    symbol_to_bank = dict(zip(listed["nse_symbol"], listed["bank_id"]))
    parts = []
    if args.prices:
        parts.append(pd.read_csv(args.prices))
    if args.bhavcopy_dir:
        parts.append(read_bhavcopy_folder(args.bhavcopy_dir, set(symbol_to_bank) | set(RENAMED_SYMBOLS)))
    if args.yfinance:
        parts.append(fetch_yfinance(sorted(symbol_to_bank), args.yf_start, args.yf_end))
    if not parts:
        parser.error("give at least one of --prices, --bhavcopy-dir, --yfinance")
    columns = ["date", "symbol", "series", "close", "volume", "isin"]
    returns = daily_log_returns(pd.concat([p[columns] for p in parts], ignore_index=True))
    returns["bank_id"] = returns["symbol"].map(symbol_to_bank)
    returns = returns.dropna(subset=["bank_id"])
    market_dir = args.out_dir / "market"
    market_dir.mkdir(parents=True, exist_ok=True)
    returns[["date", "bank_id", "symbol", "series", "close", "volume", "isin", "log_return", "corporate_action_day"]].to_csv(
        market_dir / "market_daily_prices.csv", index=False, date_format="%Y-%m-%d", float_format="%.6f")
    quarter_ends = pd.date_range("2014-03-31", "2026-03-31", freq="QE")
    edges = quarterly_correlations(returns, quarter_ends, args.window_days, args.min_obs)
    edges.to_csv(market_dir / "market_correlation_edges.csv", index=False)
    coverage = edges.groupby("quarter_end_date").agg(pairs=("rho", "size"), median_rho=("rho", "median"), median_obs=("n_obs", "median")).reset_index()
    coverage["banks"] = [len(set(edges.loc[edges["quarter_end_date"] == q, "bank_i"]) | set(edges.loc[edges["quarter_end_date"] == q, "bank_j"])) for q in coverage["quarter_end_date"]]
    coverage.to_csv(market_dir / "market_correlation_coverage.csv", index=False)
    macro_dir = args.out_dir / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)
    build_macro(args.macro_source, args.tbill_csv).to_csv(macro_dir / "macro_quarterly.csv", index=False, date_format="%Y-%m-%d", float_format="%.4f")
    print(f"prices {len(returns)}, edges {len(edges)}, quarters with edges {edges['quarter_end_date'].nunique()}")


if __name__ == "__main__":
    main()
