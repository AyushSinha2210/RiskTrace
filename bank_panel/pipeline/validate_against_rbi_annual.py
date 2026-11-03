"""Check March-end panel values against RBI's separate annual bank-wise tables.

The panel comes from RBI's quarterly off-site returns (report 1221). The annual tables are RBI's
published Statistical Tables, which reflect audited accounts, so they are an independent check.
Usage: python validate_against_rbi_annual.py
Writes data/validation/annual_crosscheck_rows.csv and annual_crosscheck_summary.csv.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

ANNUAL = Path("data/source_rbi_annual")
ALIASES = {
    "BANK OF BARODA": "BOB", "BANK OF INDIA": "BOI", "BANK OF MAHARASHTRA": "BOM", "CANARA BANK": "CANARA",
    "CENTRAL BANK OF INDIA": "CENTRALBK", "INDIAN BANK": "INDIANB", "INDIAN OVERSEAS BANK": "IOB",
    "PUNJAB AND SIND BANK": "PSB", "PUNJAB NATIONAL BANK": "PNB", "STATE BANK OF INDIA": "SBI", "UCO BANK": "UCO",
    "UNION BANK OF INDIA": "UNION", "ALLAHABAD BANK": "ALLAHABAD", "ANDHRA BANK": "ANDHRA", "CORPORATION BANK": "CORPBANK",
    "DENA BANK": "DENA", "ORIENTAL BANK OF COMMERCE": "OBC", "SYNDICATE BANK": "SYNDICATE", "UNITED BANK OF INDIA": "UNITED",
    "VIJAYA BANK": "VIJAYA", "STATE BANK OF BIKANER AND JAIPUR": "SBBJ", "STATE BANK OF HYDERABAD": "SBH",
    "STATE BANK OF MYSORE": "SBM", "STATE BANK OF PATIALA": "SBP", "STATE BANK OF TRAVANCORE": "SBT",
    "BHARATIYA MAHILA BANK": "BMB", "IDBI BANK": "IDBI", "HDFC BANK": "HDFCB", "ICICI BANK": "ICICI", "AXIS BANK": "AXIS",
    "KOTAK MAHINDRA BANK": "KOTAK", "INDUSIND BANK": "INDUSIND", "YES BANK": "YES", "IDFC FIRST BANK": "IDFCFIRST",
    "IDFC BANK": "IDFCFIRST", "BANDHAN BANK": "BANDHAN", "DCB BANK": "DCB", "DEVELOPMENT CREDIT BANK": "DCB",
    "FEDERAL BANK": "FEDERAL", "RBL BANK": "RBL", "RATNAKAR BANK": "RBL", "SOUTH INDIAN BANK": "SOUTHIND",
    "KARUR VYSYA BANK": "KVB", "CITY UNION BANK": "CUB", "TAMILNAD MERCANTILE BANK": "TMB", "KARNATAKA BANK": "KTKBANK",
    "JAMMU AND KASHMIR BANK": "JKBANK", "JAMMU & KASHMIR BANK": "JKBANK", "CSB BANK": "CSB", "CATHOLIC SYRIAN BANK": "CSB",
    "DHANLAXMI BANK": "DHANLAXMI", "NAINITAL BANK": "NAINITAL", "LAKSHMI VILAS BANK": "LVB", "ING VYSYA BANK": "INGVYSYA",
}


def normalise(name: str) -> str:
    name = re.sub(r"[*#@$]|\(.*?\)", " ", name.upper()).replace(".", " ")
    name = re.sub(r"\b(LTD|LIMITED|THE)\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def number(text: str) -> float:
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return np.nan


def read_table(pattern: str, columns: dict[str, int]) -> pd.DataFrame:
    path = next(ANNUAL.glob(pattern))
    rows, year = [], None
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        for line in csv.reader(handle):
            if len(line) <= max(columns.values()):
                continue
            if re.fullmatch(r"\d{4}", line[1].strip()):
                year = int(line[1])
            bank = ALIASES.get(normalise(line[2]))
            if year and bank:
                rows.append({"year": year, "bank_id": bank, **{k: number(line[i]) for k, i in columns.items()}})
    return pd.DataFrame(rows).drop_duplicates(["year", "bank_id"])


def main() -> None:
    panel = pd.read_csv("data/panel/panel_raw.csv", parse_dates=["quarter_end_date"])
    march = panel[panel["quarter_end_date"].dt.month == 3].copy()
    march["year"] = march["quarter_end_date"].dt.year
    annual = read_table("6.*", {"gross_npa": 7, "net_npa": 9})
    annual = annual.merge(read_table("9.*", {"deposits": 11, "borrowings": 20, "investments": 29, "net_advances": 56}), on=["year", "bank_id"], how="outer")
    annual = annual.merge(read_table("3.*", {"tier1_ratio": 9, "crar": 11}), on=["year", "bank_id"], how="outer")
    merged = march.merge(annual, on=["year", "bank_id"], how="inner")
    merged["investments_panel"] = merged["raw_slr_gsec_investments"] + merged["raw_non_slr_investments"]
    pairs = {
        "gross_npa": "raw_gross_npa", "net_npa": "raw_net_npa", "deposits": "raw_total_deposits", "borrowings": "raw_borrowings",
        "investments": "investments_panel", "net_advances": "raw_net_advances", "crar": "rbi_crar_ratio", "tier1_ratio": "rbi_tier1_ratio",
    }
    rows, summary = [], []
    for annual_column, panel_column in pairs.items():
        both = merged[merged[annual_column].notna() & merged[panel_column].notna() & (merged["data_scope"] == "global")]
        gap = both[panel_column] - both[annual_column]
        pct = 100 * gap.abs() / both[annual_column].abs().replace(0, np.nan)
        summary.append({"variable": annual_column, "bank_years_compared": len(both), "median_abs_pct_gap": round(pct.median(), 3),
                        "within_0.5pct": round((pct <= 0.5).mean() * 100, 1), "within_2pct": round((pct <= 2).mean() * 100, 1),
                        "over_10pct": int((pct > 10).sum())})
        rows.append(pd.DataFrame({"variable": annual_column, "bank_id": both["bank_id"], "year": both["year"], "panel_value": both[panel_column],
                                  "annual_value": both[annual_column], "abs_pct_gap": pct.round(3)}))
    out = Path("data/validation")
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).sort_values("abs_pct_gap", ascending=False).to_csv(out / "annual_crosscheck_rows.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "annual_crosscheck_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
