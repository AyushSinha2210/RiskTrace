"""Convert RBI DBIE report 1221 (banks-*.json) into a bank-quarter panel with the brief's raw_* schema.

Usage: python build_rbi_panel.py --source data/source_rbi_1221 --out data/panel/panel_raw.csv
Scope rule: global operations where RBI publishes them (Mar 2015 onward), domestic before that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bank_registry_seed import BANKS, STRUCTURAL_BREAKS


def calendar_quarter(date: pd.Timestamp) -> str:
    return f"{date.year}Q{date.quarter}"


class Rbi:
    def __init__(self, source: Path):
        self.tables = {p.stem.replace("banks-", ""): json.loads(p.read_text()) for p in source.glob("banks-*.json")}
        self.dates = self.tables["balance-sheet"]["dates"]
        for name, table in self.tables.items():
            if name not in ("infra", "infra-2014") and table["dates"] != self.dates:
                raise ValueError(f"{name}: dates differ from the balance sheet table")

    def series(self, table: str, slugs: list[str], variable: str, op: str) -> np.ndarray | None:
        """Values on the master date index, coalesced across slugs. None if RBI has no such series."""
        source = self.tables[table]
        found = None
        for slug in slugs:
            entry = source["series"].get(f"{slug}__{variable}__{op}")
            if entry is None:
                continue
            values = pd.Series(entry["data"], index=source["dates"], dtype=float).reindex(self.dates).to_numpy()
            found = values if found is None else np.where(np.isnan(found), values, found)
        return found


    def infrastructure(self, table: str, slugs: list[str]) -> np.ndarray:
        """Total infrastructure lending. These two tables use keys with no operations suffix."""
        source = self.tables[table]
        values = np.full(len(self.dates), np.nan)
        for slug in slugs:
            entry = source["series"].get(f"{slug}__all_infrastructure")
            if entry is not None:
                part = pd.Series(entry["data"], index=source["dates"], dtype=float).reindex(self.dates).to_numpy()
                values = np.where(np.isnan(values), part, values)
        return values


STOCKS = {  # output column: (table, RBI variable)
    "raw_total_assets": ("balance-sheet", "total_assets"),
    "raw_total_capital_and_liabilities": ("balance-sheet", "total_capital_and_liabilities"),
    "raw_paid_up_capital": ("balance-sheet", "paid_up_share_capital"),
    "raw_reserves_surplus": ("balance-sheet", "reserves_and_surplus"),
    "raw_total_deposits": ("balance-sheet", "total_deposits"),
    "raw_current_deposits": ("balance-sheet", "current_deposits"),
    "raw_savings_deposits": ("balance-sheet", "savings_deposits"),
    "raw_time_deposits": ("balance-sheet", "time_deposits"),
    "raw_certificates_of_deposit": ("balance-sheet", "certificates_of_deposits"),
    "raw_deposits_of_banks": ("balance-sheet", "deposits_of_banks"),
    "raw_borrowings": ("balance-sheet", "total_borrowings"),
    "raw_other_liabilities": ("balance-sheet", "other_liabilities"),
    "raw_cash_funds": ("balance-sheet", "cash_funds"),
    "raw_due_from_banks": ("balance-sheet", "due_from_banks_fis_ccps"),
    "raw_slr_gsec_investments": ("balance-sheet", "slr_approved_securities"),
    "raw_non_slr_investments": ("balance-sheet", "non_slr_investments"),
    "raw_bs_net_loans": ("balance-sheet", "net_loans_and_advances"),
    "raw_fixed_assets": ("balance-sheet", "fixed_assets_net"),
    "raw_other_assets": ("balance-sheet", "other_assets"),
    "raw_gross_advances": ("asset-quality", "gross_advances"),
    "raw_net_advances": ("asset-quality", "net_advances"),
    "raw_gross_npa": ("asset-quality", "gross_npas_at_the_end_of_the_period"),
    "raw_net_npa": ("asset-quality", "net_npas"),
    "raw_provisions_held": ("asset-quality", "provisions_for_npas_at_end_of_the_fy"),
    "raw_stressed_advances": ("asset-quality", "stressed_advances"),
    "raw_cet1_capital": ("capital", "common_equity_tier_i_capital"),
    "raw_tier1_capital": ("capital", "tier_i_capital"),
    "raw_tier2_capital": ("capital", "tier_ii_capital"),
    "raw_total_capital": ("capital", "total_capital"),
    "raw_rwa": ("capital", "total_rwas"),
    "rbi_gnpa_ratio": ("asset-quality", "gross_npas_to_gross_advances"),
    "rbi_nnpa_ratio": ("asset-quality", "net_npas_to_net_advances"),
    "rbi_pcr_ratio_write_off_adj": ("asset-quality", "provision_coverage_ratio_write_off_adj"),
    "rbi_crar_ratio": ("capital", "capital_ratio_crar"),
    "rbi_cet1_ratio": ("capital", "cet_i_capital_ratio"),
    "rbi_tier1_ratio": ("capital", "tier_i_capital_ratio"),
    "rbi_nim_annualized": ("performance", "net_interest_margin_nim_annualized"),
    "rbi_roa_annualized": ("performance", "return_on_total_assets_annualized"),
    "rbi_roe_annualized": ("performance", "return_on_equity_annualized"),
    "rbi_cost_income_ratio": ("performance", "efficiency_cost_income_ratio"),
}
FLOWS = {  # April-to-date in the source; converted to single-quarter values
    "raw_net_interest_income": "net_interest_income_nii",
    "raw_net_profit": "net_profit_loss_pat",
    "raw_provision_charge_q": "risk_provisions_total",
    "raw_operating_profit_q": "earning_before_provisions_and_taxes_ebpt",
    "raw_write_offs_q": "write_offs_total",
}
SECTORS = {  # domestic gross advances
    "raw_sec_agriculture": [("sectoral", "agriculture_and_allied_activities_gross_advances"), ("priority", "priority_agriculture_and_allied_activities_gross_advances")],
    "raw_sec_industry": [("sectoral", "industry_gross_advances")],
    "raw_sec_services": [("sectoral", "services_gross_advances")],
    "raw_sec_retail": [("sectoral", "retail_loans_gross_advances")],
    "raw_sec_housing": [("sectoral", "retail_loans_housing_loans_gross_advances")],
    "raw_sec_msme_priority": [("priority", "priority_mse_gross_advances")],
}
# RBI drops a series that is zero throughout, so a wholly absent component really is zero.
ZERO_IF_ABSENT = {"raw_borrowings", "raw_current_deposits", "raw_savings_deposits", "raw_time_deposits",
                  "raw_certificates_of_deposit", "raw_deposits_of_banks", "raw_non_slr_investments",
                  "raw_gross_npa", "raw_net_npa"}


def fiscal_position(date: pd.Timestamp) -> int:
    """1 for June, 2 September, 3 December, 4 March."""
    return {6: 1, 9: 2, 12: 3, 3: 4}[date.month]


def single_quarter(cumulative: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """April-to-date amounts to one-quarter amounts. A missing previous quarter gives NaN, not a guess."""
    result = np.full(len(cumulative), np.nan)
    for i, date in enumerate(dates):
        if fiscal_position(date) == 1:
            result[i] = cumulative[i]
        elif i > 0 and (dates[i - 1] + pd.offsets.QuarterEnd(1)).normalize() == date.normalize():
            result[i] = cumulative[i] - cumulative[i - 1]
    return result


def build(source: Path) -> pd.DataFrame:
    rbi = Rbi(source)
    dates = pd.DatetimeIndex(pd.to_datetime(rbi.dates)) + pd.offsets.MonthEnd(0)
    rows = []
    for bank_id, name, tier, nse, bse, start, end, reason, acquirer, merged, slugs, note in BANKS:
        frame = pd.DataFrame({"bank_id": bank_id, "bank_name": name, "ownership_tier": tier}, index=range(len(dates)))
        frame["quarter_end_date"] = dates
        global_assets = rbi.series("balance-sheet", slugs, "total_assets", "global")
        domestic_assets = rbi.series("balance-sheet", slugs, "total_assets", "domestic")
        use_global = ~np.isnan(global_assets) if global_assets is not None else np.zeros(len(dates), bool)
        frame["data_scope"] = np.where(use_global, "global", "domestic")
        frame.loc[~use_global & np.isnan(domestic_assets), "data_scope"] = "none"
        for column, (table, variable) in STOCKS.items():
            g = rbi.series(table, slugs, variable, "global")
            d = rbi.series(table, slugs, variable, "domestic")
            if g is None and d is None:
                frame[column] = 0.0 if column in ZERO_IF_ABSENT else np.nan
                continue
            g = g if g is not None else np.full(len(dates), np.nan)
            d = d if d is not None else np.full(len(dates), np.nan)
            frame[column] = np.where(use_global, g, d)
        for column, variable in FLOWS.items():
            g = rbi.series("profitability", slugs, variable, "global")
            d = rbi.series("profitability", slugs, variable, "domestic")
            g_q = single_quarter(g, dates) if g is not None else np.full(len(dates), np.nan)
            d_q = single_quarter(d, dates) if d is not None else np.full(len(dates), np.nan)
            # The quarter straddling the switch to global (Mar 2015) has no global Dec value, so it stays domestic.
            frame[column] = np.where(np.isnan(g_q), d_q, g_q)
        profit_global = rbi.series("profitability", slugs, "net_profit_loss_pat", "global")
        profit_global_q = single_quarter(profit_global, dates) if profit_global is not None else np.full(len(dates), np.nan)
        frame["raw_flow_scope"] = np.where(np.isnan(profit_global_q), "domestic", "global")
        for column, sources in SECTORS.items():
            values = np.full(len(dates), np.nan)
            for table, variable in sources:
                candidate = rbi.series(table, slugs, variable, "domestic")
                if candidate is not None:
                    values = np.where(np.isnan(values), candidate, values)
            frame[column] = values
        early = rbi.infrastructure("infra-2014", slugs)
        late = rbi.infrastructure("infra", slugs)
        frame["raw_sec_infrastructure"] = np.where(np.isnan(late), early, late)
        frame["raw_domestic_gross_advances"] = rbi.series("sectoral", slugs, "gross_advances", "domestic")
        frame["calendar_quarter"] = [calendar_quarter(x) for x in dates]
        keep = (frame["calendar_quarter"] >= start) & (frame["calendar_quarter"] <= end) & (frame["data_scope"] != "none")
        # String comparison is safe for YYYYQn. Dec 2013 is retained only as the lag for 2014Q1 ratios.
        lag_row = (frame["calendar_quarter"] == "2013Q4") & (frame["data_scope"] != "none")
        frame["is_lag_only_row"] = lag_row
        rows.append(frame[keep | lag_row])
    panel = pd.concat(rows, ignore_index=True)
    panel["raw_total_equity"] = panel["raw_paid_up_capital"] + panel["raw_reserves_surplus"]
    panel["raw_total_liabilities_ex_equity"] = panel["raw_total_deposits"] + panel["raw_borrowings"] + panel["raw_other_liabilities"]
    panel["raw_casa_deposits"] = panel["raw_current_deposits"] + panel["raw_savings_deposits"]
    panel["raw_cash_and_rbi_balances"] = panel["raw_cash_funds"] + panel["raw_due_from_banks"]
    panel["raw_bulk_wholesale_deposits"] = panel["raw_certificates_of_deposit"] + panel["raw_deposits_of_banks"]
    panel["raw_gross_minus_net_npa"] = panel["raw_gross_npa"] - panel["raw_net_npa"]
    for column in [c for c in panel.columns if c.startswith("raw_sec_")]:
        panel[column.replace("raw_sec_", "sec_share_")] = 100 * panel[column] / panel["raw_domestic_gross_advances"].where(panel["raw_domestic_gross_advances"] > 0)
    # IDBI was a public sector bank until RBI recategorised it on 2019-01-21.
    panel.loc[(panel["bank_id"] == "IDBI") & (panel["quarter_end_date"] < pd.Timestamp("2019-01-21")), "ownership_tier"] = "PSB"
    breaks = {(b, q) for b, q, _ in STRUCTURAL_BREAKS}
    panel["merger_break_flag"] = [int((b, q) in breaks) for b, q in zip(panel["bank_id"], panel["calendar_quarter"])]
    panel["infra_definition"] = np.where(panel["quarter_end_date"] >= pd.Timestamp("2022-06-30"), "2022_list", "2014_list")
    return panel.sort_values(["bank_id", "quarter_end_date"]).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/source_rbi_1221"))
    parser.add_argument("--out", type=Path, default=Path("data/panel/panel_raw.csv"))
    args = parser.parse_args()
    result = build(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, float_format="%.2f", date_format="%Y-%m-%d")
    print(f"wrote {len(result)} rows, {result['bank_id'].nunique()} banks, to {args.out}")
