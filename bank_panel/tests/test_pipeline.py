"""Unit tests. The small frames below are test fixtures only and never enter the data files.
Run from the project root: python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import audit_temporal_leakage as audit  # noqa: E402
import build_labels  # noqa: E402
import build_rbi_panel  # noqa: E402
import fetch_market_and_macro as market  # noqa: E402
import ingest_and_clean_panel as ingest  # noqa: E402
import ratios  # noqa: E402


def test_safe_div_blocks_zero_and_negative():
    result = ratios.safe_div(pd.Series([1.0, 1.0, 1.0]), pd.Series([0.0, -5.0, 2.0]))
    assert result.isna().tolist() == [True, True, False]
    assert result.iloc[2] == 50.0


@pytest.mark.parametrize("year,quarter,expected", [(2018, 1, "2017-06-30"), (2018, 2, "2017-09-30"), (2018, 3, "2017-12-31"), (2018, 4, "2018-03-31")])
def test_fiscal_dates(year, quarter, expected):
    assert ingest.fiscal_to_quarter_end(year, quarter) == pd.Timestamp(expected)


def test_information_dates_use_covid_extensions():
    frame = pd.DataFrame({"quarter_end_date": pd.to_datetime(["2019-12-31", "2019-03-31", "2020-03-31", "2020-06-30"])})
    dates = ingest.add_information_dates(frame)["information_availability_date"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == ["2020-02-14", "2019-05-30", "2020-07-31", "2020-09-15"]


def test_single_quarter_needs_previous_quarter():
    dates = pd.DatetimeIndex(pd.to_datetime(["2017-06-30", "2017-09-30", "2017-12-31", "2018-03-31", "2018-06-30"]))
    result = build_rbi_panel.single_quarter(np.array([10.0, 25.0, np.nan, 60.0, 5.0]), dates)
    assert result[0] == 10 and result[1] == 15 and np.isnan(result[2]) and np.isnan(result[3]) and result[4] == 5


def base_row():
    return pd.Series({"raw_gross_npa": 100.0, "raw_net_npa": 40.0, "raw_provisions_held": 60.0, "raw_net_advances": 1000.0,
                      "raw_total_assets": 2000.0, "raw_total_equity": 200.0, "raw_cet1_capital": 150.0, "raw_tier1_capital": 160.0,
                      "raw_total_liabilities_ex_equity": 1800.0})


def test_npa_shock_keeps_identities():
    shocked = ratios.apply_npa_shock(base_row(), 50.0)
    assert np.isclose(shocked["raw_total_assets"], shocked["raw_total_liabilities_ex_equity"] + shocked["raw_total_equity"])
    assert np.isclose(shocked["raw_gross_npa"] - shocked["raw_net_npa"], shocked["raw_provisions_held"])
    assert np.isclose(shocked["raw_total_assets"], 1970.0)


def test_forward_fill_limit_and_direction():
    frame = pd.DataFrame({"bank_id": "A", "quarter_end_date": pd.date_range("2020-03-31", periods=6, freq="QE")})
    frame["calendar_quarter"] = frame["quarter_end_date"].dt.to_period("Q").astype(str)
    for feature in ingest.FEATURES:
        frame[feature] = [np.nan, 1.0, np.nan, np.nan, np.nan, 2.0]
    filled = ingest.forward_fill_features(frame, "2021Q4")
    assert np.isnan(filled["gnpa_ratio"].iloc[0])  # nothing earlier, and never back-filled from later
    assert filled["gnpa_ratio"].iloc[2] == 1.0 and filled["gnpa_ratio"].iloc[3] == 1.0
    assert np.isnan(filled["gnpa_ratio"].iloc[4])  # past the two-quarter limit


def label_frame():
    frame = pd.DataFrame({"bank_id": "A", "quarter_end_date": pd.date_range("2016-03-31", periods=8, freq="QE")})
    return ingest.add_information_dates(frame)


def events_frame(date, end, onset=1, kind="PCA_Entry"):
    return pd.DataFrame({"bank_id": ["A"], "event_type": [kind], "is_onset": [onset], "event_date_used": pd.to_datetime([date]), "exit_date": [end]})


def test_onset_labels_and_blind_window():
    labelled = build_labels.supervisory_labels(label_frame(), events_frame("2017-05-09", "2018-06-01"))
    row = labelled.set_index("quarter_end_date")
    assert row.loc["2017-03-31", "y_event_onset_1"] == 1 and row.loc["2017-03-31", "onset_in_blind_window"] == 1
    assert row.loc["2017-03-31", "mask_onset_usable"] == 0
    assert row.loc["2016-12-31", "y_event_onset_1"] == 0 and row.loc["2016-12-31", "y_event_onset_2"] == 1
    assert row.loc["2016-12-31", "mask_onset_usable"] == 1
    assert row.loc["2017-06-30", "under_pca_or_moratorium_at_t"] == 1 and row.loc["2017-06-30", "y_event_onset_1"] == 0


def test_short_moratorium_is_active_in_its_quarter():
    labelled = build_labels.supervisory_labels(label_frame(), events_frame("2017-02-05", "2017-02-18", kind="Moratorium_S45"))
    row = labelled.set_index("quarter_end_date").loc["2017-03-31"]
    assert row["y_active_regulatory_distress"] == 1 and row["under_pca_or_moratorium_at_t"] == 0


def test_horizon_past_data_end_is_censored():
    labelled = build_labels.supervisory_labels(label_frame(), pd.DataFrame(columns=["bank_id", "event_type", "is_onset", "event_date_used", "exit_date"]).astype({"event_date_used": "datetime64[ns]"}))
    assert np.isnan(labelled["y_event_onset_4"].iloc[-1]) and labelled["y_event_onset_1"].iloc[0] == 0


def test_correlation_ignores_quarter_end_and_later():
    dates = pd.bdate_range("2020-01-01", "2020-04-30")
    rng = np.random.default_rng(1)
    a = rng.normal(size=len(dates))
    b = a + rng.normal(size=len(dates)) * 0.2
    late = dates >= "2020-03-31"
    b_changed = np.where(late, -a * 50, b)
    def tidy(y):
        return pd.concat([pd.DataFrame({"date": dates, "bank_id": "X", "log_return": a}), pd.DataFrame({"date": dates, "bank_id": "Y", "log_return": y})])
    end = pd.DatetimeIndex(["2020-03-31"])
    before = market.quarterly_correlations(tidy(b), end, min_obs=10)["rho"].iloc[0]
    after = market.quarterly_correlations(tidy(b_changed), end, min_obs=10)["rho"].iloc[0]
    assert before == after


def test_returns_skip_gaps_and_ex_dates():
    prices = pd.DataFrame({"date": ["2019-09-18", "2019-09-19", "2019-09-20", "2019-12-20"], "symbol": "HDFCBANK", "series": "EQ", "close": [2187.75, 1101.05, 1199.6, 1200.0], "volume": 1, "isin": ""})
    returns = market.daily_log_returns(prices)["log_return"].tolist()
    assert np.isnan(returns[1]) and not np.isnan(returns[2]) and np.isnan(returns[3])


def test_auditor_catches_a_backdated_information_date():
    frame = ingest.add_information_dates(pd.DataFrame({"quarter_end_date": pd.to_datetime(["2019-12-31", "2019-03-31"])}))
    assert audit.check_information_lag(frame) == []
    frame.loc[0, "information_availability_date"] = pd.Timestamp("2020-01-05")
    assert audit.check_information_lag(frame)


def test_auditor_catches_a_label_with_no_event():
    frame = pd.DataFrame({"bank_id": ["A"], "quarter_end_date": pd.to_datetime(["2017-03-31"]), "y_event_onset_1": [1.0], "y_event_onset_2": [0.0], "y_event_onset_4": [0.0],
                          "under_pca_or_moratorium_at_t": [0], "mask_onset_usable": [1], "onset_in_blind_window": [0]})
    assert audit.check_labels(frame, events_frame("2015-01-01", "2016-01-01"))
