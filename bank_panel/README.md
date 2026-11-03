# Indian bank quarterly panel, Q1 2014 to Q1 2026

49 banks, 1,901 bank-quarters, 49 quarters. Rupee amounts are in crore and ratios are in per cent.

## Start here

| File | What it is |
|---|---|
| `data/panel/panel_final.csv` | Main file: raw variables, ratios, labels, flags, one row per bank-quarter |
| `data/market/market_correlation_edges.csv` | Pairwise return correlations per quarter, for graph edges |
| `data/market/market_daily_prices.csv` | Daily NSE closes and log returns, including delisted banks |
| `data/macro/macro_quarterly.csv` | Repo rate, 10-year yield, GDP growth, CPI inflation |
| `data/reference/indian_banks_registry.csv` | Deliverable 1 |
| `data/reference/rbi_distress_events.csv` | Deliverable 3, with a source URL per event |
| `data/reference/data_dictionary.csv` | Deliverable 2 |
| `data/reference/structural_breaks.csv` | Mergers that make an acquirer's series jump |
| `data/validation/` | Cross-checks and the coverage report |
| `pipeline/` | Deliverable 4, plus the panel and registry builders |

To rebuild everything: `pip install -r requirements.txt`, then `./run_all.sh`.
To fill the 2025-26 price gap on a machine with internet: `./run_all.sh --yfinance`.

## Where the data came from

These are not downloads from each bank's website. They could not be reached from the build environment,
and the sources below are better for a panel anyway.

- **Bank financials: RBI DBIE report 1221**, "Variables to be published bank and bank group wise". These are the
  banks' own off-site returns to RBI. Taken from a public copy at
  https://github.com/bebhuvan/measure-of-india (`data/series/banks-*.json`, vintage 2026-10-10).
- **Annual cross-check: RBI bank-wise tables** on NPAs, CRAR, sensitive sectors and maturity profile. Taken from a
  public copy at https://github.com/Arnab-Afk/bigO.
- **Prices: NSE daily bhavcopies**, from a public archive at https://github.com/girishg4t/nse-bse-bhavcopy.
- **Macro:** MoSPI GDP and CPI and RBI 91-day T-bill auction yields (DBIE report 307), all from the same repository as RBI 1221;
  FRED INDIRLTLT01STQ for the 10-year yield; and a public repo rate history matched to BIS month by month.

All three repositories are third-party copies. The report 1221 copy was checked against an official download from
https://data.rbi.org.in (Statistics > Financial Sector > Banking - Performance Indicators) on 3 November 2026: every compared
cell matched to within ₹0.01 crore or 0.01 points (see `official_1221_vs_github_copy.csv`). Cite RBI directly.
Check NSE's terms if you redistribute prices.

## How it was checked

- **Balance sheet:** assets equal liabilities plus equity on all 1,901 rows, to within ₹0.03 crore.
- **Recomputed ratios against the ratios RBI prints:** GNPA 100%, NNPA 99.0%, CRAR and CET1 99.7% of rows within 0.05 points.
- **Against RBI's separate annual tables** (median gap and share within 0.5%, global-scope March rows, 2015 to 2025):
  gross NPA 0.00% and 95.1%; deposits 0.00% and 97.5%; net advances 0.00% and 85.4%; CRAR 0.00% and 96.3%;
  net NPA 0.00% and 84.4%; borrowings 0.00% and 90.6%. Investments is the weak one: median 0.94%, 36.8% within 0.5%.
  Row-level gaps are in `annual_crosscheck_rows.csv`.
- **Against 30 figures published elsewhere:** see `published_figure_spotcheck.csv`.
- **Leakage auditor and 16 unit tests:** all pass.
- **Not tested live:** the Yahoo path was tested against a stub because Yahoo is unreachable from the build environment.

## Issues that affect your results

1. **No onset happens after March 2020.** A train/test split at 2021Q4 leaves no positive onset labels and no threshold
   breaches in the test set. Use leave-one-bank-out or leave-one-crisis-out validation: 2015, 2017 to early 2018, and 2019 to 2020.
2. **Positives are few and clustered.** Usable positive rows are 7 (k=1), 20 (k=2) and 46 (k=4), from 13 onset events at 13 banks
   (United Bank's entry predates the first row), and 9 of those events fall in 2017 to January 2018. Banks are not independent there, since RBI tightened PCA that April.
3. **Threshold breach nearly gives the answer.** At k=4, the regime breach flag at t covers 42 of 46 positive rows, with 210 false alarms.
   That is the baseline to beat. Report your model against it, not against chance.
4. **Avoiding tautology with `y_threshold_breach`.** It is a deterministic function of nnpa, crar, cet1 and ROA. Never predict it from the same
   quarter's features. Use `y_new_breach_k`, which is future and excludes banks already in breach, and compare with a persistence baseline.
5. **Use `mask_onset_usable` when training.** It removes banks already under PCA and rows whose onset falls between quarter end and the date
   the numbers became public. RBI acts on March accounts in May and June, so those rows would otherwise leak.
6. **The data are supervisory returns, not what investors saw.** Headline values usually match published results, but details such as sectoral
   advances may not have been public at the time, so the disclosure lag is an assumption for those columns. Capital can also differ at crisis
   points: Yes Bank Mar 2020 CRAR is 12.83 here and 8.46 in RBI's annual table.
7. **Scope changes in March 2015.** RBI's global figures start then, so 2014 rows are domestic operations (`data_scope`). This matters
   for banks with large overseas books, such as BOB and BOI. Sector shares are domestic throughout.
8. **Merger jumps.** Acquirers' series jump at `merger_break_flag = 1`. Exclude those quarters from growth rates.

## Differences from the brief

- **49 banks, not about 42.** Added Bharatiya Mahila Bank and ING Vysya, and all five SBI associates count individually.
- **PCA 2021 framework:** issued 2 November 2021, effective 1 January 2022. It dropped ROA. CRAR and CET1 floors became 11.5 and 8.0.
  Your fixed 10.25 and 6.75 applied only around March 2017, so `y_threshold_breach_regime` uses the rule in force on each date. The static rule is kept as you specified.
- **Dena Bank** had no Section 45 moratorium. It is recorded as a forced amalgamation while under PCA.
- **Disclosure lag:** SEBI extended deadlines for Mar 2020, Jun 2020 and Mar 2021, so those quarters use the extended dates.
- **Threshold level per bank** is calculated, because RBI does not publish it: see the `derived_*` columns in the events file.
- **`raw_provisions_held`** was a mix of stock and flow, now split. Gross minus net NPA is a soft check for the same reason.
- **Yahoo-only prices** would delete every merged bank, so exchange files are the base and Yahoo only fills recent dates.
- **DBIE has no stable public series IDs**, so RBI sources are given by report and table number.

## Known gaps

- **Prices thin out after 9 May 2025.** 2025Q2 and 2025Q3 edges rest on about 31 days. 2025Q4 and 2026Q1 have no edges until you run `--yfinance`.
- **Small bonus issues are not stripped from returns.** Splits and large bonuses are; a 1-for-10 bonus leaves a one-day drop of about 9% (City Union Bank does this often). Yahoo data has no such issue.
- Unlisted banks (SBH, SBP, BMB, Nainital) have no market edges.
- Not published bank by bank, so absent or proxied: bulk deposits, HTM/AFS split, NBFC and commercial real estate exposure, unsecured retail. See the dictionary.
- Three bank-quarters are missing for reasons in the source: Kotak 2015Q3, Bandhan 2019Q3, Nainital 2019Q3. Bandhan also has no data for its first two quarters.
- Entry dates for United, IOB and Dhanlaxmi are month-level and rest on press reports, some conflicting. See `verification` and `notes` in the events file.
