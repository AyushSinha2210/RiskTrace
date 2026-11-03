#!/bin/bash
# Rebuilds every derived file from data/source_* and the supplied prices. Run from this folder.
set -e
python pipeline/build_rbi_panel.py
python pipeline/build_registry.py
python pipeline/fetch_market_and_macro.py --prices data/market/market_daily_prices.csv --tbill-csv data/source_macro/tbill_91d_quarterly.csv "$@"
python pipeline/ingest_and_clean_panel.py
python pipeline/build_labels.py
python pipeline/validate_against_rbi_annual.py
python pipeline/audit_temporal_leakage.py
python -m pytest tests -q
