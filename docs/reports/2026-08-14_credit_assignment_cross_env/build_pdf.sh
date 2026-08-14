#!/bin/bash
# Rebuild the report end-to-end: collect -> figures -> PDF.
set -e
cd "$(dirname "$0")"
python3 assets/collect_results.py
python3 assets/build_figures.py
~/.local/bin/tectonic report.tex
echo "built $(pwd)/report.pdf"
