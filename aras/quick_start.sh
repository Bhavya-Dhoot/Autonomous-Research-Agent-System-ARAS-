#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.12+ and re-run."
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r aras/requirements.txt

# Download NLTK data (punkt for tokenization)
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
# Set matplotlib backend for non-interactive use
python3 -c "import matplotlib; matplotlib.use('Agg')"

cp -n aras/.env.example .env || true

python3 aras/main.py --topic "${1:-Autonomous Research Agents}"

