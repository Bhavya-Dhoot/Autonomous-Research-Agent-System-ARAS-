$ErrorActionPreference = "Stop"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  Write-Host "Python launcher 'py' not found. Install Python 3.12+ from python.org, then re-run."
  exit 1
}

py -m venv .venv
& .\.venv\Scripts\Activate.ps1
py -m pip install -U pip
py -m pip install -r .\aras\requirements.txt

# Download NLTK data (punkt for tokenization)
py -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
# Set matplotlib backend for non-interactive use
py -c "import matplotlib; matplotlib.use('Agg')"

if (-not (Test-Path .\.env)) {
  Copy-Item .\aras\.env.example .\.env
}

$topic = $args[0]
if (-not $topic) { $topic = "Autonomous Research Agents" }

py .\aras\main.py --topic $topic

