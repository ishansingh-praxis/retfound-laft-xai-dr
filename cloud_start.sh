#!/usr/bin/env bash
# Bootstrap + launch on a fresh cloud GPU machine (Ubuntu/Debian or any Linux with Python >= 3.10).
#
#   export KAGGLE_USERNAME=...  KAGGLE_KEY=...  HF_TOKEN=...
#   bash cloud_start.sh                 # the real study
#   bash cloud_start.sh --smoke         # 30-minute synthetic check of every step first (recommended)
#   bash cloud_start.sh --seeds 1337    # any run_all.py option can be appended
#
# It creates a virtualenv, installs requirements, then starts run_all.py detached (survives an
# SSH disconnect). Watch progress with:  tail -f outputs/<run>/logs/run_all.log
# After an interruption or reboot, run the SAME command again: finished steps are skipped and
# training resumes from its checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
command -v curl >/dev/null || { echo "curl not found (apt-get install -y curl)"; exit 1; }
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install -q -U pip
python -m pip install -q -r requirements.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no NVIDIA GPU visible)"
nohup python run_all.py "$@" > run_all.nohup.log 2>&1 &
echo "started run_all.py (pid $!). Log: $(pwd)/run_all.nohup.log"
echo "step logs: outputs/<run>/logs/    final index: outputs/<run>/RESULTS_INDEX.md"
