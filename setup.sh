#!/usr/bin/env bash
# Install everything the minijev POC and web playground need. Safe to run again.
#
#   ./setup.sh              Python deps (poc/) and web deps (web/)
#   ./setup.sh --model      also download Qwen2.5-0.5B-Instruct (~1 GB) so the first run starts fast
#   ./setup.sh --model-1.5b also download Qwen2.5-1.5B-Instruct (~3 GB)
#
# It never installs system tools by itself. If one is missing, it says what to install and stops.
set -euo pipefail
cd "$(dirname "$0")"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*"; exit 1; }

MODELS=()
for arg in "$@"; do
  case "$arg" in
    --model) MODELS+=("Qwen/Qwen2.5-0.5B-Instruct") ;;
    --model-1.5b) MODELS+=("Qwen/Qwen2.5-1.5B-Instruct") ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) fail "unknown option: $arg (try --help)" ;;
  esac
done

bold "1. Tools"
command -v uv >/dev/null || fail "uv is missing. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
ok "uv $(uv --version | cut -d' ' -f2)"
command -v node >/dev/null || fail "Node.js is missing. Install Node 20 or newer (https://nodejs.org, or: brew install node)"
NODE_MAJOR=$(node -p 'process.versions.node.split(".")[0]')
[ "$NODE_MAJOR" -ge 20 ] || fail "Node $NODE_MAJOR is too old; the web app needs Node 20 or newer"
ok "node $(node --version)"
command -v npm >/dev/null || fail "npm is missing (it ships with Node.js)"
ok "npm $(npm --version)"
if command -v tilt >/dev/null; then ok "tilt $(tilt version | cut -d, -f1) (optional: 'tilt up' runs everything)"
else printf '  - tilt not found (optional). Install: brew install tilt-dev/tap/tilt\n'; fi

bold "2. Python environment (poc/)"
# uv installs Python 3.12 if it is missing. torch is pinned to 2.2.2: the last build for Intel Macs.
(cd poc && uv sync)
ok "poc/.venv: $(cd poc && uv run --no-sync python -c 'import torch, transformers; print(f"torch {torch.__version__}, transformers {transformers.__version__}")')"

bold "3. Web app (web/)"
(cd web && npm ci --no-audit --no-fund)
ok "web/node_modules"

if [ ${#MODELS[@]} -gt 0 ]; then
  bold "4. Model weights (Hugging Face cache)"
  for m in "${MODELS[@]}"; do
    (cd poc && uv run --no-sync python -c "from huggingface_hub import snapshot_download as s; s('$m')" >/dev/null)
    ok "$m"
  done
fi

bold "Done. Start the playground with one of:"
echo "  tilt up                                   # both servers, with a dashboard at http://localhost:10350"
echo "  cd poc && uv run uvicorn server:app --port 8000   # then, in a second terminal:"
echo "  cd web && npm run dev                     # open http://localhost:3000"
echo "  docker compose up --build                 # everything in Docker"
