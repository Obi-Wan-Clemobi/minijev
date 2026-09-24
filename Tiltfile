# minijev: run the API and the web playground together.
#   ./setup.sh   (once)
#   tilt up      then press space for the dashboard, or open http://localhost:3000
#
# Both run natively (no containers): the model runs on the CPU through uv. Tilt restarts the API when a
# file in deps changes (the model loads again, ~10 s); the web app hot-reloads by itself.

local_resource(
    "api",
    serve_cmd="uv run uvicorn server:app --port 8000",
    serve_dir="poc",
    deps=["poc/server.py", "poc/minijev_poc.py", "poc/experiments.py", "poc/minijev.env"],
    readiness_probe=probe(period_secs=3, http_get=http_get_action(port=8000, path="/v1/health")),
    links=[link("http://localhost:8000/docs", "API docs")],
    labels=["app"],
)

local_resource(
    "web",
    serve_cmd="npm run dev -- --port 3000",
    serve_dir="web",
    resource_deps=["api"],
    readiness_probe=probe(period_secs=3, http_get=http_get_action(port=3000, path="/")),
    links=[link("http://localhost:3000", "Playground")],
    labels=["app"],
)

# Checks: start them from the dashboard. They do not run on every file change.
local_resource("test-python", cmd="uv run pytest -q", dir="poc", auto_init=False,
               trigger_mode=TRIGGER_MODE_MANUAL, labels=["checks"])
local_resource("test-web", cmd="npm test && npm run lint && npx tsc --noEmit", dir="web", auto_init=False,
               trigger_mode=TRIGGER_MODE_MANUAL, labels=["checks"])
