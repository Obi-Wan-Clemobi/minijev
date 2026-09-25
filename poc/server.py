"""Compatibility module: the API lives in the minijev package. `uvicorn server:app` (from poc/) keeps working;
`minijev serve` is the new way to start it."""

from minijev.api.server import app  # noqa: F401
