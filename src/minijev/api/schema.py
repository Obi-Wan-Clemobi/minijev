"""Request shapes of the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Req(BaseModel):
    state: object = Field(..., description="Text, or any JSON value")
    questions: dict[str, dict]
    mode: str = "packed"
    settings: dict = Field(default_factory=dict, description="Settings overrides, e.g. {'temp_noul': 2.72}")


class CompareReq(Req):
    methods: list[str] = ["readout", "logprobs_cached", "generate_cached", "generate_json"]


class ModelReq(BaseModel):
    name: str
