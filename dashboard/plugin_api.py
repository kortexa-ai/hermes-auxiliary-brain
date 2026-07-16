"""Authenticated dashboard routes for Hermes Auxiliary Brain.

Hermes mounts this router under ``/api/plugins/auxiliary-brain`` and applies
its normal dashboard authentication middleware.  Authentication deliberately
stays in the host; this module only defines the plugin's bounded API surface.
"""

from __future__ import annotations

try:
    from auxiliary_brain.local_api import redact_tree
    from auxiliary_brain.plugin import RUNTIME
    from auxiliary_brain.runtime import (
        REMOTE_INPUT_MAX_CHARS,
        BrainRuntimeError,
        resolve_api_key,
    )
except ModuleNotFoundError as exc:
    # Directory plugins are namespaced by Hermes when they are not also on
    # sys.path as a checkout. Dashboard API files are loaded separately, so
    # use that canonical namespace as the installed-plugin fallback.
    if exc.name not in {"auxiliary_brain", "auxiliary_brain.plugin"}:
        raise
    from hermes_plugins.auxiliary_brain.auxiliary_brain.local_api import redact_tree
    from hermes_plugins.auxiliary_brain.auxiliary_brain.plugin import RUNTIME
    from hermes_plugins.auxiliary_brain.auxiliary_brain.runtime import (
        REMOTE_INPUT_MAX_CHARS,
        BrainRuntimeError,
        resolve_api_key,
    )

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter()


class CheckinBody(BaseModel):
    """One fixed-purpose local progress check-in."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=REMOTE_INPUT_MAX_CHARS)


@router.get("/status")
def status(refresh: bool = False) -> dict:
    """Return the shared structured status report."""

    try:
        return redact_tree(RUNTIME.status(refresh=refresh), resolve_api_key())
    except BrainRuntimeError as exc:
        raise HTTPException(status_code=503, detail=_safe_detail(exc)) from exc


@router.post("/checkin")
def checkin(body: CheckinBody) -> dict:
    """Run the fixed progress-checkin task without accepting runtime overrides."""

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be blank")
    try:
        result = RUNTIME.run(
            "progress_checkin",
            text,
            source="dashboard-api",
        )
        return redact_tree(
            {
                "output": result.output,
                "model": result.model,
                "latency_ms": result.latency_ms,
                "prediction_id": result.prediction_id,
            },
            resolve_api_key(),
        )
    except BrainRuntimeError as exc:
        raise HTTPException(status_code=503, detail=_safe_detail(exc)) from exc


def _safe_detail(exc: Exception) -> str:
    try:
        return redact_tree(str(exc), resolve_api_key())
    except BrainRuntimeError:
        return "The local auxiliary brain is unavailable. Run `hermes brain doctor` locally."
