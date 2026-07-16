"""Hermes-facing runtime for the local auxiliary brain."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import BrainConfig
from .local_api import (
    EndpointProbe,
    LocalAPIError,
    OpenAICompatibleClient,
    discover_endpoint,
    probe_endpoint,
)
from .store import BrainStore, PredictionRecord
from .tasks import BASE_INSTRUCTION, TaskParseError, get_task
from .version import __version__

PLUGIN_ID = "auxiliary-brain"
PLUGIN_VERSION = __version__
AUXILIARY_TASK_KEY = "auxiliary_brain_reflex"
API_KEY_ENV = "AUXILIARY_BRAIN_API_KEY"


class BrainRuntimeError(RuntimeError):
    """A user-facing local-brain operation failed safely."""


@dataclass(frozen=True, slots=True)
class RunResult:
    task_key: str
    output: dict[str, Any]
    raw_output: str
    model: str
    base_url: str
    latency_ms: float
    event_id: str | None = None
    prediction_id: str | None = None
    repaired: bool = False


class BrainRuntime:
    """Resolve config, call only the configured local endpoint, and persist data."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stores: dict[Path, BrainStore] = {}
        self._probe_cache: tuple[tuple[Any, ...], float, EndpointProbe, str] | None = None

    def config(self) -> BrainConfig:
        """Load the active Hermes profile's plugin and auxiliary-task settings."""

        try:
            from hermes_cli.config import load_config

            root = load_config() or {}
        except Exception as exc:  # pragma: no cover - host import failure
            raise BrainRuntimeError(f"cannot read Hermes config: {exc}") from exc

        entries = _mapping(_mapping(root.get("plugins")).get("entries"))
        entry = _mapping(entries.get(PLUGIN_ID))
        settings = dict(_mapping(entry.get("config")))

        # Endpoint routing belongs to Hermes' registered auxiliary task.  Keep
        # credentials out of config.yaml; the optional key comes from .env.
        auxiliary = _mapping(root.get("auxiliary"))
        route = _mapping(auxiliary.get(AUXILIARY_TASK_KEY))
        provider = str(route.get("provider") or "custom").strip().lower()
        if provider != "custom":
            raise BrainRuntimeError(
                f"auxiliary.{AUXILIARY_TASK_KEY}.provider must be custom; "
                "run `hermes brain setup --auto` to repair it"
            )
        if route.get("api_key"):
            raise BrainRuntimeError(
                f"remove auxiliary.{AUXILIARY_TASK_KEY}.api_key from config.yaml; "
                f"put the credential in {API_KEY_ENV} instead"
            )
        if route.get("base_url"):
            settings["base_url"] = route["base_url"]
        if route.get("model"):
            settings["model"] = route["model"]
        if route.get("timeout") is not None:
            settings["timeout_seconds"] = route["timeout"]
        settings.pop("api_key", None)
        settings["api_key"] = os.environ.get(API_KEY_ENV) or None

        try:
            return BrainConfig.from_mapping(settings)
        except (TypeError, ValueError) as exc:
            raise BrainRuntimeError(f"invalid auxiliary-brain config: {exc}") from exc

    def save_configuration(
        self,
        *,
        base_url: str,
        model: str,
        mode: str = "explicit",
        capture: bool = True,
        auto_discover: bool = True,
        timeout_seconds: float = 8.0,
        discovery_timeout_seconds: float = 0.75,
        max_input_chars: int = 8_000,
    ) -> BrainConfig:
        """Atomically persist behavior and auxiliary routing without credentials."""

        candidate = BrainConfig(
            mode=mode,
            base_url=base_url,
            model=model,
            auto_discover=auto_discover,
            capture=capture,
            timeout_seconds=timeout_seconds,
            discovery_timeout_seconds=discovery_timeout_seconds,
            max_input_chars=max_input_chars,
        )
        raw, config_path = _read_raw_config_strict()

        plugins = raw.setdefault("plugins", {})
        if not isinstance(plugins, dict):
            raise BrainRuntimeError("config.yaml plugins must be a mapping")
        entries = plugins.setdefault("entries", {})
        if not isinstance(entries, dict):
            raise BrainRuntimeError("config.yaml plugins.entries must be a mapping")
        entry = entries.setdefault(PLUGIN_ID, {})
        if not isinstance(entry, dict):
            raise BrainRuntimeError(f"config.yaml plugins.entries.{PLUGIN_ID} must be a mapping")
        behavior = entry.setdefault("config", {})
        if not isinstance(behavior, dict):
            raise BrainRuntimeError(
                f"config.yaml plugins.entries.{PLUGIN_ID}.config must be a mapping"
            )
        behavior.update(
            {
                "mode": candidate.mode,
                "auto_discover": candidate.auto_discover,
                "capture": candidate.capture,
                "discovery_timeout_seconds": candidate.discovery_timeout_seconds,
                "max_input_chars": candidate.max_input_chars,
            }
        )
        # Older/manual versions may have put endpoint secrets in the plugin
        # block. Remove them while setup already has the file open.
        behavior.pop("base_url", None)
        behavior.pop("model", None)
        behavior.pop("api_key", None)

        auxiliary = raw.setdefault("auxiliary", {})
        if not isinstance(auxiliary, dict):
            raise BrainRuntimeError("config.yaml auxiliary must be a mapping")
        route = auxiliary.setdefault(AUXILIARY_TASK_KEY, {})
        if not isinstance(route, dict):
            raise BrainRuntimeError(f"config.yaml auxiliary.{AUXILIARY_TASK_KEY} must be a mapping")
        route.update(
            {
                "provider": "custom",
                "model": candidate.model or "",
                "base_url": candidate.base_url or "",
                "timeout": candidate.timeout_seconds,
            }
        )
        route.pop("api_key", None)

        try:
            from hermes_cli.config import atomic_config_write

            config_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_config_write(config_path, raw, sort_keys=False)
        except Exception as exc:
            raise BrainRuntimeError(f"cannot save {config_path}: {exc}") from exc

        with self._lock:
            self._probe_cache = None
        return candidate

    def set_mode(self, mode: str, *, capture: bool | None = None) -> BrainConfig:
        """Change behavior without contacting or validating the model server."""

        raw, config_path = _read_raw_config_strict()
        plugins = raw.setdefault("plugins", {})
        if not isinstance(plugins, dict):
            raise BrainRuntimeError("config.yaml plugins must be a mapping")
        entries = plugins.setdefault("entries", {})
        if not isinstance(entries, dict):
            raise BrainRuntimeError("config.yaml plugins.entries must be a mapping")
        entry = entries.setdefault(PLUGIN_ID, {})
        if not isinstance(entry, dict):
            raise BrainRuntimeError(f"config.yaml plugins.entries.{PLUGIN_ID} must be a mapping")
        behavior = entry.setdefault("config", {})
        if not isinstance(behavior, dict):
            raise BrainRuntimeError(
                f"config.yaml plugins.entries.{PLUGIN_ID}.config must be a mapping"
            )
        try:
            candidate = BrainConfig.from_mapping(
                {
                    "mode": mode,
                    "capture": behavior.get("capture", True) if capture is None else capture,
                }
            )
        except (TypeError, ValueError) as exc:
            raise BrainRuntimeError(f"invalid auxiliary-brain mode: {exc}") from exc
        behavior["mode"] = candidate.mode
        behavior["capture"] = candidate.capture
        try:
            from hermes_cli.config import atomic_config_write

            config_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_config_write(config_path, raw, sort_keys=False)
        except Exception as exc:
            raise BrainRuntimeError(f"cannot save {config_path}: {exc}") from exc
        return candidate

    def probe(self, *, refresh: bool = False) -> tuple[EndpointProbe, str]:
        cfg = self.config()
        cache_key = (
            cfg.base_url,
            cfg.model,
            cfg.auto_discover,
            cfg.api_key,
            cfg.discovery_timeout_seconds,
        )
        with self._lock:
            cached = self._probe_cache
            if (
                not refresh
                and cached is not None
                and cached[0] == cache_key
                and time.monotonic() - cached[1] < 30
            ):
                return cached[2], cached[3]

        try:
            if cfg.base_url:
                probe = probe_endpoint(
                    cfg.base_url,
                    api_key=cfg.api_key,
                    timeout=cfg.discovery_timeout_seconds,
                )
            elif cfg.auto_discover:
                if cfg.api_key:
                    raise BrainRuntimeError(
                        f"{API_KEY_ENV} is set but no base URL is configured; run "
                        "`hermes brain setup --base-url <loopback-url>` so the token "
                        "is sent only to the endpoint you selected"
                    )
                found = discover_endpoint(
                    api_key=cfg.api_key,
                    timeout=cfg.discovery_timeout_seconds,
                )
                if found is None:
                    raise BrainRuntimeError(
                        "no local OpenAI-compatible server found; start one and run "
                        "`hermes brain setup --auto`"
                    )
                probe = found
            else:
                raise BrainRuntimeError(
                    "no local endpoint configured; run `hermes brain setup --auto`"
                )
        except ValueError as exc:
            raise BrainRuntimeError(f"unsafe local endpoint configuration: {exc}") from exc

        if not probe.reachable:
            raise BrainRuntimeError(
                f"local endpoint {probe.base_url} is unavailable: {probe.error}"
            )
        model = probe.choose_model(cfg.model, strict=bool(cfg.model))
        if not model:
            if cfg.model:
                available = ", ".join(probe.models) or "none"
                raise BrainRuntimeError(
                    f"configured model {cfg.model!r} is not exposed by {probe.base_url}; "
                    f"available: {available}"
                )
            raise BrainRuntimeError(
                f"local endpoint {probe.base_url} exposed no model; load a model first"
            )
        with self._lock:
            self._probe_cache = (cache_key, time.monotonic(), probe, model)
        return probe, model

    def run(
        self,
        task_key: str,
        text: str,
        *,
        source: str,
        session_id: str | None = None,
        capture: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunResult:
        cfg = self.config()
        if cfg.mode == "off":
            raise BrainRuntimeError("auxiliary brain is off; run `hermes brain mode explicit`")
        if not text.strip():
            raise BrainRuntimeError("input text cannot be empty")
        try:
            task = get_task(task_key)
        except KeyError as exc:
            raise BrainRuntimeError(str(exc)) from exc

        clipped = cfg.clip_input(text.strip())
        probe, model = self.probe()
        client = OpenAICompatibleClient(
            probe.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout_seconds,
        )
        messages = task.build_messages(clipped)
        started = time.perf_counter()
        raw = self._complete(client, task, messages, model=model)
        repaired = False
        try:
            output = task.parse(raw)
        except TaskParseError as first_error:
            repair_messages = [
                *messages,
                {"role": "assistant", "content": raw[:4_000]},
                {
                    "role": "user",
                    "content": (
                        "The previous output did not satisfy the JSON schema "
                        f"({first_error}). Return one corrected JSON object only."
                    ),
                },
            ]
            raw = client.complete(
                repair_messages,
                model=model,
                temperature=0.0,
                max_tokens=task.max_tokens,
            )
            try:
                output = task.parse(raw)
            except TaskParseError as exc:
                raise BrainRuntimeError(f"local model returned invalid {task_key}: {exc}") from exc
            repaired = True
        latency_ms = round((time.perf_counter() - started) * 1_000, 1)

        event_id = None
        prediction_id = None
        should_capture = cfg.capture if capture is None else capture
        if should_capture:
            event_metadata = {
                **dict(metadata or {}),
                "source": source,
                "plugin_version": PLUGIN_VERSION,
                "task_contract_hash": _task_contract_hash(task),
                "repaired": repaired,
            }
            event = self.store().record_event(
                kind="local_task",
                input_text=clipped,
                session_id=session_id,
                task_key=task_key,
                metadata=event_metadata,
            )
            prediction = self.store().record_prediction(
                event_id=event.id,
                task_key=task_key,
                output=output,
                raw_output=raw,
                model=model,
                base_url=probe.base_url,
                latency_ms=latency_ms,
            )
            event_id = event.id
            prediction_id = prediction.id

        return RunResult(
            task_key=task_key,
            output=output,
            raw_output=raw,
            model=model,
            base_url=probe.base_url,
            latency_ms=latency_ms,
            event_id=event_id,
            prediction_id=prediction_id,
            repaired=repaired,
        )

    def correct(
        self,
        prediction_id: str,
        corrected: Mapping[str, Any],
        *,
        note: str | None = None,
    ) -> PredictionRecord:
        prediction = self.store().get_prediction(prediction_id)
        if prediction is None:
            raise BrainRuntimeError(f"prediction not found: {prediction_id}")
        task = get_task(prediction.task_key)
        try:
            validated = task.parse(json.dumps(dict(corrected), ensure_ascii=False))
        except TaskParseError as exc:
            raise BrainRuntimeError(
                f"correction does not match {prediction.task_key}: {exc}"
            ) from exc
        self.store().record_correction(
            prediction_id=prediction.id,
            corrected=validated,
            note=note,
        )
        return prediction

    def export(
        self,
        destination: str | Path | None = None,
        *,
        task_key: str | None = None,
        corrected_only: bool = True,
    ) -> tuple[Path, int]:
        if destination is None:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            destination = self.data_root() / "exports" / f"training-{stamp}.jsonl"
        path = Path(destination).expanduser().resolve()
        count = self.store().export_jsonl(
            path,
            task_key=task_key,
            corrected_only=corrected_only,
        )
        return path, count

    def evaluate(
        self,
        *,
        task_key: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        examples = self.store().training_examples(
            task_key=task_key,
            corrected_only=True,
            limit=limit,
        )
        if not examples:
            raise BrainRuntimeError("no corrected examples are available to evaluate")
        exact = 0
        fields = 0
        matching_fields = 0
        failures: list[dict[str, str]] = []
        for example in examples:
            try:
                result = self.run(
                    example["task"],
                    example["input"],
                    source="evaluation",
                    capture=False,
                )
            except BrainRuntimeError as exc:
                failures.append({"prediction_id": example["prediction_id"], "error": str(exc)})
                continue
            expected = _without_confidence(example["output"])
            actual = _without_confidence(result.output)
            if actual == expected:
                exact += 1
            for key, value in expected.items():
                fields += 1
                if actual.get(key) == value:
                    matching_fields += 1
        return {
            "examples": len(examples),
            "completed": len(examples) - len(failures),
            "exact_matches": exact,
            "exact_accuracy": round(exact / len(examples), 4),
            "field_accuracy": round(matching_fields / fields, 4) if fields else 0.0,
            "failures": failures,
        }

    def status(self, *, refresh: bool = False) -> dict[str, Any]:
        from .diagnostics import build_status_report

        return build_status_report(self, refresh=refresh)

    def doctor(self) -> dict[str, Any]:
        from .diagnostics import build_doctor_report

        return build_doctor_report(self)

    def store(self) -> BrainStore:
        path = self.data_root() / "brain.db"
        with self._lock:
            store = self._stores.get(path)
            if store is None:
                store = BrainStore(path)
                self._stores[path] = store
            return store

    @staticmethod
    def data_root() -> Path:
        try:
            from hermes_constants import get_hermes_home

            home = Path(get_hermes_home())
        except Exception:  # pragma: no cover - standalone fallback
            configured = os.environ.get("HERMES_HOME")
            home = Path(configured) if configured else Path.home() / ".hermes"
        return home.expanduser().resolve() / "auxiliary-brain"

    @staticmethod
    def _complete(client: OpenAICompatibleClient, task: Any, messages: list, *, model: str) -> str:
        try:
            return client.complete(
                messages,
                model=model,
                temperature=task.temperature,
                max_tokens=task.max_tokens,
                response_format=task.response_format,
            )
        except LocalAPIError as exc:
            # Several otherwise-compatible local servers do not implement
            # json_schema response_format. The schema remains in the prompt.
            if not any(code in str(exc) for code in ("HTTP 400", "HTTP 404", "HTTP 422")):
                raise BrainRuntimeError(str(exc)) from exc
            try:
                return client.complete(
                    messages,
                    model=model,
                    temperature=task.temperature,
                    max_tokens=task.max_tokens,
                )
            except LocalAPIError as retry_exc:
                raise BrainRuntimeError(str(retry_exc)) from retry_exc


def _read_raw_config_strict() -> tuple[dict[str, Any], Path]:
    try:
        from hermes_cli.config import fast_safe_load, get_config_path

        path = Path(get_config_path())
    except Exception as exc:  # pragma: no cover - host import failure
        raise BrainRuntimeError(f"cannot locate Hermes config: {exc}") from exc
    if not path.exists():
        return {}, path
    try:
        with path.open(encoding="utf-8") as handle:
            value = fast_safe_load(handle)
    except Exception as exc:
        raise BrainRuntimeError(
            f"refusing to change unreadable or invalid config {path}: {exc}"
        ) from exc
    if value is None:
        return {}, path
    if not isinstance(value, dict):
        raise BrainRuntimeError(f"config root must be a mapping: {path}")
    return value, path


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _without_confidence(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "confidence"}


def _task_contract_hash(task: Any) -> str:
    contract = {
        "base_instruction": BASE_INSTRUCTION,
        "key": task.key,
        "instruction": task.instruction,
        "schema": task.schema,
        "max_tokens": task.max_tokens,
        "temperature": task.temperature,
    }
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
