from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from auxiliary_brain import plugin
from auxiliary_brain.training import TrainingError


def training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    plugin.setup_cli(parser)
    return parser


def test_training_parser_defaults_and_options() -> None:
    parser = training_parser()

    status = parser.parse_args(["train", "status"])
    assert status.train_command == "status"
    assert status.json is False

    prepare = parser.parse_args(
        [
            "train",
            "prepare",
            "--task",
            "generic_extract",
            "--seed",
            "7",
            "--holdout-percent",
            "25",
            "--min-examples",
            "8",
            "--min-train",
            "6",
            "--min-holdout",
            "2",
            "--acknowledge-unattributed-gateway",
            "--allow-small",
            "--json",
        ]
    )
    assert prepare.task == "generic_extract"
    assert prepare.seed == 7
    assert prepare.holdout_percent == 25
    assert prepare.min_examples == 8
    assert prepare.min_train == 6
    assert prepare.min_holdout == 2
    assert prepare.acknowledge_unattributed_gateway is True
    assert prepare.allow_small is True
    assert prepare.json is True

    run = parser.parse_args(["train", "run"])
    assert run.bundle is None
    assert run.smoke is False
    assert run.allow_cpu is False
    assert run.seed == 42
    assert run.max_length == 512
    assert run.epochs == 3.0
    assert run.max_steps is None
    assert run.learning_rate == 0.0001
    assert run.gradient_accumulation == 4
    assert run.timeout is None

    convert = parser.parse_args(["train", "convert"])
    assert convert.run is None
    assert convert.timeout == 900.0

    evaluate = parser.parse_args(["train", "evaluate"])
    assert evaluate.run is None
    assert evaluate.port == plugin.DEFAULT_EVALUATION_PORT
    assert evaluate.startup_timeout == 600.0

    assert parser.parse_args(["train", "promote"]).run is None
    assert parser.parse_args(["train", "rollback"]).train_command == "rollback"

    logs = parser.parse_args(["train", "logs"])
    assert logs.run is None
    assert logs.stage == "trainer"
    assert logs.lines == 100


def test_train_status_and_prepare_dispatch_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = training_parser()
    monkeypatch.setattr(plugin, "training_status", lambda: {"ready": True, "latest": None})

    assert plugin.brain_command(parser.parse_args(["train", "status", "--json"])) == 0
    assert json.loads(capsys.readouterr().out) == {"ready": True, "latest": None}

    observed: dict[str, Any] = {}

    def fake_prepare(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {
            "created": True,
            "path": tmp_path / "bundle-abc",
            "manifest": {
                "counts": {"train": 6, "holdout": 2},
                "promotion": {"promotable": False},
            },
        }

    monkeypatch.setattr(plugin, "prepare_training", fake_prepare)
    args = parser.parse_args(
        [
            "train",
            "prepare",
            "--task",
            "generic_extract",
            "--seed",
            "7",
            "--holdout-percent",
            "25",
            "--min-examples",
            "8",
            "--min-train",
            "6",
            "--min-holdout",
            "2",
            "--acknowledge-unattributed-gateway",
            "--allow-small",
            "--json",
        ]
    )

    assert plugin.brain_command(args) == 0
    assert observed == {
        "task_key": "generic_extract",
        "seed": 7,
        "holdout_percent": 25,
        "min_unique_examples": 8,
        "min_train_examples": 6,
        "min_holdout_examples": 2,
        "acknowledge_unattributed_gateway": True,
        "allow_small": True,
    }
    output = json.loads(capsys.readouterr().out)
    assert output["created"] is True
    assert output["path"] == str(tmp_path / "bundle-abc")


def test_train_run_convert_and_logs_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = training_parser()
    calls: dict[str, Any] = {}

    def fake_run(bundle: str | None, **kwargs: Any) -> dict[str, Any]:
        calls["run"] = {"bundle": bundle, **kwargs}
        return {"run_id": "run-7", "status": "trained"}

    def fake_convert(run: str | None, *, timeout_seconds: float) -> dict[str, Any]:
        calls["convert"] = {"run": run, "timeout_seconds": timeout_seconds}
        return {"run_id": "run-7"}

    def fake_logs(run: str | None, *, stage: str, lines: int) -> str:
        calls["logs"] = {"run": run, "stage": stage, "lines": lines}
        return "tiny trainer survived"

    monkeypatch.setattr(plugin, "run_training", fake_run)
    monkeypatch.setattr(plugin, "convert_training_run", fake_convert)
    monkeypatch.setattr(plugin, "read_training_logs", fake_logs)

    run_args = parser.parse_args(
        [
            "train",
            "run",
            "bundle-abc",
            "--smoke",
            "--allow-cpu",
            "--seed",
            "7",
            "--max-length",
            "384",
            "--epochs",
            "1.5",
            "--max-steps",
            "3",
            "--learning-rate",
            "0.0002",
            "--gradient-accumulation",
            "2",
            "--timeout",
            "45",
        ]
    )
    assert plugin.brain_command(run_args) == 0
    assert calls["run"] == {
        "bundle": "bundle-abc",
        "smoke": True,
        "allow_cpu": True,
        "seed": 7,
        "max_length": 384,
        "epochs": 1.5,
        "max_steps": 3,
        "learning_rate": 0.0002,
        "gradient_accumulation_steps": 2,
        "timeout_seconds": 45.0,
    }
    assert "Training complete: run-7" in capsys.readouterr().out

    assert (
        plugin.brain_command(parser.parse_args(["train", "convert", "run-7", "--timeout", "12"]))
        == 0
    )
    assert calls["convert"] == {"run": "run-7", "timeout_seconds": 12.0}
    assert "Adapter converted: run-7" in capsys.readouterr().out

    assert (
        plugin.brain_command(
            parser.parse_args(["train", "logs", "run-7", "--stage", "converter", "--lines", "4"])
        )
        == 0
    )
    assert calls["logs"] == {"run": "run-7", "stage": "converter", "lines": 4}
    assert capsys.readouterr().out == "tiny trainer survived\n"


@pytest.mark.parametrize(
    ("quality_passed", "expected_code", "expected_quality"),
    [(True, 0, "pass"), (False, 1, "fail")],
)
def test_train_evaluate_uses_quality_gate_exit_code(
    quality_passed: bool,
    expected_code: int,
    expected_quality: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = training_parser()
    observed: dict[str, Any] = {}

    def fake_evaluate(run: str | None, **kwargs: Any) -> dict[str, Any]:
        observed.update(run=run, **kwargs)
        return {
            "run_id": "run-7",
            "evaluation": {
                "quality_passed": quality_passed,
                "promotion_eligible": quality_passed,
            },
        }

    monkeypatch.setattr(plugin, "evaluate_training_run", fake_evaluate)
    args = parser.parse_args(
        ["train", "evaluate", "run-7", "--port", "9191", "--startup-timeout", "12"]
    )

    assert plugin.brain_command(args) == expected_code
    assert observed == {"run": "run-7", "port": 9191, "startup_timeout": 12.0}
    output = capsys.readouterr().out
    assert f"quality    : {expected_quality}" in output
    assert ("promotion  : eligible" if quality_passed else "promotion  : not eligible") in output


def test_train_promote_and_rollback_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = training_parser()
    observed: list[str | None] = []

    def fake_promote(run: str | None) -> dict[str, Any]:
        observed.append(run)
        return {"active": {"run_id": "run-7"}, "managed_server_restarted": True}

    def fake_rollback() -> dict[str, Any]:
        observed.append("rollback")
        return {"active": None, "managed_server_restarted": False}

    monkeypatch.setattr(plugin, "promote_training_run", fake_promote)
    monkeypatch.setattr(plugin, "rollback_training_deployment", fake_rollback)

    assert plugin.brain_command(parser.parse_args(["train", "promote", "run-7"])) == 0
    assert "Promoted adapter: run-7" in capsys.readouterr().out
    assert plugin.brain_command(parser.parse_args(["train", "rollback"])) == 0
    assert "Rolled back to: unchanged base model" in capsys.readouterr().out
    assert observed == ["run-7", "rollback"]


@pytest.mark.parametrize(
    ("argv", "target"),
    [
        (["train", "status"], "training_status"),
        (["train", "prepare"], "prepare_training"),
        (["train", "run"], "run_training"),
        (["train", "convert"], "convert_training_run"),
        (["train", "evaluate"], "evaluate_training_run"),
        (["train", "promote"], "promote_training_run"),
        (["train", "rollback"], "rollback_training_deployment"),
        (["train", "logs"], "read_training_logs"),
    ],
)
def test_training_failures_are_safe_cli_errors(
    argv: list[str],
    target: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise TrainingError("tiny trainer dropped its spear")

    monkeypatch.setattr(plugin, target, fail)

    assert plugin.brain_command(training_parser().parse_args(argv)) == 1
    assert capsys.readouterr().out == "Auxiliary brain: tiny trainer dropped its spear\n"
