from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from auxiliary_brain import trainer_backend
from auxiliary_brain.trainer_backend import (
    DEFAULT_BASE_MODEL,
    BundleValidationError,
    RequestValidationError,
    TrainerBackendError,
    inspect_training_bundle,
    parse_training_request,
    run_request,
    select_device,
)


def _request(tmp_path: Path, **updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format_version": 1,
        "bundle_dir": "bundle",
        "output_dir": "artifacts",
    }
    value.update(updates)
    return value


def _messages(answer: str = '{"ok":true}') -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": "Summarize this."},
        {"role": "assistant", "content": answer},
    ]


def _write_bundle(tmp_path: Path, rows: list[dict[str, Any]] | None = None) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with (bundle / "train.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows or [{"messages": _messages(), "example_id": "safe-id"}]:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return bundle


def test_import_does_not_load_machine_learning_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
import json
import sys
import auxiliary_brain.trainer_backend
print(json.dumps(sorted(set(sys.modules) & {"torch", "transformers", "trl", "peft", "datasets"})))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_parse_request_uses_low_memory_lfm_defaults(tmp_path: Path) -> None:
    request = parse_training_request(_request(tmp_path), base_dir=tmp_path)

    assert request.bundle_dir == (tmp_path / "bundle").resolve()
    assert request.output_dir == (tmp_path / "artifacts").resolve()
    assert request.result_path == (tmp_path / "training-result.json").resolve()
    assert request.base_model == DEFAULT_BASE_MODEL
    assert request.allow_cpu is False
    assert request.seed == 42
    assert request.max_length == 512
    assert request.gradient_accumulation_steps == 8
    assert request.lora.target_modules == ("q_proj", "k_proj", "v_proj")
    assert request.lora.rank == 8
    assert request.lora.alpha == 16
    assert request.lora.dropout == 0.05


def test_parse_request_accepts_bounded_explicit_settings(tmp_path: Path) -> None:
    request = parse_training_request(
        _request(
            tmp_path,
            result_path="state/result.json",
            base_model="local/model",
            base_model_revision="abc123",
            allow_cpu=True,
            seed=7,
            max_length=1_024,
            epochs=1.5,
            max_steps=2,
            learning_rate=1e-4,
            gradient_accumulation_steps=4,
            lora={
                "target_modules": ["q_proj", "v_proj"],
                "rank": 4,
                "alpha": 8,
                "dropout": 0.1,
            },
        ),
        base_dir=tmp_path,
    )

    assert request.result_path == (tmp_path / "state" / "result.json").resolve()
    assert request.allow_cpu is True
    assert request.max_steps == 2
    assert request.lora.target_modules == ("q_proj", "v_proj")
    assert request.lora.rank == 4


def test_parse_request_accepts_orchestrator_flat_lora_settings(tmp_path: Path) -> None:
    request = parse_training_request(
        _request(
            tmp_path,
            target_modules=["q_proj", "k_proj", "v_proj"],
            rank=8,
            alpha=16,
            dropout=0.05,
        ),
        base_dir=tmp_path,
    )

    assert request.lora.target_modules == ("q_proj", "k_proj", "v_proj")
    assert request.lora.rank == 8


@pytest.mark.parametrize(
    "updates",
    [
        {"unexpected": True},
        {"format_version": 2},
        {"allow_cpu": 1},
        {"seed": True},
        {"max_length": 4_097},
        {"learning_rate": float("nan")},
        {"gradient_accumulation_steps": 0},
        {"lora": {"target_modules": ["q_proj", "q_proj"]}},
        {"lora": {"rank": 0}},
        {"lora": {"surprise": "party"}},
    ],
)
def test_parse_request_rejects_unsafe_or_ambiguous_values(
    tmp_path: Path,
    updates: dict[str, Any],
) -> None:
    with pytest.raises(RequestValidationError):
        parse_training_request(_request(tmp_path, **updates), base_dir=tmp_path)


def test_parse_request_keeps_result_and_artifacts_separate(tmp_path: Path) -> None:
    with pytest.raises(RequestValidationError, match="result_path"):
        parse_training_request(
            _request(tmp_path, result_path="artifacts"),
            base_dir=tmp_path,
        )

    request = parse_training_request(
        _request(tmp_path, result_path="artifacts/result.json"),
        base_dir=tmp_path,
    )
    assert request.result_path == (tmp_path / "artifacts" / "result.json").resolve()

    with pytest.raises(RequestValidationError, match="bundle_dir"):
        parse_training_request(
            _request(tmp_path, result_path="bundle/train.jsonl"),
            base_dir=tmp_path,
        )


def test_inspect_bundle_streams_conversational_rows(tmp_path: Path) -> None:
    rows = [
        {"messages": _messages('{"value":1}'), "task": "one"},
        {"messages": _messages('{"value":2}'), "task": "two"},
    ]
    bundle = _write_bundle(tmp_path, rows)

    info = inspect_training_bundle(bundle)

    assert info.examples == 2
    assert info.bytes == (bundle / "train.jsonl").stat().st_size
    assert len(info.sha256) == 64
    assert info.train_path == (bundle / "train.jsonl").resolve()


def test_inspect_bundle_verifies_and_exposes_reproducibility_manifest(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    train = bundle / "train.jsonl"
    manifest = {
        "format_version": 1,
        "model": {"id": DEFAULT_BASE_MODEL, "revision": "pinned-commit"},
        "files": {
            "train.jsonl": {
                "sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
                "bytes": train.stat().st_size,
                "examples": 1,
            }
        },
    }
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (bundle / "manifest.json").write_bytes(raw)

    info = inspect_training_bundle(bundle)

    assert info.manifest_path == (bundle / "manifest.json").resolve()
    assert info.manifest_sha256 == hashlib.sha256(raw).hexdigest()
    assert info.model_id == DEFAULT_BASE_MODEL
    assert info.model_revision == "pinned-commit"


def test_inspect_bundle_rejects_train_file_changed_after_manifest(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    train = bundle / "train.jsonl"
    manifest = {
        "format_version": 1,
        "model": {"id": DEFAULT_BASE_MODEL, "revision": "pinned-commit"},
        "files": {
            "train.jsonl": {
                "sha256": "0" * 64,
                "bytes": train.stat().st_size,
                "examples": 1,
            }
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleValidationError, match="does not match"):
        inspect_training_bundle(bundle)


def test_verified_snapshot_rejects_change_after_bundle_inspection(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    info = inspect_training_bundle(bundle)
    (bundle / "train.jsonl").write_text(
        json.dumps({"messages": _messages('{"changed":true}')}) + "\n",
        encoding="utf-8",
    )
    snapshot = tmp_path / "private-cache" / "train.jsonl"

    with pytest.raises(BundleValidationError, match="changed after validation"):
        trainer_backend._copy_verified_train(info, snapshot)

    assert not snapshot.exists()


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"messages": []},
        {"messages": [{"role": "user", "content": "only prompt"}]},
        {
            "messages": [
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": ""},
            ]
        },
        {
            "messages": [
                {"role": "assistant", "content": "answer first"},
                {"role": "user", "content": "prompt last"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
                {"role": "assistant", "content": "answer"},
            ]
        },
    ],
)
def test_inspect_bundle_rejects_invalid_conversations(tmp_path: Path, row: dict[str, Any]) -> None:
    bundle = _write_bundle(tmp_path, [row])
    with pytest.raises(BundleValidationError):
        inspect_training_bundle(bundle)


def test_bundle_errors_never_repeat_dataset_text(tmp_path: Path) -> None:
    secret_text = "do-not-repeat-this-private-example"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "train.jsonl").write_text(f'{{"messages":"{secret_text}"}}\n', encoding="utf-8")

    with pytest.raises(BundleValidationError) as caught:
        inspect_training_bundle(bundle)

    assert secret_text not in str(caught.value)


class _FakeCuda:
    def __init__(self, available: bool, bf16: bool = False) -> None:
        self.available = available
        self.bf16 = bf16

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16

    def get_device_name(self, _index: int) -> str:
        return "Tiny GPU"

    def empty_cache(self) -> None:
        return None


class _FakeMPS:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


def _fake_torch(*, cuda: bool = False, bf16: bool = False, mps: bool = False) -> Any:
    return SimpleNamespace(
        cuda=_FakeCuda(cuda, bf16),
        backends=SimpleNamespace(mps=_FakeMPS(mps)),
        bfloat16="torch.bfloat16",
        float16="torch.float16",
        float32="torch.float32",
    )


def test_device_policy_prefers_cuda_bf16_then_fp16() -> None:
    bf16 = select_device(_fake_torch(cuda=True, bf16=True), allow_cpu=False)
    fp16 = select_device(_fake_torch(cuda=True, bf16=False), allow_cpu=False)

    assert (bf16.kind, bf16.precision, bf16.bf16) == ("cuda", "bf16", True)
    assert (fp16.kind, fp16.precision, fp16.fp16) == ("cuda", "fp16", True)


def test_device_policy_uses_mps_fp16() -> None:
    plan = select_device(_fake_torch(mps=True), allow_cpu=False)
    assert (plan.kind, plan.precision, plan.fp16) == ("mps", "fp16", True)


def test_device_policy_requires_explicit_cpu_acknowledgement() -> None:
    with pytest.raises(TrainerBackendError, match="allow_cpu"):
        select_device(_fake_torch(), allow_cpu=False)

    plan = select_device(_fake_torch(), allow_cpu=True)
    assert (plan.kind, plan.precision) == ("cpu", "fp32")


def test_run_request_writes_atomic_success_state(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "state" / "result.json"
    request_path.write_text(
        json.dumps(_request(tmp_path, result_path=str(result_path))),
        encoding="utf-8",
    )

    def fake_train(request: Any, bundle: Any) -> dict[str, Any]:
        running = json.loads(request.result_path.read_text(encoding="utf-8"))
        assert running["status"] == "running"
        assert bundle.examples == 1
        return {"adapter_dir": str(tmp_path / "adapter")}

    outcome = run_request(request_path, train_fn=fake_train)

    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert outcome.payload["status"] == "complete"
    assert outcome.payload["exit_code"] == 0
    assert persisted["status"] == "complete"
    assert persisted["artifacts"]["adapter_dir"].endswith("adapter")
    assert persisted["adapter_dir"].endswith("adapter")
    assert list(result_path.parent.glob(".*.tmp")) == []


def test_run_request_accepts_existing_orchestrator_workspace(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request_path = run_dir / "trainer-request.json"
    request_path.write_text(
        json.dumps(
            _request(
                tmp_path,
                bundle_dir=str(tmp_path / "bundle"),
                output_dir=str(run_dir),
                result_path=str(run_dir / "trainer-result.json"),
                target_modules=["q_proj", "k_proj", "v_proj"],
                rank=8,
                alpha=16,
                dropout=0.05,
            )
        ),
        encoding="utf-8",
    )

    def fake_train(request: Any, _bundle: Any) -> dict[str, Any]:
        assert request.output_dir == run_dir.resolve()
        return {
            "adapter_dir": str(run_dir / "trainer-artifacts" / "adapter"),
            "base_config_dir": str(run_dir / "trainer-artifacts" / "base_config"),
        }

    outcome = run_request(request_path, train_fn=fake_train)
    persisted = json.loads((run_dir / "trainer-result.json").read_text(encoding="utf-8"))

    assert outcome.payload["status"] == "complete"
    assert persisted["status"] == "complete"
    assert Path(persisted["adapter_dir"]).parts[-2:] == ("trainer-artifacts", "adapter")
    assert Path(persisted["base_config_dir"]).parts[-2:] == (
        "trainer-artifacts",
        "base_config",
    )


def test_run_request_sanitizes_unexpected_training_failure(tmp_path: Path) -> None:
    secret_text = "private-dataset-text-must-not-escape"
    _write_bundle(tmp_path, [{"messages": _messages(secret_text)}])
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request(tmp_path)), encoding="utf-8")

    def fail(_request: Any, _bundle: Any) -> dict[str, Any]:
        raise RuntimeError(f"backend echoed: {secret_text}")

    outcome = run_request(request_path, train_fn=fail)
    serialized = (tmp_path / "request.result.json").read_text(encoding="utf-8")

    assert outcome.payload["status"] == "failed"
    assert outcome.payload["exit_code"] == 1
    assert outcome.payload["error"]["code"] == "training_failed"
    assert secret_text not in serialized


def test_run_request_exits_failed_when_final_result_cannot_be_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_bundle(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request(tmp_path)), encoding="utf-8")
    real_write = trainer_backend._write_json_atomic
    calls = 0

    def fail_final(path: Path, value: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated final publish failure")
        real_write(path, value)

    monkeypatch.setattr(trainer_backend, "_write_json_atomic", fail_final)
    outcome = run_request(request_path, train_fn=lambda *_args: {"adapter_dir": "safe"})

    assert outcome.payload["status"] == "failed"
    assert outcome.payload["exit_code"] == 1
    assert outcome.payload["error"]["code"] == "result_write_failed"


def test_cli_records_invalid_request_without_traceback(tmp_path: Path) -> None:
    request_path = tmp_path / "broken.json"
    request_path.write_text("not json and definitely not a secret example", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "auxiliary_brain" / "trainer_backend.py"),
            "--request",
            str(request_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    result = json.loads((tmp_path / "broken.result.json").read_text(encoding="utf-8"))
    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    assert json.loads(completed.stdout)["status"] == "failed"
    assert result["status"] == "failed"
    assert result["phase"] == "request_validation"


class _FakeDataset:
    column_names = ["messages", "task"]

    def __init__(self, count: int = 1) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Any:
        for _index in range(self.count):
            yield {"messages": _messages()}

    def select_columns(self, columns: list[str]) -> _FakeDataset:
        assert columns == ["messages"]
        self.column_names = columns
        return self


class _FakeConfig:
    model_max_length = 2_048
    max_position_embeddings = 2_048
    _commit_hash = "resolved-model-commit"
    use_cache = True

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}\n", encoding="utf-8")


class _FakeTokenizer:
    chat_template = "{% generation %}{{ message.content }}{% endgeneration %}"
    pad_token_id = 0
    eos_token_id = 7
    model_max_length = 1_024

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> dict[str, list[int]]:
        assert messages
        return {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 1]}

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")


class _FakeModel:
    def __init__(self) -> None:
        self.config = _FakeConfig()

    def save_pretrained(self, path: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(b"safe adapter")

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        return (123, 230_000_000)


def test_train_lora_wires_memory_safe_sft_and_publishes_converter_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_bundle(tmp_path)
    request = parse_training_request(
        _request(tmp_path, allow_cpu=True, max_length=2_048, max_steps=1),
        base_dir=tmp_path,
    )
    bundle = inspect_training_bundle(bundle_dir)
    calls: dict[str, Any] = {}
    model = _FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _FakeTokenizer:
            calls["tokenizer"] = (model_id, kwargs)
            return _FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _FakeModel:
            calls["model"] = (model_id, kwargs)
            return model

    def fake_set_seed(seed: int, *, deterministic: bool) -> None:
        calls["seed"] = (seed, deterministic)

    class FakeLoraConfig:
        def __init__(self, **kwargs: Any) -> None:
            calls["lora"] = kwargs

    class FakeSFTConfig:
        def __init__(self, **kwargs: Any) -> None:
            calls["sft"] = kwargs

    class FakeSFTTrainer:
        def __init__(self, **kwargs: Any) -> None:
            calls["trainer"] = kwargs
            self.model = kwargs["model"]

        def train(self) -> Any:
            return SimpleNamespace(
                metrics={
                    "train_loss": 0.25,
                    "train_runtime": 1.0,
                    "unsafe_text": "a dataset row would be filtered",
                    "nan_metric": float("nan"),
                }
            )

    def fake_load_dataset(*args: Any, **kwargs: Any) -> _FakeDataset:
        calls["dataset"] = (args, kwargs)
        return _FakeDataset()

    dependencies = trainer_backend._MLDependencies(
        torch=_fake_torch(),
        transformers=SimpleNamespace(
            set_seed=fake_set_seed,
            AutoTokenizer=FakeAutoTokenizer,
            AutoModelForCausalLM=FakeAutoModel,
        ),
        trl=SimpleNamespace(SFTConfig=FakeSFTConfig, SFTTrainer=FakeSFTTrainer),
        peft=SimpleNamespace(LoraConfig=FakeLoraConfig),
        datasets=SimpleNamespace(load_dataset=fake_load_dataset),
        versions={
            "torch": "2.9.0",
            "transformers": "5.2.0",
            "trl": "0.29.0",
            "peft": "0.18.0",
            "datasets": "4.0.0",
            "accelerate": "1.10.0",
        },
    )
    monkeypatch.setattr(trainer_backend, "_load_ml_dependencies", lambda: dependencies)
    request.output_dir.mkdir()

    artifacts = trainer_backend.train_lora(request, bundle)

    assert calls["seed"] == (42, True)
    assert calls["model"][1] == {
        "dtype": "torch.float32",
        "device_map": {"": "cpu"},
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    assert calls["lora"] == {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    sft = calls["sft"]
    assert sft["per_device_train_batch_size"] == 1
    assert sft["max_length"] == 1_024
    assert sft["max_length"] <= request.max_length
    assert sft["packing"] is False
    assert sft["assistant_only_loss"] is True
    assert sft["report_to"] == "none"
    assert sft["optim"] == "adamw_torch"
    assert sft["gradient_checkpointing"] is True
    assert sft["dataloader_num_workers"] == 0
    assert "quantization_config" not in calls["model"][1]
    assert artifacts["resolved_revision"] == "resolved-model-commit"
    assert Path(artifacts["artifact_dir"]).name == "trainer-artifacts"
    assert Path(artifacts["adapter_dir"], "adapter_model.safetensors").is_file()
    assert Path(artifacts["base_config_dir"], "config.json").is_file()
    assert Path(artifacts["base_config_dir"], "tokenizer.json").is_file()

    metadata = json.loads(Path(artifacts["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["base_model"]["config_dir"] == "base_config"
    assert metadata["dataset"]["sha256"] == bundle.sha256
    assert metadata["device"]["precision"] == "fp32"
    assert metadata["device"]["flash_attention"] is False
    assert metadata["device"]["bitsandbytes"] is False
    assert metadata["lora"]["trainable_parameters"] == 123
    assert metadata["training"]["metrics"] == {
        "train_loss": 0.25,
        "train_runtime": 1.0,
    }
    assert metadata["training"]["assistant_mask_preflight"] == {
        "examples": 1,
        "minimum_assistant_tokens": 2,
    }


def test_assistant_mask_preflight_rejects_fully_truncated_answers() -> None:
    class TruncatingTokenizer:
        def apply_chat_template(self, _messages: Any, **kwargs: Any) -> dict[str, list[int]]:
            if kwargs["truncation"]:
                return {"input_ids": [1, 2], "assistant_masks": [0, 1]}
            return {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 1]}

    dataset = [{"messages": _messages("private answer that must not appear in errors")}]

    with pytest.raises(TrainerBackendError) as caught:
        trainer_backend._preflight_assistant_masks(
            dataset,
            TruncatingTokenizer(),
            max_length=64,
        )

    assert caught.value.code == "assistant_tokens_truncated"
    assert "private answer" not in caught.value.safe_message
