# Local LoRA training

Hermes Auxiliary Brain can turn reviewed corrections into a small LFM2.5-230M
LoRA adapter without putting PyTorch, Transformers, or TRL into Hermes' runtime
environment. Every state-changing step is an explicit local CLI command. There
is no gateway or dashboard training endpoint, scheduled training, automatic
promotion, or cloud fallback.

## Before you start

Python 3.11 or newer is required to create the isolated training environments.
The default gate needs at least 20 unique corrected examples, split into at
least 16 training and 4 holdout rows. A promotable bundle must also contain at
least one unique training row and one unique holdout row for every built-in
task: `route`, `progress_checkin`, `follow_up`, `research_note`, and
`generic_extract`. More varied, carefully reviewed examples are much more
useful than repeated near-duplicates. Normalized duplicate inputs are kept
together so the same text cannot leak across the deterministic split, so one
example per task is not enough to guarantee both sides of the split.

Training files live under the active profile at
`HERMES_HOME/auxiliary-brain/training/`. They can contain the original input and
the corrected answer in plain text. The bundle builder rejects common secret
patterns and requires explicit acknowledgement for unattributed gateway
captures, but those checks cannot prove that data is safe or
anonymous. Review the source corrections and protect the profile directory.
Fine-tuned adapters may memorize examples. Training does not upload the bundle
or push artifacts to a model hub, but environment and model installation do
contact Python package indexes, Hugging Face, and the pinned llama.cpp source
host. Do not commit or casually share profile bundles, logs, runs, or adapters.
The managed API is keyless loopback in v0.5, so other OS users on a shared host
may be able to query it; do not train private material on a shared machine.

The native and GGUF checkpoints use Liquid AI's
[LFM Open License v1.0](https://www.liquid.ai/lfm-open-license); review its
terms before distributing a model or derivative. The plugin itself and
[llama.cpp](https://github.com/ggml-org/llama.cpp) are MIT licensed; installed
Python dependencies retain their own licenses.

## Hardware and storage

An NVIDIA GPU is the happy path. The installer detects `nvidia-smi` and uses
the official CUDA PyTorch wheel; Apple Silicon uses MPS when the installed
PyTorch build exposes it. CPU training is deliberately opt-in with
`--allow-cpu` and may be very slow.

The low-memory defaults are batch size 1, 512 tokens, gradient accumulation 4,
rank 8 over LFM2's query/key/value projections, no packing, and no
bitsandbytes or FlashAttention. Training and llama.cpp evaluation run as
separate subprocesses and are not loaded into Hermes. On a small machine, stop
other local-model applications before training and run stages sequentially.

Allow roughly 6-8 GB of free disk for two isolated environments, package/model
caches, the native checkpoint, the pinned llama.cpp source, and the Q4 base.
The exact total varies by platform and PyTorch wheel. The adapter itself is
small. Caches remain profile-local so later runs do not redownload everything.

## 1. Review corrections and check readiness

Record complete corrected JSON objects with `hermes brain correct`, then run:

```console
hermes brain train status
hermes brain train status --json
```

Status is read-only. It reports eligible corrected examples, split size,
isolated environments, pinned model assets, the latest run, and the active
deployment.

Explicit task commands create predictions for the four extraction tasks.
`route` predictions come from ordinary Hermes turns while the brain is in
`shadow` or `assist` mode. To gather and review them without changing normal
answers, use shadow mode, interact with Hermes normally, export the captured
rows to find their prediction IDs, and correct the useful examples:

```console
hermes brain mode shadow
hermes brain export route-review.jsonl --task route --include-uncorrected
hermes brain correct <prediction-id> --file corrected-route.json --note "reviewed"
```

Assist mode records the same route decision and may also run the selected
extraction task. Switch back to `explicit` or `off` when collection is done.

## 2. Freeze a training bundle

```console
hermes brain train prepare
```

The bundle contains the exact current task system/user messages and canonical
corrected JSON assistant response. Its identity is derived from its contents;
running the same preparation again reuses the same immutable bundle. Contract
hashes prevent corrections from an older prompt/schema from silently entering
a current bundle.

If reviewed rows came from a gateway path whose plugin hook did not supply a
sender ID, inspect them, then acknowledge that missing attribution explicitly:

```console
hermes brain train prepare --acknowledge-unattributed-gateway
```

For plumbing tests with too few rows, create an experimental bundle:

```console
hermes brain train prepare --allow-small
```

To focus a plumbing test on one task, combine `--task` with `--allow-small`:

```console
hermes brain train prepare --task generic_extract --allow-small
```

Selecting one task omits the other built-in contracts, so it cannot satisfy
the all-task promotion gate. `--allow-small` converts missing minimum or
per-task coverage findings into warnings. Lowering any of `--min-examples`,
`--min-train`, or `--min-holdout` below its default also permanently marks the
bundle experimental, even when `--allow-small` was not needed for another
finding. An experimental bundle and every `--smoke` run are non-promotable,
even if their toy evaluation passes.

## 3. Install isolated dependencies

```console
hermes brain train install all
```

You can install stages separately to spread out downloads:

```console
hermes brain train install trainer
hermes brain train install converter
```

The environments and reproducibility manifests are stored under the active
profile. The trainer pins the native model revision and ML packages; the
converter downloads and checksum-verifies the exact llama.cpp source used for
GGUF conversion. No package is installed into Hermes or the plugin environment.

## 4. Train and convert

```console
hermes brain train run
hermes brain train convert <run-id>
```

The first command prints a run ID. If no bundle or run is named, the newest one
is selected; naming IDs explicitly is safer in scripts. Training
refuses CPU execution unless `--allow-cpu` is present. It also validates the
assistant token mask after truncation and refuses a row unless the complete
corrected answer survives the token window.

For a cheap end-to-end infrastructure proof:

```console
hermes brain train run --smoke
hermes brain train convert <run-id>
hermes brain train evaluate <run-id>
```

Smoke mode defaults to two optimizer steps, batch size 1, and at most 512
tokens. It proves that train, conversion, adapter loading, and structured
generation work; it does not prove useful learning and can never be promoted.
An incomplete smoke holdout may fail the all-task or no-regression quality gate,
so `train evaluate` can print a valid report and exit with status 1. That is an
expected smoke-test result rather than a broken evaluation service; inspect the
report and stage logs to distinguish a quality failure from an infrastructure
error.

## 5. Evaluate and promote

```console
hermes brain train evaluate <run-id>
hermes brain train promote <run-id>
```

Evaluation downloads the checksum-pinned Q4 base when needed, launches an
ephemeral loopback llama.cpp server, loads the candidate adapter without
applying it globally, and evaluates baseline scale 0 versus candidate scale 1
on a deterministic sample of at most 100 rows from the same frozen holdout. It
still validates and binds the full holdout file and preserves at least one row
per represented task in the sample. Each generation has a 30-second request
timeout and the evaluation stage has a fixed 30-minute ceiling. It then shuts
the server down.

A candidate is promotion-eligible only when it is not experimental, the frozen
holdout covers every built-in task, every candidate response satisfies its task
schema, and exact/field accuracy does not regress overall or for any task.
Promotion still requires this explicit human command.

The active deployment is a small atomic pointer to verified artifacts. If the
managed llama.cpp server is already running, promotion restarts it with the new
adapter; if it is stopped, the next `hermes brain server start` loads the active
adapter. A failed restart restores the prior pointer and attempts to restore the
prior server.

## 6. Roll back

```console
hermes brain train rollback
```

Rollback selects the previous verified adapter from bounded history, or the
unchanged Q4 base when there is no earlier adapter. Its managed-server restart
uses the same transactional restore behavior as promotion.

## Logs and recovery

```console
hermes brain train logs <run-id> --stage trainer --lines 100
hermes brain train logs <run-id> --stage converter --lines 100
hermes brain train logs <run-id> --stage evaluation_server --lines 100
```

Run records preserve failure states and point to bounded stage logs. Fix the
reported problem and rerun the failed stage; completed bundles and artifacts
are not silently overwritten. `hermes brain train status --json` is the
machine-readable view for local scripts.

Trainer, converter, and evaluation children are killed with the parent on
Windows and Linux. macOS has no equivalent kernel parent-death primitive in
this release; after force-killing Hermes there, check for a surviving stage
process before retrying. Normal command failure and interruption still stop the
child explicitly on every platform.

An ordinary failed promotion restores the prior pointer and server. If the
machine loses power or Hermes is force-killed during that short transaction,
run `hermes brain server stop` followed by `hermes brain server start` to
reconcile the live server with the durable deployment pointer.

## Reproducibility pins

The implementation records exact revisions and hashes in bundle, environment,
run, evaluation, and deployment metadata. The v0.5 defaults are:

| Component | Pin |
| --- | --- |
| Native model | `LiquidAI/LFM2.5-230M` at `37b30cce3446f3f2e26a0d3f8c67c9167f5079d7` |
| Evaluation base | `LFM2.5-230M-Q4_K_M.gguf` at GGUF revision `fa224d4cb60cffe61eb58726712ef255bb64d0b7`, SHA-256 `7bbd90384d3deffe4c646ec9643b212802d32d4ce417c90a1ec9282100650062` |
| llama.cpp | `b10046` (`32e789fdfd598e9a1872da55ac941e4d94f030bd`), source archive SHA-256 `0c6608b4382c8056f4c398b57a801abe090a056d4160e7c4f90af9536b0c5745` |
| Trainer | PyTorch 2.13.0, Transformers 5.2.0, TRL 1.8.0, PEFT 0.19.1, Accelerate 1.14.0, Datasets 5.0.0, Safetensors 0.8.0 |

These pins are intentionally code-owned rather than user-facing environment
variables. Updating them is a reviewed plugin change with a new validation
cycle, not an invisible local drift.
