# Hermes Auxiliary Brain

A small, local-model sidecar for [Hermes Agent](https://github.com/NousResearch/hermes-agent).
It gives Hermes a fast private-ish reflex for repeatable classification and
extraction jobs while the main cloud model keeps the difficult reasoning,
conversation, tool use, and safety-sensitive decisions.

The first target is
[LiquidAI LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M), but the
plugin talks to a normal OpenAI-compatible API and is not coupled to one model
or server. The tiny model is a reflex, not the philosopher king. Giving a
230-million-parameter model the nuclear codes remains outside the roadmap.

> **Status:** early alpha. The plugin is useful, but its data and command
> contracts may still change before 1.0.

## What it does

- Runs explicit `hermes brain` jobs without calling the main cloud model.
- Extracts structured progress check-ins, follow-ups, and research notes.
- Provides a generic extraction path for other small, deterministic jobs.
- Can shadow normal turns to collect local predictions without changing the
  answer.
- Can optionally add compact local context to the *current* user turn. It does
  not mutate the system prompt or old conversation history.
- Stores predictions and human corrections locally in SQLite.
- Exports reviewed examples for deliberate evaluation and fine-tuning.
- Fails open: if the local server is asleep, ordinary Hermes conversations keep
  working.

It does **not** silently replace Hermes' main model, add a model tool to every
API request, take real-world actions, or fine-tune itself in the night while
making ominous GPU noises.

## Install

Hermes' native plugin installer is the recommended path:

```console
hermes plugins install kortexa-ai/hermes-auxiliary-brain --enable
```

Use a fresh Hermes process and let the plugin install, start, and configure its
default local server:

```console
hermes brain server start
hermes brain doctor
```

The first start downloads the checksum-pinned llama.cpp CPU build for your
platform, then llama.cpp downloads `LiquidAI/LFM2.5-230M-GGUF:Q4_K_M`. Both
the runtime and model cache stay under the active Hermes profile. The server
binds to `127.0.0.1:8080`; the plugin saves that endpoint only after the server
is ready and reports the exact requested model.

Useful lifecycle commands:

```console
hermes brain server status
hermes brain server logs --lines 100
hermes brain server stop
hermes brain server install --force
```

Restart a running messaging gateway after installation if you enable the
optional shadow or assist hook:

```console
hermes gateway restart
```

### Install from a development checkout

The standard-library-only bootstrap copies just the runtime files into the
active Hermes profile and enables the plugin without granting tool-override
permission:

```console
git clone https://github.com/kortexa-ai/hermes-auxiliary-brain.git
cd hermes-auxiliary-brain
python install.py --dry-run
python install.py
```

Useful options:

```console
python install.py --hermes-home /path/to/profile
python install.py --force
python install.py --no-enable
```

`--hermes-home` is especially useful with named Hermes profiles. Otherwise the
installer honors `HERMES_HOME` and then uses Hermes' platform default
(`%LOCALAPPDATA%\hermes` on Windows, `~/.hermes` elsewhere).

## Start a local model

`hermes brain server start` is the simplest path. If you already use another
local server, the plugin can discover these loopback endpoints instead:

The plugin probes these loopback endpoints, in order:

| Server | OpenAI-compatible base URL |
| --- | --- |
| LM Studio | `http://127.0.0.1:1234/v1` |
| llama.cpp | `http://127.0.0.1:8080/v1` |
| Ollama | `http://127.0.0.1:11434/v1` |
| vLLM | `http://127.0.0.1:8000/v1` |

Here are small starting points. The model is downloaded separately and keeps
its own license; it is not bundled with this MIT-licensed plugin.

### llama.cpp

The official
[GGUF repository](https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF) can be
served directly:

```console
llama serve -hf LiquidAI/LFM2.5-230M-GGUF:Q4_K_M
```

On installations that still expose the older executable name:

```console
llama-server -hf LiquidAI/LFM2.5-230M-GGUF:Q4_K_M --port 8080
```

### Ollama

Recent Ollama releases can run the Hugging Face GGUF directly:

```console
ollama run hf.co/LiquidAI/LFM2.5-230M-GGUF:Q4_K_M
```

Keep Ollama running; its OpenAI-compatible route is on port `11434`.

### LM Studio

Download or import `LiquidAI/LFM2.5-230M-GGUF`, load it, then enable the local
server from LM Studio's Developer view. The plugin reads the loaded model ID
from `/v1/models`, so it does not need to guess LM Studio's alias.

### vLLM

For the native checkpoint:

```console
vllm serve LiquidAI/LFM2.5-230M
```

Use another OpenAI-compatible server or model if you prefer; pass its base URL
and model to `hermes brain setup` instead of using auto-discovery. The plugin
accepts only `localhost` or loopback IP addresses; a server on another machine
is deliberately outside the current trust boundary.

Local/keyless is the default. If an endpoint requires a bearer token, put the
optional credential in the active Hermes profile's `.env`, never in
`config.yaml`:

```dotenv
AUXILIARY_BRAIN_API_KEY=replace-with-your-token
```

When that variable is set, setup requires an explicit loopback URL:

```console
hermes brain setup --base-url http://127.0.0.1:1234/v1 --model your-model-id
```

Authenticated auto-discovery is intentionally rejected so the bearer token is
not offered to unrelated services listening on other local ports. Keyless
`--auto` discovery remains the easy path.

## Use it

The `hermes brain` command tree is the strict no-cloud boundary:

```console
hermes brain run progress_checkin "Completed a 30-minute practice session."
hermes brain run follow_up "Send the revised draft by Friday."
hermes brain run research_note "Record this claim and what source should verify it."
hermes brain run generic_extract "Pull the people, dates, claims, and open questions."
hermes brain status
hermes brain doctor
```

Corrections accept a complete JSON object. `--file` avoids shell-quoting
acrobatics, especially on Windows:

```console
hermes brain correct <prediction-id> --file corrected.json --note "reviewed"
```

Built-in runtime task keys are `progress_checkin`, `follow_up`,
`research_note`, and `generic_extract`; `route` is the conservative classifier
used by opt-in shadow/assist behavior. Each task has a strict JSON contract and
treats its input as data, not instructions.

Use either `hermes brain --help` or `hermes brain help` for the command catalog
and copy-paste examples installed by your version.

### Status and diagnosis

`status` is the quick snapshot. It shows the plugin/profile, effective
configuration, configured and live model identity, managed-versus-external
server ownership, PID/binary/build/log paths, credential presence without its
value, storage path, and record counts:

```console
hermes brain status
hermes brain status --json
```

`doctor` refreshes the live probe and runs named `PASS`, `WARN`, and `FAIL`
checks. Failures include a concrete fix and produce a non-zero exit code. JSON
uses the same report model as the human output:

```console
hermes brain doctor
hermes brain doctor --json
```

### Authenticated status and check-in API

When `hermes dashboard` or the headless `hermes serve` backend is running, the
enabled plugin exposes two host-authenticated routes:

```text
GET  /api/plugins/auxiliary-brain/status
POST /api/plugins/auxiliary-brain/checkin
```

They inherit Hermes' dashboard authentication. On the normal loopback server,
use the same rotating session token as the dashboard. Hermes injects it into
the loaded page as `window.__HERMES_SESSION_TOKEN__`; it is intentionally not
printed. You can copy it from browser developer tools for local scripting:

```console
curl -H "X-Hermes-Session-Token: $TOKEN" \
  http://127.0.0.1:9119/api/plugins/auxiliary-brain/status

curl -X POST \
  -H "X-Hermes-Session-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"Completed a planned practice session."}' \
  http://127.0.0.1:9119/api/plugins/auxiliary-brain/checkin
```

Headless `hermes serve` has no page from which to read its random token. For a
loopback automation process, set a high-entropy `HERMES_DASHBOARD_SESSION_TOKEN`
secret before starting `serve`, and use that same value as `$TOKEN`. On a
non-loopback bind, Hermes' OAuth/password session gate is authoritative; the
loopback session header intentionally does not bypass it.

The check-in route accepts exactly one nonblank `text` field (maximum 8,000
characters), always runs the fixed `progress_checkin` task, and permits no
task, endpoint, model, process-control, or capture override. Its response omits
raw model text and endpoint details. Requests without valid Hermes dashboard
authentication receive `401`; the plugin has no second home-grown auth header.

These routes are unavailable in a messaging-gateway-only process. Remote
service tokens and generic run/correct APIs need a separate durable
authentication and abuse-boundary contract and remain deferred.

### Why there is no `/brain` slash command yet

Hermes supports plugin slash commands, but the released gateway busy-session
path can still treat a dynamic command received during an in-flight turn as
ordinary follow-up text and send it to the main model. The generic host fix is
tracked by [issue #58559](https://github.com/NousResearch/hermes-agent/issues/58559)
and [draft PR #58591](https://github.com/NousResearch/hermes-agent/pull/58591).
That gap makes a slash-command privacy promise impossible to keep today.

The plugin therefore fails closed and does not register `/brain` yet. The
strict local path is `hermes brain ...`. Slash read/run commands can be enabled
in a later release after Hermes provides an authenticated, busy-safe dynamic
command path. This is a host integration limitation, not a tiny-brain tantrum.

## Operating modes

| Mode | Behavior |
| --- | --- |
| `off` | No local inference. Stored data remains available to inspect/export. |
| `explicit` | Default. Run only when you invoke `hermes brain`. |
| `shadow` | Also classify normal turns locally and record the prediction; never alter the answer. |
| `assist` | Also attach a small, bounded local hint to the current user turn before the main model runs. |

`shadow` and `assist` still use the main model for the ordinary conversation.
They are learning/integration modes, not cloud-bypass modes. If local inference
fails, the hook returns no hint and Hermes continues normally.

Change mode without contacting the local server, for example:

```console
hermes brain mode shadow
hermes brain mode off
```

Rerunning setup preserves the current mode and capture setting unless you pass
`--mode`, `--capture`, or `--no-capture` explicitly.

## How learning works

The plugin separates **capturing experience** from **changing weights**:

1. A local task produces a structured prediction.
2. The prediction and task metadata are stored under
   `HERMES_HOME/auxiliary-brain/`.
3. You inspect it and record a correction when needed.
4. The plugin exports only reviewable, versioned learning examples.
5. You fine-tune a candidate model or adapter in an explicit external job.
6. You evaluate the candidate against held-out examples before changing the
   model served at the local endpoint.

This repository intentionally does not auto-train or auto-promote weights.
One bad correction should become one fixable row, not a personality transplant.
An exported dataset may contain private text; inspect and protect it like the
source material.

`hermes brain evaluate` is a regression check against corrected rows already
in the profile database. Those rows are not automatically held out from
training, so the command is not a promotion gate; manage a separate frozen
holdout in the external training workflow.

Corrections reference prediction IDs. When a prediction has several
corrections, its newest correction wins in the export. Corrected-only export is
the safe default; each JSONL row retains the task, input, selected output,
metadata, model, and provenance IDs needed for an auditable training pipeline.
Rows also carry a dataset-format version, plugin version, and task-contract hash
so a later prompt/schema change cannot silently masquerade as the same dataset.

## Architecture

```text
explicit hermes brain command ────────► local OpenAI-compatible server
        │                                          │
        └──────────── result + correction ◄────────┘
                             │
                             ▼
               HERMES_HOME/auxiliary-brain/brain.db

normal Hermes turn ───────────────────────────────► main cloud model
        │                                                ▲
        └─ optional shadow/assist pass ─► local model ────┘ compact hint only
```

The repository is a standalone user plugin. It registers existing Hermes
extension surfaces (a CLI command, lifecycle hook, and auxiliary-model task)
and does not modify `hermes-agent` core files.
The auxiliary task contributes the normal Hermes model-picker/config location,
but the plugin deliberately accepts only its `custom` local endpoint/model/timeout
fields. Inference uses the plugin's proxy-free, redirect-free loopback client;
provider fallbacks and cloud credential profiles are never consulted.
The longer [feasibility review](docs/feasibility.md) explains why the first
release uses these surfaces instead of a transparent mid-conversation router.

## Privacy and safety

- Keep unauthenticated local servers bound to `127.0.0.1`, not `0.0.0.0`.
- The client rejects non-loopback URLs and user-info URLs, ignores environment
  HTTP proxies, and refuses redirects so an explicit `hermes brain` request cannot
  wander off-machine wearing a fake moustache.
- “Local” describes the endpoint, not necessarily the server's logging policy.
  Review the server you run.
- Captured inputs, predictions, corrections, and exports can be sensitive.
  They stay outside the replaceable plugin code directory, but they are still
  files on your machine and should inherit appropriate disk protections.
- Storage is profile-wide, not per gateway user. On a multi-user gateway,
  shadow/assist capture writes all participating users' examples into the same
  profile database; leave those modes off unless that policy is appropriate.
- Small-model output is untrusted data. The plugin does not give it tools or
  autonomous authority.
- The research-note task organizes claims and questions; it does not verify
  them or make decisions on the user's behalf.

## Versioning and releases

Plugin releases follow Semantic Versioning:

- patch: compatible fixes and hardening;
- minor: backward-compatible capabilities;
- major: incompatible command, configuration, or data-contract changes.

`auxiliary_brain/version.py` is the canonical Python version. The static
`plugin.yaml` version must match it, and CI enforces that parity. Releases are
tagged `vMAJOR.MINOR.PATCH` and published through GitHub Releases.

Hermes currently installs `owner/repo` from that repository's default branch,
so `hermes plugins update auxiliary-brain` follows the latest `main`. For a
reproducible pinned installation, clone a tag and run the checkout installer:

```console
git clone --branch v0.3.0 --depth 1 https://github.com/kortexa-ai/hermes-auxiliary-brain.git
cd hermes-auxiliary-brain
python install.py
```

Database schema, exported-dataset format, and task-contract hashes are
versioned independently. A plugin release does not imply a database migration
or silently make old training examples compatible with a changed prompt.

## Update or remove

```console
hermes plugins disable auxiliary-brain
hermes plugins remove auxiliary-brain
```

For a native Git installation, update with:

```console
hermes plugins update auxiliary-brain
```

For a development-checkout installation, rerun the checkout installer:

```console
python install.py --force
```

Removing the plugin code intentionally leaves
`HERMES_HOME/auxiliary-brain/` alone. Delete or archive that data separately
only when you mean to.

## Develop

The runtime has no third-party Python dependency. Tests use `pytest`:

```console
python -m pytest
python -m compileall -q auxiliary_brain __init__.py install.py
```

For the plugin-discovery integration test, use the Python environment from an
adjacent `hermes-agent` checkout. See [plan.md](plan.md) for the current design
decisions and deferred work.

## License

Plugin code is available under the [MIT License](LICENSE). Models and local
inference servers have their own licenses; review those before use or
distribution.
