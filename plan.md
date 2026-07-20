# Hermes Auxiliary Brain - Working Plan

Last updated: 2026-07-20

This file is the rude-raccoon recovery point. It records what we are building,
why the design looks this way, and what has actually been completed.

## Resume here

- `v0.5.0` is the current release line. The annotated `v0.5.0` tag is the
  reproducible code checkpoint; `main` remains the development branch used by
  Hermes' native Git installer.
- The complete explicit LoRA lifecycle is implemented: prepare, install, run,
  convert, evaluate, promote, rollback, status, and bounded logs.
- A consistent SQLite backup helper and machine-migration procedure preserve
  `brain.db` while platform-specific runtimes and ML environments are rebuilt.
- The synthetic two-step hardware smoke proves the infrastructure but is
  permanently non-promotable. The next learning milestone is a fresh,
  meaningful corrected-data cycle on the intended training machine.
- Before changing code, run the validation commands in `AGENTS.md` and inspect
  the latest GitHub CI result. Resolve the release checkpoint with
  `git rev-parse v0.5.0^{commit}` instead of copying a possibly stale SHA here.

## Outcome

Build a standalone Hermes plugin that gives Hermes a small, local auxiliary
model for fast, private, repeatable tasks while leaving the cloud model in
charge of reasoning, tool use, safety-sensitive decisions, and conversation.

The first target is LiquidAI LFM2.5-230M behind any OpenAI-compatible local
server. The runtime is model-agnostic so another small local model can be used
without changing the plugin.

## Design decisions

- Repository: `kortexa-ai/hermes-auxiliary-brain`.
- Plugin id: `auxiliary-brain`; user-facing command: `hermes brain`.
- Standalone user plugin only. No changes to `hermes-agent` for the first release.
- Use Hermes' plugin APIs: CLI command, lifecycle hook, and a plugin-defined
  auxiliary-model task. Do not add a permanent model tool.
- Explicit `hermes brain` commands are the reliable local fast path. A transparent
  natural-language turn router is intentionally deferred because Hermes does
  not currently expose a generic plugin turn-routing contract.
- The messaging `/brain` surface is a narrow, single-profile opt-in. Current
  Hermes versions dispatch it correctly while idle but can misroute it during
  an active turn, so enablement requires an explicit risk acknowledgement and
  the documented rule is to invoke it only between turns. Multiplex gateways
  fail closed until Hermes gives plugin handlers a reliable event-profile scope.
- Shadow/enrichment hooks are opt-in and fail open: if the local model is down,
  normal Hermes operation continues unchanged.
- Keep the system prompt byte-stable. Any optional enrichment is injected into
  the current user turn through Hermes' `pre_llm_call` hook.
- Store observations, predictions, corrections, and exports under
  `HERMES_HOME/auxiliary-brain/`, outside the replaceable plugin installation.
- Never fine-tune automatically. The plugin gathers reviewable examples,
  exports a versioned dataset, and requires an explicit train/evaluate/promote
  workflow before an adapter may be used.
- High-stakes text is limited to research/note extraction, not autonomous
  action or personalized medical, legal, or financial advice.

## Initial task catalog

1. `route` - decide whether a request is suitable for the local brain or needs
   the main model.
2. `progress_checkin` - turn a generic progress update into structured facts and a
   small next action.
3. `follow_up` - extract commitments, due hints, and next actions.
4. `research_note` - extract claims, dates, entities, sources to
   verify, and open questions from research notes.
5. `generic_extract` - general schema-light structured extraction for other small jobs.

## Work plan

- [x] **1. Verify feasibility and host contracts**
  - Confirmed standalone plugins can register CLI/slash commands, hooks, and
    auxiliary tasks.
  - Confirmed auxiliary tasks can select a local OpenAI-compatible endpoint
    independently of the main cloud model.
  - Confirmed prompt-safe `pre_llm_call` context is injected into the current
    user message rather than the system prompt.
  - Confirmed the target GitHub repository name is available and GitHub auth is
    active for the `kortexa-ai` organization.
  - Found a Hermes busy-session edge case that can misroute dynamic plugin
    slash commands as ordinary cloud-bound follow-up text. Version 0.1 fails
    closed by exposing the strict local path only as `hermes brain`.

- [x] **2. Bootstrap the peer repository**
  - Created `C:\src\hermes-auxiliary-brain`.
  - Initialized Git with `main` as the initial branch.
  - Added this recovery plan before the laptop can wander into the forest.

- [x] **3. Implement the local-brain runtime**
  - Added endpoint discovery and health checks for common OpenAI-compatible servers.
  - Added strict task prompts, schema validation, and one bounded repair attempt.
  - Added a profile-local SQLite event/prediction/correction store.
  - Added corrected-example export and current-model evaluation.

- [x] **4. Wire the Hermes plugin surfaces**
  - Added `plugin.yaml` and the standalone `register(ctx)` entry point.
  - Added `hermes brain setup|doctor|status|tasks|mode|run|correct|export|evaluate`.
  - Registered a plugin-defined auxiliary task for model-picker/config integration.
  - Added an optional synchronous shadow/assist hook, disabled in explicit mode.
  - Deliberately withheld `/brain` from v0.1 while its gateway safety contract
    was investigated; no private host monkey-patching was added.

- [x] **5. Make installation and configuration boring**
  - Added a cross-platform, standard-library-only installer.
  - Added safe staged copy into the active profile and explicit enablement.
  - Added automatic endpoint probing plus explicit endpoint/model flags.
  - Added after-install, update, uninstall, privacy, and recovery instructions.

- [x] **6. Preserve and sanitize the feasibility record**
  - Converted the earlier Desktop DOCX to `docs/feasibility.md`.
  - Removed personal examples and replaced them with generic progress and
    research-workflow examples.
  - Kept the architecture conclusions, code evidence, and source links intact.

- [x] **7. Test the whole path**
  - 110 unit and integration tests pass.
  - Real Hermes discovery/registration tests pass against the adjacent checkout.
  - A real loopback E2E test crosses the Hermes CLI handler, GET `/v1/models`,
    POST `/v1/chat/completions`, schema validation, and SQLite persistence.
  - Model-ID mismatch fails before completion or database writes.
  - Installer dry-run and install/enable paths pass against temporary profiles.

- [x] **8. Publish the Sparta build**
  - Reviewed the complete diff and confirmed no secrets or personal tracking
    details are present.
  - Committed directly to `main` under the agreed Sparta workflow.
  - Created public `kortexa-ai/hermes-auxiliary-brain` and configured `origin`.
  - Pushed `main`; GitHub reports it as the public default branch.

- [x] **9. Validate the published artifact**
  - Installed `kortexa-ai/hermes-auxiliary-brain --enable` through Hermes' native
    Git installer into an isolated profile.
  - Loaded the installed `hermes brain --help` command tree from that profile.
  - GitHub Actions passed tests on Python 3.11, 3.12, and 3.13 plus Ruff lint
    and format checks.

- [x] **10. Ship the one-command local server (`v0.2.0`)**
  - [x] Implement a profile-local llama.cpp manager for Windows, macOS, and
    Ubuntu Linux on x64 and arm64.
  - [x] Pin llama.cpp `b10046`; verify the selected release asset's size and
    SHA-256 before safe extraction.
  - [x] Default to `LiquidAI/LFM2.5-230M-GGUF:Q4_K_M` on `127.0.0.1:8080`.
  - [x] Reuse an installed `llama`/`llama-server`, otherwise install the CPU
    runtime under the active Hermes profile.
  - [x] Keep the Hugging Face model cache, PID metadata, and logs profile-local;
    verify process identity before stop and reject non-loopback binds.
  - [x] Exercise the real pinned Windows asset and a real LFM download,
    start, health check, and stop lifecycle.
  - [x] Wire `hermes brain server install|start|status|stop` into the plugin CLI.
  - [x] Configure the auxiliary-brain endpoint only after the server is ready
    and reports the requested model.
  - [x] Finish operator documentation, the complete test pass, and an isolated
    install-from-GitHub smoke test.
  - [x] Tag and publish `v0.1.0`, then tag and publish `v0.2.0` with release notes.

- [x] **11. Make operation and diagnosis obvious (`v0.3.0`)**
  - Keep `status` as the quick read-only snapshot. Expand it to show the plugin
    version, active profile, effective configuration, configured endpoint and
    model, managed-versus-external server ownership, PID/binary/build/port,
    live health and model identity, auth presence without secrets, storage
    location, and record counts.
  - Keep `doctor` as the deeper live check. Report named `PASS`, `WARN`, and
    `FAIL` checks with concrete fixes and a non-zero exit code for failures.
    Include config parsing, loopback policy, endpoint reachability, exact model
    match, managed PID/image/port state, binary/cache/log paths, and writable
    data storage.
  - Add `--json` to `status` and `doctor`, generated from the same report model
    as human output so scripts and future APIs do not scrape terminal prose.
  - Add `hermes brain help` as an explicit alias for the existing
    `hermes brain --help`, with a few copy-paste examples and no model call.
  - Add `hermes brain server logs [--lines N]`; status and doctor should point
    to it when startup fails.
  - Do not add a second mutable `config` command merely to print settings.
    `status` owns inspection; `setup` and `mode` remain the mutation paths.
  - Expose authenticated, bounded dashboard-plugin routes for JSON `status`
    and a fixed-purpose `POST /checkin` route. Reuse the same report/runtime
    code, cap inputs, and allow no endpoint/model override. No unauthenticated
    route and no remote server/process control.
  - [x] Pass the Python 3.11/3.12/3.13 CI matrix and Ruff, then install from the
    public Git repository into an isolated profile and verify the CLI plus all
    dashboard API/entry artifacts.
  - [x] Tag and publish `v0.3.0` with release notes.

- [x] **12. Add opt-in remote gateway check-ins (`v0.4.0`)** - complete
  - Register one process-wide `/brain` handler, then require the active single
    profile to enable its idle-path capability with
    `hermes brain gateway enable --acknowledge-busy-risk`. Keep capability off
    by default and read `gateway status|enable|disable` on every invocation, so
    setting changes need no gateway restart after the plugin itself is loaded.
  - Limit the messaging surface to `/brain help`, sanitized `/brain status`,
    and four fixed local tasks: `checkin`, `followup`, `note`, and `extract`.
    Cap task input at 8,000 characters, treat blank `/brain` as help, and reject
    missing task text or unknown requests.
  - Do not expose setup, endpoint/model selection, server lifecycle, mode,
    corrections, export, evaluation, arbitrary task names, or training through
    the gateway. Return sanitized failures without credentials, endpoints,
    tracebacks, or raw server responses.
  - Document the current contract honestly: plugin commands dispatch correctly
    while the gateway is idle, but an invocation received during an active turn
    can still be treated as cloud-bound follow-up text before the plugin runs.
    Users must invoke `/brain` only between turns; queue, steer, and interrupt
    settings do not remove the risk.
  - Upstream tracks the generic busy-session fix in
    [issue #58559](https://github.com/NousResearch/hermes-agent/issues/58559) and
    [PR #58591](https://github.com/NousResearch/hermes-agent/pull/58591).
    Keep the opt-in acknowledgement until the released Hermes host provides
    and the plugin verifies that busy-safe contract; do not carry a private
    host monkey patch.
  - Reject `/brain` on multiplex-profile gateways until Hermes scopes every
    process-global plugin-command invocation to the routed event's profile.
    Resolve endpoint credentials through Hermes' active profile secret scope,
    and fail closed rather than falling back to another profile's environment.
  - [x] Pass the complete local suite against the adjacent Hermes checkout,
    including real idle gateway dispatch and a
    plugin-loader-to-loopback-model-to-SQLite slash-command E2E. Ruff,
    formatting, compilation, and diff checks pass; an isolated installed
    profile proves default-off, acknowledgement, registration, and disable.
  - [x] Push `main`, pass the remote Python 3.11/3.12/3.13 matrix, then tag and
    publish `v0.4.0` with release notes.

- [x] **13. Define programmatic access without publishing the local model** - design complete
  - Hermes already has an authenticated OpenAI-compatible `api_server` gateway
    for normal agent turns. Enabled plugins participate in those turns through
    their registered hooks and tools, but this is not a direct auxiliary-brain
    task API and normally still invokes the main model.
  - Hermes also supports authenticated dashboard-plugin backend routes under
    `/api/plugins/<name>/...`; this is the preferred direct JSON extension
    surface when `hermes dashboard` or `hermes serve` is running.
  - The same `hermes serve` backend already exposes authenticated JSON-RPC over
    WebSocket. `cli.exec` can run non-interactive `brain` CLI commands today;
    `command.dispatch` can invoke the opt-in plugin slash handler while the
    host is idle. Treat these as host APIs, not reasons to invent a second
    daemon in this plugin.
  - The authenticated status/check-in dashboard routes in step 11 ship in
    `v0.3.0` and need no Hermes core change. Document clearly that they are
    unavailable when only the messaging gateway is running.
  - Apart from the deliberately narrow shipped `POST /checkin` route, defer
    mutating/headless endpoints such as generic `POST /run`, `POST /correct`,
    or service bearer-token access until the contract includes strict body
    caps, fixed task names, no endpoint/model override, idempotency, rate
    limits, audit-safe errors, and an explicit authentication story.
  - Never expose llama.cpp directly to the internet. Remote clients talk to an
    authenticated Hermes surface; Hermes talks to the loopback model.

- [x] **14. Add an explicit training and promotion pipeline (`v0.5.0`)** - complete
  - **Released implementation:**
    - [x] Added dependency-free readiness inspection and content-addressed
      train/holdout bundles with exact task messages, current-contract checks,
      deterministic normalized-input grouping, secret lint, explicit review of
      unattributed gateway rows, and non-promotable small-dataset experiments.
    - [x] Added checksum-pinned local model and LoRA artifact support to the
      managed llama.cpp process state. Generic `--lora` overrides remain
      blocked; old state files remain readable.
    - [x] Added separate profile-local trainer and converter environment
      contracts so Torch/TRL never become Hermes runtime dependencies.
    - [x] Added low-memory TRL/PEFT orchestration, pinned llama.cpp source
      acquisition, PEFT-to-GGUF conversion, frozen-holdout Q4 evaluation,
      explicit promotion, transactional managed-server restart, and rollback.
    - [x] Added focused data, trainer-backend, orchestration, managed-adapter,
      and CLI lifecycle tests without importing ML packages into Hermes.
    - [x] Hardened artifact provenance, bounded reads/logs/reports, sanitized
      child environments, exact managed-server identity, Windows/Linux child
      lifetime, deterministic max-100 holdout evaluation, and transactional
      candidate cleanup. The official Linux llama.cpp symlink layout is
      preserved and verified without allowing links outside the runtime tree.
      Readiness and bundle creation now preflight and stream only the selected
      correction from one bounded SQLite snapshot, one row at a time.
    - [x] Finished the complete existing regression suite and two independent
      final reviews. The local suite reports 345 passed / 11 optional skips;
      all eight adjacent-Hermes integration tests pass, for 353 passing tests
      and three platform-only skips when that checkout is enabled. Ruff,
      formatting, compilation, and diff checks are clean.
    - [x] Installed the working tree into a fresh temporary profile without
      enabling it, then loaded version 0.5.0 and ran `train status --json` from
      the installed copy. This smoke did not download or load model weights.
    - [x] Passed the hard gate on this laptop: a two-step rank-8 CUDA LoRA must
      train, convert with llama.cpp `b10046`, load beside the exact pinned Q4
      base, and produce a schema-valid completion. Keep the smoke candidate
      permanently non-promotable regardless of its toy score.
      The live proof used the RTX 3070 Laptop GPU with batch 1 and 512 tokens;
      training took 1.4 seconds after model load, conversion produced a
      494,592-byte GGUF adapter, and all eight candidate holdout calls were
      schema-valid. The toy accuracy remained zero, as expected from two steps.
    - [ ] Run a meaningful corrected-data training/evaluation cycle as the
      first real learning milestone. This is deliberately separate from
      releasing the explicit, guarded training infrastructure in `v0.5.0`;
      no adapter may be promoted without its own passing evaluation.
      The cycle is complete only when a non-experimental bundle passes the
      default split and all-task gates, a non-smoke train/convert/evaluate run
      produces schema-valid outputs without overall or per-task regression,
      at least one predeclared aggregate metric improves, and promotion plus
      rollback are verified. Record only aggregate metrics, revisions, hashes,
      hardware, and timings; never commit private source rows or artifacts.
  - The current correction/export/evaluate loop is the data foundation, but
    the managed llama.cpp server is an inference runtime, not a trainer.
    Do not productize llama.cpp's own finetune example: upstream describes it
    as FP32-only on limited hardware and "very much WIP" in the
    [training README](https://github.com/ggml-org/llama.cpp/blob/b10046/examples/training/README.md).
  - Train from LiquidAI's native Hugging Face checkpoint with a separate,
    isolated Python environment. Start with supervised LoRA fine-tuning through
    a supported TRL or Unsloth path; do not update GGUF weights in place.
  - Upgrade the audit export into a training bundle whose rows contain the
    exact task system/user messages and corrected JSON assistant response.
    Permanently group normalized duplicate inputs into one side of a
    deterministic train/holdout split so evaluation cannot reuse training text.
  - Add dataset lint and minimum-example gates, task-contract pinning, privacy
    review, a reproducible training manifest, seed/dependency/model revisions,
    and resumable artifact metadata.
  - Convert the reviewed PEFT adapter to a llama.cpp-compatible GGUF LoRA (or
    merge and quantize only when required), then let the managed server load the
    candidate adapter alongside the unchanged base GGUF.
  - Require baseline-versus-candidate evaluation, per-task thresholds, explicit
    human promotion, rollback, and artifact provenance. Never train or promote
    merely because enough time or examples elapsed.
  - Add `hermes brain train status` and `hermes brain train prepare` here, not
    in `v0.3.0` or `v0.4.0`. `status` reports training readiness without
    changing weights; `prepare` lints reviewed examples and creates the
    deterministic TRL train/holdout bundle plus reproducibility manifest used
    by the trainer.
    Its privacy lint must flag unattributed `gateway-slash` rows for explicit
    review before they enter either split.
  - Support local GPU, Apple Silicon where the selected trainer supports it,
    and an explicit user-selected remote GPU runner later. CPU-only training
    may be allowed for experiments but must not be marketed as the happy path.
  - First implementation gate: prove one tiny native-checkpoint LoRA can train,
    convert, load, and answer through the pinned managed server before building
    the general orchestration UI. Relevant upstream contracts are LiquidAI's
    [LFM2.5-230M model card](https://huggingface.co/LiquidAI/LFM2.5-230M),
    [TRL guide](https://docs.liquid.ai/lfm/fine-tuning/trl), llama.cpp's
    [PEFT adapter converter](https://github.com/ggml-org/llama.cpp/blob/b10046/convert_lora_to_gguf.py),
    and its [`--lora` server support](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

## Release sizing after v0.2.0

| Capability | Size | Target | Reason |
| --- | --- | --- | --- |
| Rich `status`, actionable `doctor`, `--json` | Small | `v0.3.0` | Extends existing code and unlocks scripts/API reuse. |
| Explicit `hermes brain help` | Tiny | `v0.3.0` | Argparse help already exists; this improves discovery. |
| Managed server log tail | Small | `v0.3.0` | State/log ownership already exists in `v0.2.0`. |
| Authenticated dashboard status and fixed check-in API | Small-medium | `v0.3.0` | Supported plugin surface; reuse host auth, enforce caps, and prove packaging E2E. |
| Direct task/correction service API | Medium | Future | Needs a durable auth, idempotency, and abuse-boundary contract. |
| Opt-in messaging `/brain` | Small plugin change, documented host risk | `v0.4.0` | Useful while idle; default-off acknowledgement preserves an honest boundary until upstream #58591 lands. |
| Training readiness and deterministic bundle | Medium | `v0.5.0` | Ship with the trainer contract instead of freezing a premature bundle format. |
| LoRA train/convert/evaluate/promote/rollback | Large | `v0.5.0` | Separate ML environment, hardware, reproducibility, and rollback work. |

## Acceptance criteria for v0.1.0

- A user can clone the repository, run one installer command, start/restart
  Hermes, and run `hermes brain setup --auto`.
- `hermes brain run progress_checkin ...` and a general extraction task work through a
  local OpenAI-compatible endpoint without invoking the main cloud model.
- Normal Hermes operation is unaffected when the local server is unavailable.
- No plugin model tool is added to every Hermes API request.
- Data capture is local, inspectable, correctable, exportable, and off the
  automatic-training trigger path.
- Tests pass locally and in GitHub Actions.

## Deferred ideas

- A generic Hermes plugin turn-router contract for transparent fast-path turns.
- Extra task packs contributed as data rather than new core code.
- Optional local embeddings or semantic retrieval after the basic classifier
  and extraction loop proves useful.
- Automatic adapter selection per task; first prove one reviewed adapter can be
  trained, evaluated, promoted, and rolled back safely.
- Add authenticated llama.cpp requests (or another per-profile ownership
  proof) before treating loopback as private on shared-user hosts; randomize
  the ephemeral evaluation port as a smaller defense-in-depth step.
- Add a macOS parent-death watchdog for MPS jobs. Normal interruption is
  cleaned up now, but a force-killed parent can leave a child on Darwin.
- Compare the live managed base/adapter identity with the deployment pointer in
  status/doctor, then consider a two-phase deployment journal for abrupt-power
  recovery. The documented v0.5 recovery is server stop followed by start.
- Add explicit retention/prune commands for plaintext bundles, run logs,
  caches, and old adapters.

## Progress log

- 2026-07-16: Feasibility review complete; plugin-first route selected.
- 2026-07-16: Peer repository initialized on `main`; implementation split into
  runtime, installer/docs, integration, and test workstreams.
- 2026-07-16: Added this checkpoint document and started sanitizing the earlier
  feasibility document for the public repository.
- 2026-07-16: Completed the sanitized Markdown feasibility record at
  `docs/feasibility.md`; no personal tracking categories remain.
- 2026-07-16: Implemented the stdlib-only runtime, Hermes registration,
  commands, opt-in hook, installer, SQLite correction loop, export, and evaluation.
- 2026-07-16: Unit suite reached 62 passing tests with one optional real-Hermes
  integration test queued against the adjacent checkout.
- 2026-07-16: Hardened the client to loopback-only, proxy-free,
  redirect-free requests; added strict configured-model identity and non-finite
  JSON rejection.
- 2026-07-16: Encoded the full Hermes-to-loopback-to-SQLite path in the test
  suite; 110 tests pass.
- 2026-07-16: Removed the `/brain` slash surface after tracing a Hermes
  busy-session dispatch gap that could violate the no-cloud promise. The CLI
  remains the strict local/admin boundary; the limitation is documented.
- 2026-07-16: Made authenticated discovery fail closed: when a local endpoint
  token is configured, the operator must name the exact loopback base URL, so
  the token is never sprayed across unrelated local services.
- 2026-07-16: Published the initial build to
  `https://github.com/kortexa-ai/hermes-auxiliary-brain` on `main`.
- 2026-07-16: Reinstalled the public repository through Hermes' native plugin
  installer and confirmed the remote CI matrix is green. The tiny goblin ships.
- 2026-07-16: Started v0.2.0 work on a pinned, one-command llama.cpp + default
  LFM server lifecycle, plus a reproducible plugin version/tag story.
- 2026-07-16: Completed the cross-platform llama.cpp manager and 38 focused
  safety tests; a real pinned Windows binary and real LFM start/stop lifecycle
  passed. CLI wiring and release work remain for v0.2.0.
- 2026-07-16: Triaged diagnostics/help, remote gateway use, direct APIs, and
  training. Scoped the low-risk operator work to v0.3.0, recorded the Hermes
  busy-plugin-command dependency, and separated training into an explicit
  evaluate/promote/rollback workstream.
- 2026-07-16: Verified upstream issue #58559 and PR #58591 already cover
  the generic busy-session plugin/skill command gap, so no duplicate Hermes
  branch or PR was opened. Committed authenticated status/check-in API work to
  v0.3.0 and moved both training status and bundle preparation to v0.5.0.
- 2026-07-16: Completed v0.2.0: `hermes brain server start` now installs the
  pinned llama.cpp build when needed, waits for the default LFM model, verifies
  its identity, and only then saves configuration. All 155 tests passed, and a
  clean Hermes Git install exposed the full managed-server command tree.
- 2026-07-16: Completed the v0.3.0 implementation: rich secret-safe status and
  doctor reports, JSON output, explicit help, bounded server logs, and
  host-authenticated status/check-in dashboard routes. All 186 tests pass; the
  real-Hermes API test proves unauthenticated `401` and authenticated `200`, and
  an isolated install loads `brain help` plus `status --json`. Training
  readiness and bundle preparation remain deliberately grouped with the full
  v0.5.0 training pipeline.
- 2026-07-16: The v0.3.0 remote matrix passed on Python 3.11, 3.12, and 3.13
  with the dashboard router tests enabled. A fresh native Git installation
  loaded version 0.3.0 and contained the manifest, authenticated API module,
  and dashboard entry bundle. The release raccoon has been issued a helmet.
- 2026-07-16: Implemented the v0.4.0 default-off gateway surface with explicit
  busy-risk acknowledgement, local status/disable controls, live config gating,
  fixed bounded tasks, and sanitized replies. The upstream busy-session fix
  remains draft, so `/brain` is documented for between-turn use only. Training
  status, bundle preparation, and the adapter pipeline remain v0.5.0 work.
- 2026-07-16: Hardened gateway replies against endpoint-secret echo and
  oversized unknown actions, added profile-scoped secret resolution and
  multiplex fail-closed behavior, then passed the complete local suite plus a
  fresh-profile install/enable/register/disable smoke test. The remote Python
  3.11/3.12/3.13 matrix and lint job passed, the annotated `v0.4.0` tag and
  GitHub Release were published, and a fresh native Git install loaded the
  exact release commit, default-off gate, acknowledged enable, and `/brain`
  help. The release raccoon has traded its helmet for a tiny clipboard.
- 2026-07-17: Implemented the local v0.5 training lifecycle: deterministic
  privacy-gated bundles, isolated low-memory TRL/PEFT training, pinned
  PEFT-to-GGUF conversion, exact-Q4 baseline/candidate evaluation, explicit
  promotion, transactional managed-server restart, rollback, and bounded logs.
- 2026-07-17: Passed the real infrastructure smoke on the weak laptop. The
  first 256-token attempt correctly failed before training because truncation
  removed every assistant token; after measuring the 327-token examples, the
  default moved to 512 and the CUDA train/convert/load/schema-validation path
  passed. The synthetic candidate remains permanently non-promotable; a
  meaningful corrected-data cycle remains the next real learning milestone.
- 2026-07-17: Completed the memory-conscious v0.5 hardening pass without
  loading the model again: exact model/converter/runtime provenance, safe
  subprocess environments and parent-lifetime handling, bounded deterministic
  evaluation, immutable deployment validation, and fresh-profile packaging.
  The complete plugin plus adjacent-Hermes suite is green (353 passed, three
  platform-only skips), and independent security/final reviews found no
  remaining implementation blocker.
- 2026-07-20: Published the v0.5 implementation to `main`, fixed the Linux
  symlink-containment test so remote CI exercises the intended branch, and
  added a cold-clone development guide plus a consistent SQLite migration
  helper. Reframed the meaningful corrected-data cycle as the first real
  learning milestone rather than pretending the synthetic smoke was useful
  training. Prepared the annotated `v0.5.0` release and a clean macOS handoff.
- 2026-07-20: Published the annotated `v0.5.0` tag and
  [GitHub Release](https://github.com/kortexa-ai/hermes-auxiliary-brain/releases/tag/v0.5.0)
  from commit `30ffaf4`. The main and tag CI matrices passed on Python
  3.11/3.12/3.13; a clean Apple Silicon checkout passed all 361 tests, loaded
  the migrated SQLite state, started the pinned managed LFM server, and passed
  every doctor check. The training environments remain intentionally
  uninstalled until current-contract corrected data is ready.
