# Hermes Auxiliary Brain - Working Plan

Last updated: 2026-07-16

This file is the rude-raccoon recovery point. It records what we are building,
why the design looks this way, and what has actually been completed.

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
  - Deliberately withheld `/brain` until Hermes has an authenticated, busy-safe
    dynamic plugin command path; no private host monkey-patching was added.

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

- [ ] **8. Publish the Sparta build**
  - Review diff and confirm no secrets or personal details are present.
  - Commit directly to `main`.
  - Create public `kortexa-ai/hermes-auxiliary-brain` with `origin` configured.
  - Push `main`, verify the remote tree and default branch.

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
- An authenticated, busy-safe dynamic plugin slash-command path, followed by
  read/run-only `/brain` commands. Setup, mode, export, and evaluation stay CLI-only.
- Training orchestration and adapter registry; v0.1 only produces curated data
  and evaluation artifacts.
- Extra task packs contributed as data rather than new core code.
- Optional local embeddings or semantic retrieval after the basic classifier
  and extraction loop proves useful.

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
