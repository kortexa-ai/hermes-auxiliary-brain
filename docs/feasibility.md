**HERMES AGENT + LOCAL LFM2.5**

Plugin-first feasibility review for a trainable local auxiliary layer

> **PUBLIC-REPO NOTE** This Markdown document was converted from the
> initial feasibility review. Personal examples and requested tracking
> categories were removed or generalized; the technical findings and
> architecture decision were retained.

> **IMPLEMENTATION UPDATE** A final E2E review found that current Hermes
> gateway busy-session handling recognizes built-in commands but not dynamic
> plugin commands. A plugin slash command received mid-turn can therefore be
> treated as ordinary cloud-bound follow-up text. Version 0.1 fails closed: the
> strict local/admin surface is `hermes brain`, and `/brain` is deferred until
> Hermes has an authenticated, busy-safe dynamic command path.

> **CURRENT V0.4 STATUS** The text below preserves the original v0.1
> feasibility snapshot. Version 0.4 adds a default-off `/brain` command for
> single-profile gateways after explicit busy-risk acknowledgement. It is safe
> only between turns on current Hermes and rejects multiplex-profile gateways;
> `README.md` is the authoritative current operating contract.

**Review date:** 16 July 2026

**Checkout:** C:\src\hermes-agent \| branch main \| 007cd151329c

**Scope:** Feasibility and architecture only; no Hermes source changes
made

**Decision status:** Plugin-first pilot recommended

> **RECOMMENDATION** Proceed with a user-installed standalone plugin for
> the first pilot. The current Hermes plugin and auxiliary-model
> surfaces are sufficient for explicit local check-ins, shadow
> evaluation, structured extraction, local ledgers, and the fine-tuning
> data loop. Do not change Hermes core until the pilot proves a need for
> transparent natural-language cloud bypass.

# 1. Executive conclusion

LFM2.5-230M can be integrated alongside a cloud model without turning it
into Hermes's main agent and without adding a permanent model tool.
Hermes already supports OpenAI-compatible local endpoints, plugin-owned
model calls, plugin auxiliary-task registration, lifecycle hooks, CLI
commands, and scheduled scripts. Together, these cover the
useful pilot surface.

A model-provider plugin is not required. Hermes's existing custom
provider profile already recognizes local, vLLM, llama.cpp, and
compatible endpoints. A model-provider plugin would describe a transport
or provider quirk; it would not by itself create a second-model
workflow.

The recommended boundary is simple: the plugin owns bounded task logic
and training data; the named custom endpoint owns LFM inference; the
main cloud model remains the reasoning and verification authority; Hermes memory
and skills remain inspectable truth. The tiny model gets a clipboard,
not the nuclear launch codes.

## Decision in one page

| **Question** | **Finding** | **Decision** |
|:---|:---|:---|
| Can Hermes call LFM locally? | Yes, through an existing named custom OpenAI-compatible endpoint. | No core change |
| Can a plugin invoke it? | Yes, with structured plugin LLM calls or a plugin-owned auxiliary task. | No core change |
| Can it process explicit check-ins without the cloud model? | Yes, through `hermes brain run progress_checkin`. | No core change |
| Can it enrich ordinary cloud turns? | Yes, through pre_llm_call context injection into the user turn. | No core change |
| Can it automatically answer arbitrary natural language without GPT across every UI? | Not through a clean cross-surface early-return contract today. | Possible later core seam |
| Can it train itself safely every turn? | Technically possible to collect data, but automatic online weight updates are the wrong safety model. | Offline, evaluated LoRA only |

# 2. Model fit and constraints

Liquid AI positions LFM2.5-230M as a 230-million-parameter, text-only
model for data extraction and lightweight agentic pipelines. The model
card reports a 32,768-token context window, a mid-2024 knowledge cutoff,
native and GGUF distributions, and explicit fine-tuning support. That
profile fits bounded classification and extraction; it does not fit
high-stakes synthesis, open-ended coaching, or Hermes conversation
compression.

- **Good first jobs:** progress check-in extraction, intent classification,
  reminder-response tagging, research-note metadata, title generation,
  and candidate escalation.

- **Keep in the cloud:** ambiguous planning, consequential research synthesis,
  health or financial judgment, memory/skill authorization, and any high-consequence
  recommendation.

- **Hard incompatibility:** Hermes enforces a 64K minimum context for
  the compression auxiliary task, while LFM2.5-230M declares 32K.

- **Fine-tuning posture:** begin with supervised LoRA adapters. Liquid
  recommends LoRA because it updates a small adapter rather than all
  model weights and supports separate task adapters.

> **WEIGHTS ARE NOT MEMORY** Rapidly changing facts, schedules, goals,
> sensitive records, and source dates belong in ledgers or
> Hermes memory. Fine-tuned weights should encode stable mappings such
> as how to classify a check-in, which fields to extract, and when
> uncertainty requires escalation.

# 3. What the current Hermes code already provides

| **Surface** | **Current code finding** | **LFM implication** |
|:---|:---|:---|
| Local provider | The custom provider profile aliases local, vLLM, llama.cpp, and related names; named custom providers resolve saved URLs and keyless local endpoints. | Serve LFM externally; no new provider class. |
| Plugin LLM facade | ctx.llm supports sync/async chat and structured JSON calls with provider/model trust gates and host-owned routing. | A plugin can call LFM without adding a model tool. |
| Plugin auxiliary tasks | register_auxiliary_task creates auxiliary.\<key\> routing, config-picker visibility, defaults, and gateway bridging. | LFM can have a dedicated side-task configuration. |
| Turn hooks | pre_llm_call may inject ephemeral user context; post_llm_call receives the user message, assistant response, and conversation history. | Local classification and dataset capture are plugin-feasible. |
| Slash/CLI commands | Plugins can add in-session slash commands and `hermes <subcommand>` command trees, but current gateway busy handling is unsafe for a strict plugin-command cloud-bypass promise. | Use the CLI command now; defer slash commands pending a generic Hermes fix. |
| Middleware | LLM execution middleware can wrap provider execution, but it operates on the already-selected main runtime and must return the response shape expected by that transport. | Do not use it as a brittle cross-provider router. |
| Background review | A different review model receives a compact digest without modifying the main conversation or cached prompt; the review may write memory and skills. | Keep the cloud model here until LFM is thoroughly proven. |

## Code evidence

- hermes_cli/plugins.py:339-365 - ctx.llm is the supported host-owned
  plugin model facade.

- hermes_cli/plugins.py:504-581 - plugins can register CLI and
  in-session slash commands; gateway busy handling still needs a generic fix
  before slash can be a strict local boundary.

- hermes_cli/plugins.py:1047-1156 - plugins can register named auxiliary
  LLM tasks.

- agent/plugin_llm.py:598-773 - structured/sync plugin completion
  implementation and trust checks.

- agent/auxiliary_client.py:6089-6209 and 6228-6269 - per-task provider
  resolution and plugin-default layering.

- agent/turn_context.py:522-573 - pre_llm_call context injection into
  the user turn, preserving the system prompt.

- agent/turn_finalizer.py:395-414 - post_llm_call receives both sides of
  the completed turn.

- agent/conversation_loop.py:1364-1381 - execution middleware wraps the
  selected main provider call.

- plugins/model-providers/custom/\_\_init\_\_.py:1-103 - existing
  local/custom provider support.

> **FOCUSED VALIDATION** The current checkout's plugin auxiliary-task
> and plugin LLM contract suites passed 71 of 71 tests:
> tests/hermes_cli/test_plugin_auxiliary_tasks.py and
> tests/agent/test_plugin_llm.py.

> **V0.1 IMPLEMENTATION NOTE** The shipped plugin registers an auxiliary task
> for Hermes configuration/model-picker visibility, but performs inference with
> its own loopback-only client. It accepts only the task's custom base URL,
> model, and timeout; it does not honor cloud providers or provider fallbacks.
> Proxies and redirects are disabled so the explicit local path stays local.
> The v0.1 exporter is corrected-only by default but does not claim automatic
> redaction; exported source text must be inspected before training or sharing.

# 4. Recommended plugin-first architecture

Build a standalone user plugin outside the Hermes repository. It should be
profile-aware, disabled by default until explicitly enabled, and contain
no core-tool registration. The cloud agent does not need another tool
schema on every API request merely so a local classifier can do
paperwork.

```text
~/.hermes/plugins/auxiliary-brain/
  plugin.yaml       # opt-in standalone plugin manifest
  __init__.py       # registers CLI commands, hooks, and optional aux task
  local_model.py    # bounded structured completion wrapper
  schemas.py        # check-in, follow-up, and research-note contracts
  ledger.py         # SQLite/JSONL state and provenance
  training_data.py  # accepted/corrected examples and redaction
  cli.py            # benchmark, export, train, evaluate, promote, rollback
  SKILL.md           # explicit operator workflow for cloud verification
  data/              # profile-local datasets, metrics, adapter registry
```

## Runtime path A - explicit local CLI task

1.  The operator invokes `hermes brain run progress_checkin` with natural
    language from a local terminal.

2.  The plugin sends only the check-in text and a strict schema to LFM.

3.  Deterministic validation checks JSON shape, confidence, dates, and
    allowed fields.

4.  A valid result updates the local ledger and returns structured output
    without calling the main cloud model.

5.  Invalid or ambiguous input returns an explicit local error. The plugin
    does not silently submit failed CLI tasks to the cloud agent.

## Runtime path B - ordinary cloud conversation

6.  pre_llm_call sends the latest user message to LFM for a bounded
    classification.

7.  The plugin returns compact structured context such as progress intent,
    extracted fields, and confidence.

8.  Hermes appends that context to the current user turn, not the system
    prompt, so the cached prefix remains stable.

9.  The main cloud model reasons with the structured signal and remains responsible
    for the final answer.

*This path improves consistency and creates shadow-evaluation data, but
it does not save the cloud call.*

## Runtime path C - learning loop

10. post_llm_call stores the LFM prediction, cloud outcome, user
    correction, schema version, and provenance.

11. A local curation job redacts secrets and excludes changing private
    facts from weight training.

12. The operator exports a frozen JSONL dataset and chronological
    holdout set.

13. A native Hugging Face checkpoint is fine-tuned with supervised LoRA;
    the serving artifact is produced afterward.

14. The candidate adapter runs in shadow mode, then low-risk canary
    mode, and is promoted only after passing fixed gates.

## Suggested plugin commands

| **Command** | **Purpose** | **Cloud call?** |
|:---|:---|:---|
| hermes brain run progress_checkin \<text\> | Explicit local progress check-in | No |
| hermes brain status | Endpoint health, active adapter, and data counts | No |
| hermes brain correct \<id\> | Attach an operator correction to a stored example | No |
| hermes brain evaluate | Run a frozen stock-model or candidate evaluation | No |
| hermes brain export | Create corrected training data; manually inspect/redact in v0.1 | No |
| Future training command | Run or submit an explicit LoRA training job | No |
| Future promote/rollback command | Atomically select or revert an adapter | No |

# 5. Integration options and recommendation

| **Option** | **What it achieves** | **Core change** | **Recommendation** |
|:---|:---|:---|:---|
| Named custom provider only | Makes LFM selectable as a model endpoint, but adds no task workflow or training loop. | None | Necessary plumbing, not the solution |
| Model-provider plugin | Adds a provider profile or transport quirks. | None | Unnecessary for standard OpenAI-compatible serving |
| Standalone auxiliary-brain plugin | Commands, hooks, schemas, ledgers, capture, eval, and training operations. | None | Recommended pilot |
| Plugin + auxiliary task | Adds a dedicated model-picker route for local auxiliary work. | None, with current auxiliary client | Useful; acceptable private-plugin coupling |
| Task-aware ctx.llm | Lets the supported plugin facade select a registered auxiliary task directly. | Small generic API extension | Recommended only for polished/upstream reuse |
| Cross-surface turn router | Allows high-confidence natural language to return locally without calling GPT. | Medium generic lifecycle extension | Defer until measured demand |

## Why a model-provider plugin is not enough

Provider profiles answer 'how do I reach and shape requests for this
provider?' They do not answer 'which messages should the local model
handle, how is a correction captured, what ledger changes, when does
cloud escalation happen, and which adapter is promoted?' Those are
application-level responsibilities and belong in the standalone plugin.

## The small API ergonomics gap

Hermes exposes two complementary surfaces that do not yet meet cleanly.
register_auxiliary_task gives a plugin its own auxiliary.\<key\>
configuration and model picker entry. ctx.llm is the documented
supported facade for plugin model calls. However, ctx.llm.complete and
complete_structured do not accept a task key. A plugin can either call
agent.auxiliary_client.call_llm(task=...) directly or use ctx.llm with
trusted provider/model overrides.

> **FUTURE SMALL CORE CHANGE** If the pilot becomes a reusable plugin,
> add an optional task= argument to the four ctx.llm completion methods
> and validate that the task belongs to the calling plugin. This is a
> generic, low-footprint bridge between two existing abstractions, not
> an LFM-specific special case.

## The real plugin-only boundary

The plugin CLI command can bypass the cloud because it calls the loopback
OpenAI-compatible endpoint directly and does not enter an agent conversation.
`pre_llm_call` can classify and enrich an ordinary message, but the agent still
calls the cloud model. The existing `llm_execution` middleware is not a safe
general router: it wraps the already-selected main client and downstream
validation expects the main transport's response shape.

Although plugin slash commands dispatch before normal prompt submission on an
idle gateway, current busy-session handling resolves only built-in commands.
A dynamic `/brain` received mid-turn can fall into follow-up handling and reach
the running cloud agent. A pre-dispatch hook is not a sound plugin workaround:
it runs before authorization, has no reply contract, and would require private
gateway state. Version 0.1 therefore does not register a slash command.

Therefore, a seamless experience in which Hermes sees arbitrary text,
lets LFM decide it is a routine check-in, returns a local answer,
persists a proper user/assistant pair, and never contacts GPT across
every surface would need a new generic turn-routing contract.

# 6. If transparent routing later proves worthwhile

Do not implement this before the explicit CLI pilot. If usage data shows that
explicit-command friction materially reduces adoption, design one
cross-surface route contract rather than separate CLI, gateway, TUI, and
Desktop hacks.

```text
turn_route(user_message, session_context) ->
  { action: "continue" }
  { action: "rewrite", user_message: "..." }
  { action: "respond", response_text: "...", metadata: {...} }
```

- Run after authentication and session resolution, but before building
  or calling AIAgent.

- Persist exactly one user/assistant pair and preserve strict role
  alternation.

- Do not mutate the system prompt, tool list, or prior messages; prompt
  caching remains untouched.

- Fail open to the cloud agent on timeout, invalid JSON, low confidence,
  or local server failure.

- Require plugin opt-in and an explicit list of routes allowed to return
  without cloud verification.

- Expose identical behavior to CLI, gateway, TUI, Desktop, API server,
  and cron entry paths.

- Keep high-stakes categories in continue mode regardless of local
  confidence.

# 7. Pilot work plan

## Stage 0 - endpoint and stock-model benchmark

- Serve the official or quantized LFM checkpoint through LM Studio,
  llama.cpp, or another OpenAI-compatible local server.

- Add it with hermes model as a named custom endpoint; do not switch the
  main conversation to it.

- Test /v1/models, chat completions, output length, timeouts, malformed
  JSON recovery, and concurrent requests.

- Build frozen evaluation sets for progress_checkin, followup_intent, and
  research_note_tagger.

## Stage 1 - plugin-only explicit CLI workflow

- Implement `hermes brain run`, `hermes brain status`, deterministic schema
  validation, and a profile-local SQLite ledger.

- Run stock LFM in shadow mode first; no automatic ledger mutation until
  the error distribution is known.

- Add correction capture and a corrected-only dataset exporter; require manual
  inspection/redaction until a deterministic scrubber is proven.

- Verify the cloud request count is zero for successful explicit
  check-ins.

## Stage 2 - local enrichment and first LoRA

- Enable pre_llm_call classification for ordinary turns, but keep GPT
  responsible for the final answer.

- Train a task-specific supervised LoRA only after collecting enough
  high-quality accepted or corrected examples.

- Compare the stock model, current adapter, and candidate adapter
  against a frozen chronological holdout.

- Promote by adapter pointer, never by overwriting the only known-good
  artifact.

## Stage 3 - evidence-based architecture decision

- Measure how often explicit commands are used, how often LFM escalates,
  and how much cloud work the plugin actually saves.

- If explicit routing is sufficient, keep Hermes core unchanged.

- If the auxiliary-task/config split causes real maintenance pain, add
  task-aware ctx.llm.

- If command friction is the dominant failure mode, propose the generic
  cross-surface turn router with E2E tests.

# 8. Acceptance gates

| **Gate** | **Starting threshold** | **Failure action** |
|:---|:---|:---|
| Structured output | At least 99.5% syntactically valid JSON after one bounded repair attempt | Escalate; do not write ledger |
| Low-risk field accuracy | At least 95% exact or schema-aware field accuracy | Remain shadow-only |
| Escalation recall | At least 99% for ambiguous, health, financial-judgment, and unsupported inputs | Block autonomous use |
| Cloud bypass | Zero cloud requests for accepted `hermes brain run` cases | Inspect CLI and direct-client path |
| Prompt cache | No system-prompt or tool-schema change on ordinary cloud turns | Reject integration design |
| Privacy | No secrets, account identifiers, raw sensitive exports, or changing private facts in training JSONL | Quarantine dataset |
| Rollback | One-command adapter rollback with last-known-good artifact retained | Do not promote |
| Local outage | Automatic cloud fallback or explicit non-destructive error | Never lose a check-in silently |

# 9. Risks and controls

| **Risk** | **Why it matters** | **Control** |
|:---|:---|:---|
| False confidence | A 230M model can produce neat JSON that is semantically wrong. | Schema checks, confidence floor, fixed escalation rules, frozen holdout. |
| Self-reinforcement | Training on unchecked model outputs teaches its own mistakes. | Only accepted, corrected, rule-derived, or cloud-verified labels train. |
| Private-data leakage | Training files may contain health, financial, or other sensitive information. | Redaction, restricted ACLs, retention limits, dataset manifests, deletion workflow. |
| Profile-wide capture | Plugin state is scoped to `HERMES_HOME`, not individual gateway senders. | Keep shadow/assist off on multi-user gateways unless shared capture is explicitly acceptable. |
| Local token disclosure | Auto-discovery probes several loopback ports; attaching one bearer token to every probe could disclose it to an unrelated local service. | Require an explicit loopback base URL whenever `AUXILIARY_BRAIN_API_KEY` is set. |
| Core duplication | A plugin could accidentally rebuild memory, cron, or background review. | Use existing Hermes stores and hooks; keep plugin scope narrow. |
| Surface inconsistency | Gateway-only tricks would make CLI/Desktop behavior differ and can bypass current busy-session command recognition. | Keep v0.1 explicit tasks CLI-only; require a generic authenticated cross-surface contract later. |
| Training artifact drift | Prompt, schema, dataset, base model, and adapter may become mismatched. | Version all five and require compatibility checks at load time. |
| Server failure | A local process may stop or change its loaded model. | Health checks, model-id verification, short timeout, fail open to cloud. |
| License/distribution | The model card labels the license lfm1.0. | Review current license terms before sharing adapters or packaging commercially. |

# 10. Final recommendation

> **GO** The concept is feasible as a plugin-first experiment. Build the
> local model server and auxiliary-brain plugin without modifying
> C:\src\hermes-agent. Start with explicit `hermes brain run` tasks and shadow
> evaluation. Add no model tool, do not replace the memory provider, do
> not route compression or background memory writes to LFM, and do not
> automate adapter promotion.

After the pilot, make one of three evidence-based decisions: keep the
solution entirely as a user plugin; add only task-aware ctx.llm as a
small generic API improvement; or, if explicit-command friction is
proven, design a carefully bounded cross-surface turn router. No
LFM-specific code belongs in the narrow Hermes core.

# Appendix A - proposed task contracts

## progress_checkin_v1

```json
{
  "category": "string|null",
  "status": "completed|partial|skipped|planned|unclear",
  "duration_minutes": 0,
  "obstacle": "string|null",
  "next_action": "string|null",
  "needs_cloud_review": false,
  "confidence": 0.0
}
```

## followup_intent_v1

```json
{
  "intent": "completed|reschedule|skip|needs_help|question|unclear",
  "requested_date": "ISO-8601|null",
  "reason": "string|null",
  "needs_cloud_review": false,
  "confidence": 0.0
}
```

## research_note_tagger_v1

```json
{
  "entities": ["string"],
  "source": "string|null",
  "source_date": "ISO-8601|null",
  "note_type": "fact|claim|opinion|question|followup|unclear",
  "missing_evidence": true,
  "requires_research": true,
  "requires_high_stakes_judgment": false,
  "confidence": 0.0
}
```

*The research contract tags and queues verification work. It never
recommends or executes a high-stakes action.*

# Appendix B - reviewed sources

**1. LiquidAI/LFM2.5-230M model card.** [Source](https://huggingface.co/LiquidAI/LFM2.5-230M) - parameters,
context length, model formats, license label, intended uses, and
fine-tuning links.

**2. Liquid AI: LFM2.5-230M - Built to Run Anywhere.** [Source](https://www.liquid.ai/blog/lfm2-5-230m) - release
positioning, edge deployment, extraction/tool-use focus, and
specialization example.

**3. Liquid AI fine-tuning documentation - TRL.** [Source](https://docs.liquid.ai/lfm/fine-tuning/trl) - SFT, DPO, and
recommended LoRA workflow.

**4. Hermes plugin LLM access guide.** [Source](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/plugin-llm-access.md) -
supported ctx.llm interface, structured output, trust gates, and plugin
responsibilities.

**5. Hermes model-provider plugin guide.** [Source](https://github.com/NousResearch/hermes-agent/blob/main/plugins/model-providers/README.md) -
provider profile discovery and its intended boundary.

# Appendix C - implementation questions to answer before building

- Which Windows GPU/CPU and available VRAM will run inference and LoRA
  training?

- Will the local server run in Windows, WSL2, or both?

- After Hermes gains a busy-safe authenticated plugin-command path, should
  `/brain` read/run commands be enabled, or is the CLI boundary sufficient?

- Which channel will be primary for check-ins: Desktop, Telegram, CLI,
  or another gateway?

- Should training examples retain original text locally, or only
  redacted structured fields?

- What is the minimum acceptable escalation latency when LFM defers to
  the main cloud model?

- Should sensitive task families share one plugin database or use
  separate encrypted stores and adapters?
