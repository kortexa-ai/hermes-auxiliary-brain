# Auxiliary Brain installed

1. Start a new Hermes process, then run the managed server:

   ```console
   hermes brain server start
   hermes brain doctor
   ```

   The first start downloads a pinned llama.cpp CPU build and the default LFM
   model. Use `hermes brain setup --auto` instead when you already run LM Studio,
   Ollama, vLLM, or another loopback OpenAI-compatible server.

2. Restart a gateway that was already running during installation so it loads
   the plugin. Optionally enable `/brain` after acknowledging the current Hermes
   busy-session risk:

   ```console
   hermes brain gateway status
   hermes brain gateway enable --acknowledge-busy-risk
   hermes gateway restart
   ```

   The surface stays off by default. Only send `/brain` **between turns**, while
   Hermes is idle; current Hermes versions may route a dynamic plugin command
   received during a busy turn to the main model as follow-up text. The host fix
   is still a draft upstream. The fixed local commands are:

   ```text
   /brain help
   /brain status
   /brain checkin <text>
   /brain followup <text>
   /brain note <text>
   /brain extract <text>
   ```

   Task input is capped at 8,000 characters, and gateway errors are sanitized.
   Disable the surface with `hermes brain gateway disable`. Once the plugin is
   loaded, gateway enable/disable and shadow/assist mode changes are read on
   each invocation and need no restart. The idle `/brain` handler rejects
   multiplex-profile gateways; use the intended profile's local
   `hermes brain ...` CLI.
   With capture enabled, `/brain` inputs and outputs are stored profile-wide
   without sender attribution; use a dedicated profile for multi-user gateways.

Try `hermes brain status`, `hermes brain doctor`, or
`hermes brain server logs --lines 100`, then:

```console
hermes brain run progress_checkin "Completed a planned session."
```

When you have reviewed corrections, inspect learning readiness with
`hermes brain train status`. Training is optional, explicit, and installs its
heavy ML dependencies into profile-local environments only when you run
`hermes brain train install`. Read `docs/training.md` in the installed plugin
directory before training or sharing an adapter.

Mutable state is stored under `HERMES_HOME/auxiliary-brain/`, outside this
replaceable plugin directory. Updating the plugin does not erase its little
notebook.
