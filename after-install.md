# Auxiliary Brain installed

1. Start a new Hermes process, then run the managed server:

   ```console
   hermes brain server start
   hermes brain doctor
   ```

   The first start downloads a pinned llama.cpp CPU build and the default LFM
   model. Use `hermes brain setup --auto` instead when you already run LM Studio,
   Ollama, vLLM, or another loopback OpenAI-compatible server.

2. Restart a running messaging gateway if you plan to use shadow or assist mode:

   ```console
   hermes gateway restart
   ```

Try `hermes brain status`, then:

```console
hermes brain run progress_checkin "Completed a planned session."
```

The current release intentionally does not register `/brain`; see the README for the
current Hermes busy-session routing limitation.

Mutable state is stored under `HERMES_HOME/auxiliary-brain/`, outside this
replaceable plugin directory. Updating the plugin does not erase its little
notebook.
