# Auxiliary Brain installed

1. Start a local OpenAI-compatible server with your small model loaded.
2. Start a new Hermes process, then run:

   ```console
   hermes brain setup --auto
   hermes brain doctor
   ```

3. Restart a running messaging gateway if you plan to use shadow or assist mode:

   ```console
   hermes gateway restart
   ```

Try `hermes brain status`, then:

```console
hermes brain run progress_checkin "Completed a planned session."
```

Version 0.1 intentionally does not register `/brain`; see the README for the
current Hermes busy-session routing limitation.

Mutable state is stored under `HERMES_HOME/auxiliary-brain/`, outside this
replaceable plugin directory. Updating the plugin does not erase its little
notebook.
