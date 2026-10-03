# Draft issue for huggingface/OpenEnv (not filed)

**Title:** `HarborEnv.close()` in sync mode never closes the websocket, leaking one env session per rollout

**Version:** OpenEnv main @ 7ee88d590d36ae1ac3daf2cc11bc551bca7a804f (FineEnvs' validated pin), Python 3.12, `harbor_env` from `envs/`.

**What happens.** `HarborSession(env=HarborEnv(url), owns_env=True)` closes its client in
`HarborSession.close()`. `HarborEnv.close()` calls `super().close()`, which in sync mode creates
`MCPClientBase._close_async()` and never awaits it:

```
/opt/OpenEnv/src/openenv/harbor/client.py:199: RuntimeWarning: coroutine 'MCPClientBase._close_async' was never awaited
  super().close()
```

So the server keeps one env session per finished rollout. The comment in
`HarborSession.close()` already names the consequence: "leaving it open holds an env session on
the server until `max_concurrent_envs` is exhausted."

**What we saw** (evaluation only: one env server per checkpoint, 48 worker threads, one
`HarborSession` per cell, Modal sandboxes):

- Memory grew about 8 MB per finished rollout; a 32 GB container was killed (exit -9) near 4,000 rollouts.
- On two checkpoints, after a few hundred rollouts every `run_rollout` call failed client-side with
  `TypeError: object NoneType can't be used in 'await' expression` (logged by
  `harbor_env/harness.py` as "harbor rollout call failed").

**Workaround.** One `HarborEnv` per worker thread, passed with `owns_env=False`, replaced after any
failed call. (TODO before filing: confirm from the full Phase 0 run that memory stays flat and the TypeError stops.)

**Suggested fix.** In `HarborEnv.close()` (or `MCPClientBase.close()`), run the async close when it
returns an awaitable, the way `_call` already does with `run_async_safely`.
