# Long-Horizon Config

`config/long_horizon.toml` is a focused OpenAI Codex profile for long-running coding tasks.

## Why Context Values Are Not Hard-Coded

Codex's model metadata already defines the GPT-5.5 context window and compaction behavior:

- `context_window = 272000`
- automatic compaction defaults to 90% of the context window
- effective usable input window is 95%

The config intentionally leaves `model_context_window` and `model_auto_compact_token_limit` unset so Codex follows the current model catalog instead of freezing stale values in this repo.

## Execution Posture

The default `long-horizon` profile uses:

```toml
approval_policy = "on-request"
sandbox_mode = "workspace-write"
```

That is appropriate for interactive long-horizon work where Codex can keep momentum but still asks before broader access. For unattended full-access operation, use `config/ralph_loop.toml` with `codex --profile yolo`.
