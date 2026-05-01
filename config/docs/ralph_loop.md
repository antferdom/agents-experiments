# Ralph Loop Config

`config/ralph_loop.toml` enables Codex CLI 0.128.0+'s persistent goal behavior and provides two profiles:

```toml
[features]
goals = true
```

## Assertion

The canonical config key is `goals`, and it is disabled by default.

Evidence from the local `codex/` checkout:

- `codex-rs/features/src/lib.rs` registers `Feature::Goals` with key `"goals"` and `default_enabled: false`.
- `codex-rs/tools/src/tool_config.rs` includes goal tools only when `features.enabled(Feature::Goals)` is true.
- `codex-rs/tools/src/goal_tool.rs` defines the model-facing goal tools: `get_goal`, `create_goal`, and `update_goal`.
- `codex-rs/tui/src/chatwidget/slash_dispatch.rs` gates `/goal` behind `Feature::Goals`.
- `codex-rs/app-server/README.md` documents `thread/goal/set`, `thread/goal/get`, `thread/goal/clear`, `thread/goal/updated`, and `thread/goal/cleared`.
- `codex-rs/core/src/goals.rs` owns automatic goal continuation and requires a non-ephemeral persisted thread with a local state database.

The installed `codex-cli 0.128.0` binary also reports `goals` in `codex features list`; in that binary it is staged as `under development` and disabled by default. The local `codex/` checkout currently stages it as experimental. The stage label differs, but the config key and default-off behavior agree.

## What This Config Does

`ralph_loop.toml` keeps the feature gate global and puts execution posture into profiles:

- Enables `[features].goals = true`.
- Uses `history.persistence = "save-all"` so sessions are materialized and resumable.
- Uses `gpt-5.5`, the current Codex model-catalog frontier model for complex coding, research, and real-world work.
- Uses `model_reasoning_effort = "xhigh"`, `model_reasoning_summary = "detailed"`, and `model_verbosity = "high"` for long-running implementation work.
- Provides `ralph-loop` for interactive, safer long-horizon work: `approval_policy = "on-request"` and `sandbox_mode = "workspace-write"`.
- Provides `yolo` for externally sandboxed or fully trusted unattended runs: `approval_policy = "never"` and `sandbox_mode = "danger-full-access"`.

## Usage

Codex no longer supports switching config files with `--config`, so use a dedicated `CODEX_HOME` and copy or symlink this file as `config.toml`:

```shell
export CODEX_HOME="$HOME/.config/codex/ralph-loop"
mkdir -p "$CODEX_HOME"
cp /home/antonio/agents-experiments/config/ralph_loop.toml "$CODEX_HOME/config.toml"
codex
```

Use the autonomous profile explicitly when that is the intended trust model:

```shell
codex --profile yolo
```

Inside the TUI:

```text
/goal improve benchmark coverage
/goal pause
/goal resume
/goal clear
```

The bare `/goal` command opens or displays the current goal state once the thread exists.

## Notes

Goals are not just a prompt convention. Enabling the feature exposes persisted goal state and goal tools, lets the TUI show and control a thread goal, and allows the runtime to continue an active goal when the thread is idle. Plan mode ignores goal continuation, and ephemeral threads do not support goals.
