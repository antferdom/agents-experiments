# RLMs vs Multi-Agent Coding Systems

This folder is for notes around Recursive Language Models (RLMs) and how the
idea differs from current coding-agent and multi-agent workflows such as Codex,
Claude Code, Terminus, context folding, and hand-managed sub-agents.

The local artifacts are:

- `Recursive Language Models.pdf`: the reference paper.
- `codex_rlm.jpeg`: a screenshot showing an outer Codex session running
  `codex exec` with a short read-only prompt and capturing the child Codex
  response.
- `codex_rlm_integration.md`: synthesis of how to integrate this pattern into
  Codex loops and long-horizon tasks.

## Main Paradigm

The important difference is not simply "one model calls another model." Current
multi-agent coding systems can already do that. The difference is where control
flow lives.

In a typical multi-agent coding setup, the main LM decides to launch sub-agents
as tool calls from its own conversation. The orchestration is model-driven: the
main model has to remember the plan, emit the right number of calls, pass the
right prompts, collect the results, and decide when to stop. Files and context
folding help with memory, but the sub-agent call itself is still outside the
symbolic language that the agent is using.

In an RLM setup, recursive LM calls are embedded inside a symbolic runtime such
as a REPL, interpreter, or program. The LM call behaves like a first-class
operation in that runtime. Ordinary program constructs such as loops, maps,
filters, recursion, variables, caches, retries, and reducers can decide when and
how to call the model.

Short version:

- Multi-agent systems are usually model-orchestrated recursion.
- RLMs aim for program-orchestrated recursion.
- The key primitive is not "spawn an agent"; it is `lm_call(...)` inside a
  symbolic language.

## Why This Matters

Suppose we want to inspect 1M files and find a function with some hard-to-state
property `P`.

In a normal multi-agent system, the parent model may need to emit a huge number
of sub-agent tool calls or repeatedly decide which call to make next. Even if the
system can save intermediate state to files, the parent model is still driving
the execution at the conversation level.

In an RLM-like system, a program can do this:

```python
for path in files:
    if predicate_P(path):
        result = lm_call(f"Inspect {path} and answer the question.")
        save(result)
```

The symbolic program guarantees the loop, filter, cache, and save behavior. The
model is used only where semantic judgment is needed. This is more robust for
large fan-out tasks, conditional sub-calls, recursive decomposition, and
verifier-driven search.

## Paradigm Differences

| Axis | Multi-agent coding systems | RLM-style system |
| --- | --- | --- |
| Control flow | Mainly chosen by the parent LM through tool calls | Encoded in a program, REPL, or symbolic runtime |
| Sub-call primitive | External tool call emitted by the parent model | First-class operation inside the runtime |
| Scaling pattern | Parent model must coordinate many calls | Program can loop, map, filter, retry, and reduce |
| State | Often context, files, summaries, or agent memory | Runtime variables, files, caches, and explicit data structures |
| Failure mode | Parent forgets, skips calls, or loses global state | Program logic is stable; LM errors are isolated to call outputs |
| Best use | Flexible collaboration and coding tasks | Large structured search, recursive decomposition, verifier loops |

## What The Codex Screenshot Shows

The screenshot in `codex_rlm.jpeg` shows this kind of command:

```shell
codex exec --sandbox read-only \
  "Read README.md and answer in 1 sentence: according to the README, \
what are the three main components of an environment in Verifiers? \
Do not modify any files."
```

That is a useful minimal primitive: an outer Codex session can launch a child
Codex process, constrain it with a read-only sandbox, ask a narrow question, and
capture the answer.

By itself, this is only a nested Codex call. It becomes RLM-like when the call is
placed inside deterministic symbolic control flow: a shell loop, Python script,
test harness, search procedure, verifier, or REPL state machine. The important
shift is that the parent model no longer has to manually decide every sub-call.
The program decides the structure, and Codex handles the semantic leaf work.

## Proof Of Concept With Codex

A small proof of concept can use Codex as the recursive LM call while Python
provides the symbolic runtime.

This folder now includes a runnable version:

```shell
python rlm/codex_rlm_poc.py
```

The run writes compact child-call outputs to `rlm/poc-results.jsonl`.

Goal: scan a repository, select files with a deterministic predicate, ask Codex a
question about each selected file, save the answers, then summarize only the
compact results.

Minimal shape:

```python
from pathlib import Path
import json
import subprocess

ROOT = Path(".")
OUT = Path("rlm/poc-results.jsonl")

def predicate(path: Path) -> bool:
    return path.suffix in {".md", ".py"} and path.stat().st_size < 50_000

def codex_call(path: Path) -> dict:
    prompt = (
        f"Read {path}. Output only JSON with keys path, purpose, and notable_terms. "
        "Do not modify any files."
    )
    completed = subprocess.run(
        ["codex", "exec", "--sandbox", "read-only", prompt],
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    return {
        "path": str(path),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }

with OUT.open("w") as f:
    for path in ROOT.rglob("*"):
        if path.is_file() and predicate(path):
            f.write(json.dumps(codex_call(path)) + "\n")
```

The first version should run on a small file set. After that, the same structure
can be upgraded with:

- a stronger predicate `P`, possibly itself backed by a Codex call;
- parallel execution with a worker pool;
- caching keyed by file path and hash;
- retries and timeouts;
- a reducer step that asks Codex to summarize the JSONL result file;
- verifier checks for tasks where outputs can be mechanically scored.

## Success Criteria

The experiment demonstrates the RLM paradigm if:

1. The parent Codex session writes or invokes a driver program.
2. The driver program, not the parent conversation, controls the loop, filter,
   retries, and result storage.
3. Child Codex calls are narrow, read-only, and produce machine-readable output.
4. The parent session consumes only a compact artifact such as JSONL, not the
   full transcript of every child call.
5. Changing the predicate or loop shape changes the computation without asking
   the parent model to manually enumerate sub-agent calls.

This does not make Codex a full RLM runtime. It is a practical approximation:
Codex supplies the recursive LM call, while the filesystem and Python process
supply the symbolic execution layer. A true RLM would train and expose this
recursive call as a native part of the model's execution language.
