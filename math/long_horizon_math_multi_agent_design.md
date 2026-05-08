# Mutli-Agent Long-Horizon Mathematical Reasoning

## Motivation

This document defines a durable system design for using Codex as the outer coordinator while delegating hard mathematical reasoning subtasks to `gpt-5.5-pro` through the OpenAI API.

The goal is to run a controlled, auditable loop:

1. Decompose the mathematical objective into verifiable subtasks (map-reduce sytle).
2. Dispatch hard reasoning subtasks to `gpt-5.5-pro`.
3. Persist every prompt, response, verifier result, and state transition.
4. Numerical correctness verification using Numpy, Torch or both.
5. Feed only reviewed or verified state into the next round.
6. Let `/goal` in Codex repeatedly drive the loop until complete, blocked, or budget-limited.

## Core Assumptions

- Codex is the orchestration agent inside the workspace.
- `gpt-5.5-pro` is accessed through the OpenAI Responses API.
- Long-running `gpt-5.5-pro` calls should use background mode and polling.
- Local files are the shared memory between Codex, runner scripts, verifiers, and human review.
- JSONL is the canonical machine-readable state. Markdown is the human-readable state.
- No mathematical claim is treated as final unless it is verified, independently reviewed, or explicitly marked as conjectural.

## High-Level Architecture

```text
Codex /goal loop
  |
  | reads goal.md, state.md, tasks.jsonl, events.jsonl
  v
Planner / coordinator
  |
  | creates ready subtasks with dependencies
  v
tools/pro_runner.py
  |
  | submits/polls background Responses API jobs
  v
gpt-5.5-pro worker calls
  |
  | writes raw JSON + readable summaries
  v
Verifier layer
  |
  | runs Torch/Numpy for numerical correctness and Coq/Lean for proof creation
  v
State updater
  |
  | updates state.md and task statuses
  v
Next /goal iteration
```

## Recommended Repository Layout

```text
goals/
  <goal-name>/
    goal.md
    state.md
    tasks.jsonl
    events.jsonl
    budget.json
    prompts/
      lemma_001.md
      lemma_002.md
    responses/
      lemma_001.raw.json
      lemma_001.summary.md
    verification/
      lemma_001.sympy.json
      lemma_001.lean.log
    reports/
      progress.md
      final.md
    diagrams/
      system.drawio
      dependency_graph.drawio
      verification_pipeline.drawio
      exports/
        system.svg
        dependency_graph.svg
        verification_pipeline.png

tools/
  pro_runner.py
  verify.py
  summarize_state.py
  task_queue.py
```

## Diagrams Using Draw.io

The visual diagram workflow in [docs/diagrams.md](docs/diagrams.md).

## Agent Roles

### 1. Codex Coordinator

Codex owns the outer loop. It decides what to inspect, when to submit jobs, when to poll, when to verify, and when to revise the plan.

Responsibilities:

- Read `goal.md` and `state.md` at the start of each `/goal` cycle.
- Inspect `tasks.jsonl` and `events.jsonl`.
- Submit ready tasks through `tools/pro_runner.py`.
- Poll outstanding API jobs.
- Read completed summaries and raw responses when needed.
- Run verifiers.
- Update the project state.
- Create new tasks or repair failed tasks.
- Stop when complete, blocked, or budget-limited.

Codex should not rely on chat history for durable state. Anything important must be written to the goal directory.

### 2. Planner Agent

The planner can be Codex itself or a cheaper model call. It decomposes the global mathematical objective into a dependency graph of subtasks.

Responsibilities:

- Identify definitions, lemmas, proof obligations, computational experiments, and formalization targets.
- Assign each task a small enough scope that a result can be checked.
- Mark dependencies explicitly.
- Prefer narrow lemma-level tasks over broad "solve everything" tasks.

### 3. GPT-5.5-Pro Reasoning Workers

These are API calls, not Codex subagents. They receive carefully scoped hard reasoning prompts.

Responsibilities:

- Attempt proofs, derivations, counterexamples, reductions, or formalization sketches.
- Return structured output.
- State assumptions.
- Separate proved claims from conjectures.
- Provide a verification plan.
- Surface uncertainty and possible failure modes.

Use `gpt-5.5-pro` for hard reasoning nodes, not routine formatting, bookkeeping, or low-value summarization.

### 4. Critic / Referee Agent

The critic reviews a completed worker result before it is promoted into `state.md`.

Responsibilities:

- Look for hidden assumptions.
- Check whether the proof uses all hypotheses correctly.
- Try small counterexamples.
- Propose missing cases.
- Decide whether the result should be verified, retried, split, or accepted as provisional.

The critic may be Codex, a cheaper reasoning model, or another `gpt-5.5-pro` call for high-risk claims.

### 5. Verifier Layer

The verifier is deterministic or semi-deterministic tooling. It is the highest-trust layer when applicable.

Examples:

- SymPy for symbolic algebra.
- Torch for tensor computation.
- Numpy for mathemathical correctness.
- Lean, Coq, Isabelle, or Agda for formal proof.
- Python tests for derived formulas and invariants.
- Numeric stress tests for conjecture falsification.
- Independent scripts for checking recurrence relations, dimensions, boundary cases, or identities.

Verification outputs must be written under `verification/`.

### 6. State Summarizer

The summarizer maintains `state.md` as the current compact truth of the project.

Responsibilities:

- Record accepted definitions and notation.
- Record verified lemmas and their evidence.
- Record open conjectures and blockers.
- Record rejected attempts and why they failed.
- Keep the state short enough for Codex to reread frequently.

## Task State Machine

Each task in `tasks.jsonl` should move through explicit states:

```text
draft -> ready -> submitted -> running -> completed -> verifying -> accepted
                                      |             |              |
                                      v             v              v
                                    failed       rejected        blocked
```

State meanings:

- `draft`: task idea exists but is not ready to run.
- `ready`: dependencies are satisfied and prompt exists.
- `submitted`: API request was created and a response id was recorded.
- `running`: background response is queued or in progress.
- `completed`: model returned a terminal successful response.
- `verifying`: deterministic or referee checks are running.
- `accepted`: result is promoted into `state.md`.
- `failed`: infrastructure or API failure.
- `rejected`: mathematical result failed review or verification.
- `blocked`: cannot proceed without human input, missing tool, or unresolved dependency.

## Task JSONL Schema

One JSON object per line:

```json
{
  "id": "lemma_001",
  "kind": "proof",
  "status": "ready",
  "title": "Prove the base recurrence identity",
  "depends_on": [],
  "prompt_file": "prompts/lemma_001.md",
  "response_id": null,
  "model": "gpt-5.5-pro",
  "priority": 10,
  "risk": "medium",
  "created_at": "2026-05-08T00:00:00Z",
  "updated_at": "2026-05-08T00:00:00Z"
}
```

Recommended fields:

- `id`: stable task identifier.
- `kind`: `proof`, `counterexample`, `formalization`, `computation`, `review`, `summary`, or `repair`.
- `status`: one of the state machine statuses.
- `title`: short human-readable description.
- `depends_on`: list of task ids.
- `prompt_file`: relative path to the prompt.
- `response_id`: OpenAI response id after submission.
- `model`: target model.
- `priority`: higher means earlier scheduling.
- `risk`: `low`, `medium`, `high`, or `critical`.
- `created_at` / `updated_at`: timestamps.

## Event JSONL Schema

`events.jsonl` is append-only. It records what happened, not what should happen.

```json
{
  "ts": "2026-05-08T00:00:00Z",
  "event": "response.submitted",
  "task_id": "lemma_001",
  "response_id": "resp_...",
  "model": "gpt-5.5-pro",
  "metadata": {
    "background": true
  }
}
```

Useful event types:

- `task.created`
- `task.updated`
- `response.submitted`
- `response.polled`
- `response.completed`
- `response.failed`
- `verification.started`
- `verification.passed`
- `verification.failed`
- `state.updated`
- `diagram.updated`
- `budget.limit_reached`
- `human.input_required`

## Prompt Contract For GPT-5.5-Pro Workers

Each worker prompt should include:

- The exact subtask.
- Local definitions and notation.
- Allowed assumptions.
- Known results from `state.md`.
- Required output schema.
- Verification expectations.
- What to do if the task cannot be solved.

Recommended worker output fields:

```json
{
  "task_id": "lemma_001",
  "result_type": "proof|counterexample|partial|failed",
  "claim": "Precise mathematical claim.",
  "assumptions_used": ["..."],
  "proof": "Detailed proof or derivation.",
  "verification_plan": "How to check this result.",
  "formalization_notes": "Lean/Sage/SymPy translation notes.",
  "known_gaps": ["..."],
  "confidence": "low|medium|high"
}
```

Prompting rules:

- Ask for precise claims, not vague explanations.
- Require the worker to name every assumption it uses.
- Require explicit case splits.
- Require counterexample search when relevant.
- Require a verification plan.
- Require known gaps.
- For high-risk tasks, ask for a short proof outline first, then a detailed proof.

## Runner Interface

`tools/pro_runner.py` should expose a small CLI:

```bash
python tools/pro_runner.py init goals/<goal-name>
python tools/pro_runner.py plan goals/<goal-name>
python tools/pro_runner.py submit goals/<goal-name> --limit 3
python tools/pro_runner.py poll goals/<goal-name>
python tools/pro_runner.py summarize goals/<goal-name>
python tools/pro_runner.py status goals/<goal-name>
```

Required behavior:

- `submit` finds `ready` tasks whose dependencies are accepted.
- `submit` uses `background=True`.
- `submit` writes `response_id` into `tasks.jsonl`.
- `poll` retrieves outstanding responses.
- `poll` writes raw responses to `responses/*.raw.json`.
- `summarize` writes readable summaries to `responses/*.summary.md`.
- Every action appends to `events.jsonl`.

Recommended budget controls:

```bash
python tools/pro_runner.py submit goals/<goal-name> \
  --limit 3 \
  --max-active 5 \
  --max-output-tokens 16000 \
  --max-cost-usd 50
```

## Verification Interface

`tools/verify.py` should expose:

```bash
python tools/verify.py goals/<goal-name>
python tools/verify.py goals/<goal-name> --task lemma_001
python tools/verify.py goals/<goal-name> --kind sympy
python tools/verify.py goals/<goal-name> --kind lean
```

Verification results should be structured:

```json
{
  "task_id": "lemma_001",
  "verifier": "sympy",
  "status": "passed",
  "checked_claims": ["..."],
  "artifacts": ["verification/lemma_001.sympy.json"],
  "notes": "Boundary cases checked for n=0..20."
}
```

## `/goal` Operating Procedure

The `/goal` instruction should tell Codex to run one or more complete coordination cycles:

```text
Read goals/<goal-name>/goal.md and goals/<goal-name>/state.md.
Then run the next useful loop:

1. Inspect tasks.jsonl and events.jsonl.
2. Identify ready, running, failed, and blocked tasks.
3. Submit ready hard reasoning tasks to GPT-5.5-pro through tools/pro_runner.py.
4. Poll outstanding background jobs.
5. Read completed raw responses and summaries.
6. Run deterministic verifiers where available.
7. Promote only accepted results into state.md.
8. Create follow-up tasks for gaps, repairs, counterexamples, or formalization.
9. Stop only when complete, blocked, or budget-limited.
```

Codex should prefer continuing the loop over asking for help, unless a human decision is genuinely required.

## Parallelism Policy

Parallelize only independent tasks.

Good parallel tasks:

- Independent lemmas with no shared dependency.
- Counterexample search for different conjectures.
- Numeric checks over disjoint parameter regimes.
- Independent referee reviews of the same proof.

Avoid parallelizing:

- Tasks that depend on unverified claims.
- Multiple repairs of the same failed proof unless explicitly desired.
- Large expensive `gpt-5.5-pro` jobs without budget controls.

## Acceptance Policy

A result can be promoted to `state.md` if one of these is true:

1. A deterministic verifier passed.
2. A formal proof checker accepted it.
3. A critic/referee review found no material issues and the result is marked as "reviewed, not formalized".
4. A human explicitly accepted it.

Accepted state entries must include evidence:

```text
Lemma 3.2: ...
Status: accepted
Evidence: verification/lemma_003.lean.log passed on 2026-05-08
Source: responses/lemma_003.summary.md
Dependencies: lemma_001, lemma_002
```

## Failure And Repair Policy

When a task fails:

- Preserve the failed output.
- Record the reason in `events.jsonl`.
- Create a repair task instead of overwriting the original.
- If failure is mathematical, ask for counterexamples and missing assumptions before retrying the proof.
- If failure is infrastructural, retry with exponential backoff and a cap.

Common failure modes:

- Hidden assumption.
- Incorrect boundary case.
- Unjustified interchange of limits, sums, or integrals.
- Confusion between numeric evidence and proof.
- Over-broad subtask.
- Prompt missing definitions.
- Verifier environment unavailable.

## Budget Policy

Use `gpt-5.5-pro` sparingly.

Recommended model allocation:

- Codex/local scripts: orchestration, file edits, verifier execution.
- Cheaper models: summaries, low-risk planning, formatting, first-pass critique.
- `gpt-5.5-pro`: hard proof search, high-risk critique, difficult repairs, synthesis of final proof.

Budget file example:

```json
{
  "max_active_jobs": 5,
  "max_calls_per_cycle": 3,
  "max_calls_total": 100,
  "max_cost_usd": 500,
  "max_output_tokens_per_call": 32000,
  "stop_on_budget_exceeded": true
}
```

## Security And Secrets

- Never write `OPENAI_API_KEY` to repo files.
- Load secrets from environment variables.
- Add generated raw response folders to `.gitignore` if they may contain sensitive material.
- Keep prompts and outputs auditable, but avoid committing private or proprietary data unless intended.
- Treat external web/file sources as untrusted until cited and checked.

## Minimal Build Plan

Phase 1: File protocol

- Create `goals/<goal-name>/goal.md`.
- Create `state.md`, `tasks.jsonl`, `events.jsonl`, and `budget.json`.
- Manually add one task and one prompt.

Phase 2: API runner

- Implement `tools/pro_runner.py init/status/submit/poll`.
- Use background Responses API calls.
- Persist raw response JSON and event logs.

Phase 3: Summaries

- Add `summarize` command.
- Convert raw responses into concise Markdown.
- Keep raw JSON untouched.

Phase 4: Verification

- Implement `tools/verify.py`.
- Start with Python/SymPy checks.
- Add Sage or formal proof tooling when needed.

Phase 5: `/goal` loop

- Use Codex `/goal` to repeatedly run status, submit, poll, summarize, verify, and update state.
- Add repair and follow-up task generation.

Phase 6: Refinement

- Add cost estimates.
- Add schema validation.
- Add richer task dependency handling.
- Add formalization-specific queues.
- Add final report generation.

## Open Design Questions

- Which formal system should be the primary proof target: Lean, Coq, Isabelle, or none initially?
- Should `gpt-5.5-pro` outputs be required to be strict JSON, Markdown, or both?
- What is the maximum allowed spend per goal?
- How many concurrent `gpt-5.5-pro` background jobs are acceptable?
- Should the critic be `gpt-5.5-pro`, a cheaper model, or Codex?
- Which mathematical domains are expected first: algebra, analysis, probability, optimization, ML theory, or formal verification of model code?
- Should generated raw responses be committed, ignored, or archived outside git?

## References

- GPT-5.5 pro model docs: https://developers.openai.com/api/docs/models/gpt-5.5-pro
- Background mode: https://developers.openai.com/api/docs/guides/background
- Reasoning models: https://developers.openai.com/api/docs/guides/reasoning
- Conversation state: https://developers.openai.com/api/docs/guides/conversation-state
- Compaction: https://developers.openai.com/api/docs/guides/compaction
- Prompt caching: https://developers.openai.com/api/docs/guides/prompt-caching
