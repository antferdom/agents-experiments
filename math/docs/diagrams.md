# Diagrams Using Draw.io

Text diagrams and Mermaid are useful for quick iteration, but long-horizon mathematical projects also need editable visual diagrams for architecture, proof dependencies, verifier pipelines, and final reports.

## Draw.io MCP Setup

The draw.io MCP server is configured in Codex as a remote streamable HTTP MCP server:

```bash
codex mcp add drawio --url https://mcp.draw.io/mcp
codex mcp list
codex mcp get drawio --json
```

The expected configured entry is:

```toml
[mcp_servers.drawio]
url = "https://mcp.draw.io/mcp"
```

After adding the server, restart Codex or start a new session so the MCP tools are loaded into the agent runtime.

## Diagram Source Policy

Draw.io files should be treated as editable source artifacts, not disposable exports.

Recommended convention:

- `.drawio`: canonical editable source.
- `.svg`: preferred lightweight export for docs and GitHub previews.
- `.png`: fallback export for reports, slides, and tools that do not render SVG well.
- `.md`: diagram intent, update notes, and links to source/export files.

Example:

```text
goals/<goal-name>/diagrams/
  system.drawio
  system.notes.md
  dependency_graph.drawio
  dependency_graph.notes.md
  verification_pipeline.drawio
  exports/
    system.svg
    dependency_graph.svg
    verification_pipeline.png
```

## When To Use Draw.io Instead Of Mermaid

Use Mermaid for:

- Fast sketches.
- Task state machines.
- Small dependency graphs.
- Diagrams that should remain plain text in Markdown.

Use draw.io for:

- Dense architecture diagrams.
- Multi-lane agent workflows.
- Proof dependency maps with annotations.
- Diagrams intended for papers, slides, or reports.
- Cases where manual visual editing matters.

## Codex Diagram Loop

When `/goal` needs a visual diagram:

1. Read the relevant `state.md`, `tasks.jsonl`, and existing diagram notes.
2. Draft or update a diagram specification in `diagrams/<name>.notes.md`.
3. Use the draw.io MCP tools to create or modify `diagrams/<name>.drawio`.
4. Export a reviewable artifact to `diagrams/exports/<name>.svg` or `.png`.
5. Link the export from `reports/progress.md`, `reports/final.md`, or the root design doc.
6. Record the diagram update in `events.jsonl` with a `diagram.updated` event.

Diagram generation should not replace mathematical state. The source of truth for claims remains `state.md`, task records, and verifier artifacts.

## Suggested Diagram Types

- `system.drawio`: end-to-end architecture of Codex, task queue, GPT-5.5-pro workers, verifier layer, and state updater.
- `dependency_graph.drawio`: accepted lemmas, open conjectures, blocked tasks, and dependency edges.
- `verification_pipeline.drawio`: verifier stages, proof assistant targets, numerical checks, and acceptance gates.
- `budget_flow.drawio`: where API calls, retries, and expensive review paths happen.
- `failure_repair.drawio`: how failed proofs become counterexample searches, repair tasks, or revised assumptions.

## Diagram Acceptance Policy

A diagram is accepted when:

- Its `.drawio` source is stored under the goal directory.
- A rendered export exists.
- The diagram agrees with `state.md`.
- Any claim shown as proved has corresponding evidence in verification artifacts or accepted task records.
- The diagram notes describe what changed and what remains provisional.
