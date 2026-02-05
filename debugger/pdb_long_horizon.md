### pdb Interaction in a Persistent Shell

If we were instantiated as a code agent via a tool like codex with the provided TOML config for long-horizon coding (`config/long_horizon.toml`), We will operate in a persistent shell environment. Assumptions here:

- **Persistent shell:** A long-running REPL/session (e.g., `Docker`) where state (variables, call stacks) persists across interactions.
- **Access:** Full read/write to filesystem for saving states; ability to pipe inputs to pdb non-interactively or via simulated user commands.
- **Limitations overridden**: True interactivity without hanging on `pdb.set_trace()`.

In this setup, we handle `pdb` debugging as a real, iterative process. Below is how a sample interaction would unfold for the toy example code (divide by zero bug):

```python
def divide(a, b):
    result = a / b
    return result

x = 10
y = 0
z = divide(x, y)
print(z)
```

 We use the shell to run `python -m pdb script.py`, simulate commands, and persist data.

In the persistent shell:

1. **Start Session**: `python -m pdb buggy.py` (or embed `import pdb; pdb.set_trace()` at key points).
   - Shell output: Enters pdb prompt (Pdb) at first line.
2. **Initial Commands**:
   - l (list): Show code context.
   - b 2 (breakpoint at line 2, inside divide).
   - c (continue): Run until breakpoint.
   - Output: Hits breakpoint before result = a / b.
3. **Inspection**:
   - p a → 10
   - p b → 0 (reveals issue)
   - where (stack trace): Shows call from main to divide.
   - s (step): Enter division line → ZeroDivisionError.
4. **Error Handling**:
   - On error, up (move up stack) to inspect caller frame.
   - p x, y → 10, 0
5. **Iteration**:
   - q (quit), edit `buggy.py` to add if b == 0: raise ValueError("Division by zero").
   - Restart: `python -m pdb buggy.py` → Test with valid y=2 → p result → 5.0 → c → Prints 5.0.

#### Steps to Implement Long-Horizon Debugging

1. **Persistent Storage Setup**:
   - Mount a volume (e.g., in Docker) for `/debug_memory/`.
   - On breakpoint/error, use a wrapper script around pdb to serialize state:
     - Call stack: `import traceback; with open('stack.md', 'w') as f: f.write(''.join(traceback.format_stack()))`
     - Variables: `import json; with open('vars.md', 'w') as f: f.write('# Variables\n' + json.dumps(locals(), indent=2))`
     - This creates MD files like `stack_20260205.md` for easy reading/searching.
2. **pdb Wrapper Tool**:
   - Script: A custom `pdb_wrapper.py` that runs pdb with piped commands from a queue (e.g., read from file for non-interactive long runs).
   - Example: `echo "b 5\nc\np var\nq" > commands.txt; python -m pdb_wrapper buggy.py < commands.txt`
   - For long-horizon: Chain sessions by saving/resuming from MD files (e.g., reload variables via `exec(open('vars.md').read())` – careful with security).
3. **Iteration Workflow**:
   - Break code into modules; debug one at a time, saving states.
   - Use ***git for versioned fixes***, with MD annotations (e.g., # Hypothesis: Null check missing).
   - For multi-session: **Load previous state via pickle** (safer than exec) or MD-parsed dicts.
4. **Handling Assumptions**:
   - Persistent shell: Use **tmux** or screen to keep **sessions alive**.
   - Approval policy "on-request": Pause for user confirmation on risky commands (e.g., file writes).
   - Web search enabled: Rely on local docs (e.g., pydoc pdb in shell) and search when necessary.

### What Need Most

For effective long-horizon debugging:

1. **Persistent, Queryable Memory**: Auto-saving call stacks/variables to searchable MD/JSON files (as above) – this acts as external memory, allowing me to "recall" states across sessions without recomputing.
2. **Interactive Input Queue**: A way to feed pdb commands dynamically (e.g., via WebSocket or file watcher) for true back-and-forth without simulation.
3. **Full Filesystem Access**: Read/write to code files mid-session for in-place edits/tests.
4. **Error-Resilient State Reload**: Tools to deserialize saved states safely (e.g., via cloudpickle for complex objects).
5. **Integration with My Tools**: Link to code_execution for hybrid mode (e.g., test snippets in REPL, then full pdb in shell).