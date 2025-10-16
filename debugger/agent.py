import re
import os
import sys
import subprocess
import time
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.text import Text
from pathlib import Path
import requests

openai_api_key = os.getenv(
    "OPENAI_API_KEY",
    "xai-sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
)
openai_api_base = "https://api.x.ai/v1"  # "http://127.0.0.1:30000/v1"
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

console = Console()

wd: str = Path(__file__).parent.absolute()
debug_middleware = (wd / "middleware_debug.py").as_posix()
debug_server = subprocess.Popen([sys.executable, debug_middleware], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

messages = [
    {
        "role": "system",
        "content": 
            """
            You are Quentin, a helpful assistant who writes concise Python code to answer questions. 
            After running code, review the output and add assertions to verify the correctness based on the user's intent. 
            Use debug commands like /debug/connect, /debug/status, /debug/stacktrace, /debug/variables to inspect the state if needed.
            """,
    }
]

while True:
    user_input = console.input("[bold blue]You:[/bold blue] ")
    if user_input.lower() == "exit":
        debug_server.terminate()
        console.print("[yellow]Goodbye![/yellow]")
        break
    # chat memory
    messages.append({"role": "user", "content": user_input})

    stream = client.chat.completions.create(
        model="grok-4-latest",
        messages=messages,
        stream=True,
        temperature=0.7,
    )

    # Stream and render assistant response with styles
    assistant_response = Text()
    current_type = None
    with Live(assistant_response, refresh_per_second=4, console=console) as live:
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                content = chunk.choices[0].delta.content
                assistant_response.append(content)
                live.update(assistant_response)

    messages.append({"role": "assistant", "content": assistant_response.plain})

    # Extract Python code block from plain text
    plain_response = assistant_response.plain
    code_block_pattern = r"```python(.*?)```"
    match = re.search(code_block_pattern, plain_response, re.DOTALL)

    # Handle debug commands in the response
    debug_commands = [line.strip() for line in plain_response.split('\n') if line.strip().startswith('/debug/')]
    for cmd in debug_commands:
        parts = cmd.split()
        command = parts[0]
        if command == '/debug/connect':
            host = parts[1] if len(parts) > 1 else 'localhost'
            port = int(parts[2]) if len(parts) > 2 else 5678
            try:
                response = requests.post('http://localhost:8000/connect', json={'host': host, 'port': port})
                result = response.json()
                console.print(f"[green]Debug connect result:[/green] {result}")
                messages.append({"role": "user", "content": f"Debug connect result: {result}"})
            except Exception as e:
                console.print(f"[red]Debug connect error:[/red] {e}")
                messages.append({"role": "user", "content": f"Debug connect error: {e}"})
        elif command == '/debug/status':
            try:
                response = requests.get('http://localhost:8000/status')
                result = response.json()
                console.print(f"[green]Debug status:[/green] {result}")
                messages.append({"role": "user", "content": f"Debug status: {result}"})
            except Exception as e:
                console.print(f"[red]Debug status error:[/red] {e}")
                messages.append({"role": "user", "content": f"Debug status error: {e}"})
        # Add more commands as needed

    if match:
        code = match.group(1).strip()
        run = console.input("[bold red]Run the code? (y/n): [/bold red]").lower() == "y"
        if run:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=wd) as f:
                f.write('import debugpy\ndebugpy.listen(5678)\ndebugpy.wait_for_client()\ndebugpy.breakpoint()\n' + code)
                temp_file = f.name
            console.print(f"[yellow]Temp file: {temp_file}[/yellow]")
            proc = subprocess.Popen([sys.executable, temp_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(1)
            try:
                response = requests.post('http://localhost:8000/connect', json={'host': 'localhost', 'port': 5678, 'pid': proc.pid})
                result = response.json()
                console.print(f"[green]Connected to debug session:[/green] {result}")
                messages.append({"role": "user", "content": f"Connected to debug session: {result}"})
                try:
                    threads_response = requests.get('http://localhost:8000/threads')
                    threads_result = threads_response.json()
                    console.print(f"[green]Threads:[/green] {threads_result}")
                    messages.append({"role": "user", "content": f"Threads: {threads_result}"})
                except Exception as e:
                    console.print(f"[red]Threads error:[/red] {e}")
                    messages.append({"role": "user", "content": f"Threads error: {e}"})
                try:
                    cont_response = requests.post('http://localhost:8000/continue')
                    cont_result = cont_response.json()
                    console.print(f"[green]Continued execution:[/green] {cont_result}")
                except Exception as e:
                    console.print(f"[red]Continue error:[/red] {e}")
            except Exception as e:
                console.print(f"[red]Connect error:[/red] {e}")
            stdout, stderr = proc.communicate()
            output = stdout.decode().strip()
            error = stderr.decode().strip()
            if output:
                console.print(f"[yellow]Code output:[/yellow]\n{output}")
                messages.append({"role": "user", "content": f"Code output:\n{output}"})
            if error:
                console.print(f"[red]Code error:[/red]\n{error}")
                messages.append({"role": "user", "content": f"Code error:\n{error}"})
            os.unlink(temp_file)
    else:
        console.print("<system>No Python code block found in the response.</system>")