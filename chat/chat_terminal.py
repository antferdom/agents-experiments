# Adapted from: https://github.com/tinygrad/tinygrad/blob/9561803cb0370461bc991b3af7c4e9867cd8f0eb/examples/coder.py#L22
import re
import os
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text
from io import StringIO
from contextlib import redirect_stdout
import traceback

openai_api_key = os.getenv(
    "OPENAI_API_KEY",
    "xai-sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
)
openai_api_base = "https://api.x.ai/v1"  # "http://127.0.0.1:30000/v1"
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

console = Console()

messages = [
    {
        "role": "system",
        "content": "You are Quentin, a helpful assistant who writes concise Python code to answer questions.",
    }
]

while True:
    user_input = console.input("[bold blue]You:[/bold blue] ")
    if user_input.lower() == "exit":
        console.print("[yellow]Goodbye![/yellow]")
        break
    # chat memory
    messages.append({"role": "user", "content": user_input})

    stream = client.chat.completions.create(
        model="grok-4-latest",
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=32768,
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
    # Add assistant response to history (plain text)
    messages.append({"role": "assistant", "content": assistant_response.plain})

    # Extract Python code block from plain text
    plain_response = assistant_response.plain
    code_block_pattern = r"```python(.*?)```"
    match = re.search(code_block_pattern, plain_response, re.DOTALL)

    if match:
        code = match.group(1).strip()
        run = console.input("[bold red]Run the code? (y/n): [/bold red]").lower() == "y"
        if run:
            my_stdout = StringIO()
            try:
                with redirect_stdout(my_stdout):
                    exec(code)
                output = my_stdout.getvalue()
                console.print(f"[yellow]Code output:[/yellow]\n{output}")
                messages.append({"role": "user", "content": f"Code output:\n{output}"})
            except Exception as e:
                error_msg = "".join(traceback.format_exception_only(e))
                console.print(f"[red]Code error:[/red]\n{error_msg}")
                messages.append(
                    {"role": "user", "content": f"Code error:\n{error_msg}"}
                )
    else:
        console.print("<system>No Python code block found in the response.</system>")