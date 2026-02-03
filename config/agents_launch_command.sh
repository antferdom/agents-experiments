codex --search --model=gpt-5-codex -c model_reasoning_effort="high"
# using highest reasoning effort (xhigh)
codex --search --model=gpt-5.2 -c model_reasoning_effort="xhigh"
# default custom Grok profile
codex --config ~/grok_codex.toml
# switch between profiles
codex --config ~/grok_codex.toml --profile flagship