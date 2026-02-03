# Using Claude Code with various AI providers

**Create various settings file** I have one file for each provider, all in `~/.claude`

### KIMI K2.5: kimi_settings.json

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.moonshot.ai/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "${KIMI_API_KEY}",
    "API_TIMEOUT_MS": "3000000",
    "ANTHROPIC_MODEL": "kimi-k2.5",
    "ANTHROPIC_SMALL_FAST_MODEL": "kimi-k2.5",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "kimi-k2.5",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "kimi-k2.5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "kimi-k2.5",
    "CLAUDE_CODE_SUBAGENT_MODEL": "kimi-k2.5"
  }
}
```

### MiniMax M2.1: minimax_settings.json

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.minimax.io/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "${MINIMAX_API_KEY}",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1,
    "ANTHROPIC_MODEL": "MiniMax-M2.1",
    "ANTHROPIC_SMALL_FAST_MODEL": "MiniMax-M2.1",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M2.1",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "MiniMax-M2.1",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "MiniMax-M2.1"
  }
}
```

### ZAI GLM4.7: zai_settings.json

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "${ZAI_API_KEY}",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1,
    "ANTHROPIC_MODEL": "glm-4.7",
    "ANTHROPIC_SMALL_FAST_MODEL": "glm-4.7",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-4.7",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-4.7",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.7"
  }
}
```

### DeepSeek: deepseek_settings.json

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "${DEEPSEEK_API_KEY}",
    "API_TIMEOUT_MS": "3000000",
    "ANTHROPIC_MODEL": "deepseek-chat",
    "ANTHROPIC_SMALL_FAST_MODEL": "deepseek-chat",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-chat",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-chat",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-chat"
  }
}
```

## Start Claude using provider you want

- **Sonnet/Opus:** `claude`
- **kimi:** `claude --settings ~/.claude/kimi-settings.json`
- **minimax:** `claude --settings ~/.claude/minimax-settings.json`
- **zai:** `claude --settings ~/.claude/zai-settings.json`
- **deepseek:** `claude --settings ~/.claude/deepseek-settings.json`

## References

- [DeepSeek API Docs: Anthropic API](https://api-docs.deepseek.com/guides/anthropic_api)