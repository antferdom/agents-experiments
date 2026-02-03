Codex does not longer have a `--config`  flag. Thus we can't have multiple `config.toml` files to switch between model providers. We setup `CODEX_HOME` as alternative to use to point to a custom folder.

```shell
export CODEX_HOME=$HOME/.config/codex/grok/
mkdir -p $CODEX_HOME
mv ~/.codex/grok_config.toml $CODEX_HOME/config.toml
codex   # now uses that folder
```

