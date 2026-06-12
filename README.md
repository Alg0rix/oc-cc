# oc-cc

Claude Code → OpenCode Go proxy.

Lets you run Claude Code while routing all model calls through the OpenCode Go
API (OpenAI-compatible endpoint), so you can use models like `deepseek-v4-pro`
with Claude Code's tools.

## How it works

```
Claude Code  --(Anthropic /v1/messages)-->  oc-cc proxy
                                                  |
                                                  v
                                      OpenCode Go /v1/chat/completions
```

The proxy:

- Translates Anthropic request format to OpenAI format.
- Translates OpenAI SSE streaming chunks back to Anthropic SSE events.
- Converts tool definitions and tool call/results between formats.
- Forces the configured OpenCode Go model regardless of what Claude Code asks for.

## Install

```bash
git clone https://github.com/Alg0rix/oc-cc.git
cd oc-cc
# no dependencies to install — uses only Python stdlib
```

Make sure you have an OpenCode Go API key saved by the OpenCode CLI in:

```text
~/.local/share/opencode/auth.json
```

You can override the model or endpoint with environment variables:

```bash
export OPENCODE_GO_MODEL=deepseek-v4-pro   # default
export OPENCODE_GO_BASE_URL=https://opencode.ai/zen/go/v1
```

## Usage

### Interactive

```bash
./bin/claude-oc
```

### One-shot

```bash
./bin/claude-oc -p "say hi only" --model claude-sonnet-4-6
```

### Auto mode (skip permission prompts)

```bash
./bin/claude-oc --dangerously-skip-permissions -p "task"
```

## Files

| File | Purpose |
|---|---|
| `oc_cc/proxy.py` | The proxy server |
| `bin/claude-oc` | Wrapper that starts the proxy and launches `claude` |
| `requirements.txt` | Python dependencies |

## Development

Run the proxy directly:

```bash
python3 oc_cc/proxy.py --port 9877
```

Then in another terminal:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:9877 claude
```
