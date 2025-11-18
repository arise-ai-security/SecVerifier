# LLM Provider Configuration

SecVerifier auto-detects the right credentials based on the model prefix you pass to `--model`. Use the table below to see which environment variables are read for each provider.

| Provider prefix | Example model string | Required variables |
| --------------- | -------------------- | ------------------ |
| `openai/`, `gpt-*` (except `gpt-oss*`) | `openai/gpt-4o`, `gpt-5-mini` | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` |
| `anthropic/`, `claude*` | `anthropic/claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_BASE_URL` |
| `google/`, `gemini*` | `google/gemini-1.5-pro` | `GOOGLE_API_KEY`, optional `GOOGLE_API_BASE_URL` |
| `ollama/`, `gpt-oss*`, `openai/gpt-oss*` | `ollama/gpt-oss:120b-cloud`, `gpt-oss:20b-cloud` | `OLLAMA_API_KEY` (falls back to `LLM_API_KEY`), optional `OLLAMA_BASE_URL` |

## Using Ollama Cloud or Local Ollama

1. Export your key and endpoint (or place them in `.env`):

   ```bash
   export OLLAMA_API_KEY="<ollama api key>"
   export OLLAMA_BASE_URL="https://ollama.com"
   ```

   > Tip: point `OLLAMA_BASE_URL` at `http://localhost:11434` to use a local `ollama serve` daemon.

2. Run SecVerifier with either the explicit `ollama/` prefix *or* the concise `gpt-oss:*` alias — the CLI now normalizes these automatically:

   ```bash
   python main.py \
     --instance-id wasm3.cve-2022-28966 \
     --model gpt-oss:120b-cloud \
     --dataset SEC-bench/Seed \
     --split cve
   ```

   The factory rewrites `gpt-oss:*` and `openai/gpt-oss-*` strings to `ollama/<model>` under the hood, so LiteLLM always speaks to the Ollama-compatible endpoint while reusing the same `LLM_API_KEY` credentials.
