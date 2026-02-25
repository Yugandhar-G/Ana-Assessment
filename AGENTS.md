# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Ana AI is a Python-based AI culinary assistant with two independent parts:
- **Part 1** (port 8000): Hybrid RAG + vibe-based restaurant search with FastAPI backend and Gradio UI
- **Part 2** (port 8001): Agentic search variant with FastAPI backend and Gradio UI

Both share `data/restaurants.json` (360 Maui restaurants) and require a valid `GEMINI_API_KEY`.

### Required Secret

`GEMINI_API_KEY` must be set as an environment variable (obtain from https://makersuite.google.com/app/apikey). The app reads it via `os.getenv("GEMINI_API_KEY")`. Do NOT create a `.env` file manually — the code's `load_dotenv()` fallback picks up env vars automatically.

### Running Services

```bash
# Part 1 FastAPI (port 8000)
cd /workspace && python3 -m uvicorn part1.api:app --host 0.0.0.0 --port 8000 --reload

# Part 2 FastAPI (port 8001)
cd /workspace && python3 -m uvicorn part2.api:app --host 0.0.0.0 --port 8001 --reload

# Part 1 Gradio UI (connects to Part 1 API on port 8000)
cd /workspace/part1 && python3 conversational_gradio_app.py

# Part 2 Gradio UI (connects to Part 2 API on port 8001)
cd /workspace/part2 && python3 conversational_gradio_app.py
```

### Key Caveats

- **First startup takes 30-60s**: ChromaDB vector store must be built by embedding all 360 restaurants via the Gemini API. Subsequent starts are fast (cached in `part1/chroma_db/`).
- **No automated tests exist**: The repo has no test suite — verify manually via API endpoints or Gradio UI.
- **No lint config in repo**: Use `ruff check /workspace` for linting (ruff is installed in the dev environment).
- **`google.generativeai` deprecation warning**: The SDK shows a FutureWarning about switching to `google.genai`. This is cosmetic and does not affect functionality.
- **Part 1 and Part 2 are independent**: They can be started and tested separately.
- **PATH**: Add `$HOME/.local/bin` to `PATH` if pip-installed CLI tools (gradio, uvicorn, ruff) are not found.
