# Project Reorganization Summary

This document summarizes the reorganization of the Ana AI assignment codebase. Note: Part 2 (Video-to-Plate) has been removed from the project.

## Changes Made

### 1. Created Part 1 Structure (`part1/`)

All Part 1 (Vibe Check) code has been organized into `part1/`:

- **`part1/src/`**: All source code (pipeline, query parser, scorers, etc.)
- **`part1/prompts/`**: LLM prompt templates
- **`part1/scripts/`**: Data cleaning and preparation scripts
- **`part1/docs/`**: Part 1 specific documentation
- **`part1/README.md`**: Part 1 specific guide

### 2. Root Level Organization

- **`data/`**: Shared data directory (restaurants.json)
- **`docs/`**: Only contains `system_design.md` (Part 3)
- **`README.md`**: Updated to be project-level overview referencing both parts
- **`OLLAMA_SETUP.md`**: Shared setup guide (kept at root)
- **`requirements.txt`**: Shared dependencies

### 3. Removed/Relocated Files

- Removed root-level `src/`, `prompts/`, `scripts/` directories (moved to part1/)
- Moved Part 1 documentation to `part1/docs/`:
  - `SCORING_EXPLANATION.md`
  - `VIBE_CALCULATION.md`
  - `MIGRATION_TO_OLLAMA.md`
  - `PROJECT_ANALYSIS.md`

### 4. Code Updates

- Updated path references in `part1/src/pipeline.py` and `part1/src/vector_store.py` to correctly reference root-level `data/` directory
- Prompt paths remain correct (relative to part1/)

## Running the Code

### Part 1

```bash
# From project root
python -m part1.src.main --query "romantic spot with ocean views"
```

## Data Paths

- **Restaurant data**: `data/restaurants.json` (shared at root)
- **Vector store**: `part1/chroma_db/` (Part 1 specific)

## Note

Part 2 (Video-to-Plate) has been removed from the project. The codebase now focuses solely on Part 1: The Vibe Check.

