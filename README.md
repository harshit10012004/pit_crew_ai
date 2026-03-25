# Pit Crew AI – Foundation Day 1

## Quick Start
1. Run the setup commands in the terminal.
2. Ensure Ollama is running (`ollama serve` in background).
3. Run the dispatcher: `python main.py`
4. In another terminal, drop a task file into `queue/pending/` (e.g., copy sample_task.json).
5. Watch the console for processing.

## Test
```bash
python main.py --test