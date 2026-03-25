#!/usr/bin/env python
"""
Pit Crew AI – Dispatcher
Watches ./queue/pending/ for JSON task files, processes them with CrewAI agents,
and logs everything with Phoenix and Mem0.
"""

import os
import sys
import json
import shutil
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
import yaml
import phoenix as px

# ----------------------------------------------------------------------
# Phoenix setup – use persistent directory and environment variables
# ----------------------------------------------------------------------
phoenix_dir = "./traces/phoenix_db"
os.makedirs(phoenix_dir, exist_ok=True)
os.environ["PHOENIX_WORKING_DIR"] = phoenix_dir
os.environ["PHOENIX_PORT"] = "6006"
os.environ["PHOENIX_HOST"] = "localhost"

px.launch_app(use_temp_dir=False)   # no more port/host arguments to avoid deprecation

# ----------------------------------------------------------------------
# Load configuration
# ----------------------------------------------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ----------------------------------------------------------------------
# Mem0 setup (Chroma vector store)
# ----------------------------------------------------------------------
from mem0 import Memory

mem0_config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "pit_crew_memories",
            "path": "./memory/chroma_db",
        }
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": config["ollama"]["model"],
            "ollama_base_url": config["ollama"]["host"],   # correct param name
            "temperature": 0.1,
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": config["ollama"]["embedding_model"],
        }
    },
    "history_db_path": "./memory/mem0_history.db",
    "version": "v1.1"
}

mem = Memory.from_config(mem0_config)

# ----------------------------------------------------------------------
# CrewAI setup – use CrewAI's own LLM class
# ----------------------------------------------------------------------
from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(
    model=f"ollama/{config['ollama']['model']}",
    base_url=config["ollama"]["host"],
    temperature=0.1
)

# ----------------------------------------------------------------------
# Agent definitions (dummy for Day 1)
# ----------------------------------------------------------------------
def create_researcher():
    return Agent(
        role="Researcher",
        goal="Gather and summarize information from the provided context.",
        backstory="You are a meticulous researcher with a knack for extracting key facts.",
        llm=llm,
        memory=True,
        verbose=True,
    )

def create_writer():
    return Agent(
        role="Writer",
        goal="Produce clear, concise reports based on research findings.",
        backstory="You are a technical writer specializing in motorsport engineering.",
        llm=llm,
        memory=True,
        verbose=True,
    )

def create_crew(task_description):
    """Build a simple Crew with Researcher -> Writer for a given task."""
    researcher = create_researcher()
    writer = create_writer()

    research_task = Task(
        description=f"Research the following: {task_description}",
        expected_output="A list of key facts and relevant data points.",
        agent=researcher
    )

    writing_task = Task(
        description="Write a short summary report based on the research.",
        expected_output="A markdown report with title and bullet points.",
        agent=writer,
        context=[research_task]   # sequential
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True
    )
    return crew

# ----------------------------------------------------------------------
# File queue processor
# ----------------------------------------------------------------------
def process_task(task_path: Path):
    """Move task to processing, run Crew, then move to done/failed."""
    proc_dir = Path(config["paths"]["queue"]) / "processing"
    proc_dir.mkdir(parents=True, exist_ok=True)

    proc_path = proc_dir / task_path.name
    shutil.move(str(task_path), str(proc_path))

    # Load task
    with open(proc_path, "r") as f:
        task = json.load(f)

    try:
        # Create crew and run
        crew = create_crew(task.get("description", "No description provided"))
        result = crew.kickoff()

        # Save result to done/ folder
        done_dir = Path(config["paths"]["queue"]) / "done"
        done_dir.mkdir(parents=True, exist_ok=True)
        done_path = done_dir / task_path.name
        with open(done_path, "w") as f:
            json.dump({"task": task, "result": result}, f, indent=2)

        # Add summary to Mem0 memory
        mem.add(
            f"Task {task.get('task_id')} completed: {result}",
            user_id="orchestrator"
        )

        # Remove from processing
        proc_path.unlink()

    except Exception as e:
        # Move to failed/
        failed_dir = Path(config["paths"]["queue"]) / "failed"
        failed_dir.mkdir(parents=True, exist_ok=True)
        failed_path = failed_dir / task_path.name
        with open(failed_path, "w") as f:
            json.dump({"task": task, "error": str(e)}, f, indent=2)
        proc_path.unlink()
        print(f"Error processing {task_path.name}: {e}")

def watch_queue():
    """Poll the pending folder and process each file."""
    pending_dir = Path(config["paths"]["queue"]) / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    while True:
        for task_file in pending_dir.glob("*.json"):
            print(f"Found task: {task_file.name}")
            process_task(task_file)
        time.sleep(config["queue_poll_interval"])

# ----------------------------------------------------------------------
# Test mode
# ----------------------------------------------------------------------
def test_mode():
    """Create a sample task and process it immediately."""
    sample_task = {
        "task_id": "test_001",
        "project": "tyre_nns",
        "task": "ingest_data",
        "description": "Explain the role of tyre temperature in F1 performance.",
    }
    pending_dir = Path(config["paths"]["queue"]) / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    task_path = pending_dir / "test_001.json"
    with open(task_path, "w") as f:
        json.dump(sample_task, f, indent=2)
    print(f"Sample task created at {task_path}")
    process_task(task_path)
    print("Test completed. Check ./queue/done/test_001.json for result.")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()

    if args.test:
        test_mode()
    else:
        watch_queue()