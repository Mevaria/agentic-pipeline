"""Run the implementer on one task.

    python -m pipeline.run_implementer tasks/delete_book

A task folder holds task.json (type, short_description, spec) and a tests
folder whose files are added to the target repository's tests folder. Until
the spec and test author exists, these hand-written tests stand in for its output.
"""

import argparse
import asyncio
import json
import time
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import (
    get_context_size,
    get_implementer_settings,
    get_model_name,
    get_server_config,
    get_target_repo_path,
    select_tools,
)
from pipeline.implementer import build_implementer, recursion_limit
from pipeline.tooling import open_tools


def load_task(task_folder):
    """Read a task folder into a dict with type, short_description, spec and a tests mapping.

    The tests mapping goes from the path the file will have inside the target
    repository, such as "tests/test_delete_book.py", to its full content.
    """
    folder = Path(task_folder)
    task = json.loads((folder / "task.json").read_text(encoding="utf-8"))
    tests_folder = folder / "tests"
    # Keys are repository-relative so the implementer can write them straight into the target's tests folder.
    # sorted() keeps the order stable between runs; a task without tests gets an empty mapping.
    task["tests"] = {
        f"tests/{path.name}": path.read_text(encoding="utf-8")
        for path in sorted(tests_folder.glob("*.py"))
    } if tests_folder.exists() else {}
    return task


async def main(task_folder):
    """Load the task, start the servers and the model, run the graph, and print the outcome."""
    # Imported here so the servers-only code paths do not need Ollama's client installed.
    from langchain_ollama import ChatOllama

    task = load_task(task_folder)
    repo_path = get_target_repo_path()
    settings = get_implementer_settings()
    server_config = get_server_config()
    # temperature=0 makes runs as repeatable as the model allows; num_ctx sets the context window Ollama allocates.
    model = ChatOllama(model=get_model_name(), temperature=0, num_ctx=get_context_size())

    # The exit stack owns the server sessions; they all close when this block ends, even on error.
    async with AsyncExitStack() as exit_stack:
        # Every tool from every server, for pipeline code; the model only gets the allowlisted subset.
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        agent_tools = select_tools(all_tools, "implementer")
        graph = build_implementer(repo_path, all_tools, agent_tools, model, settings)
        print(f"Task: {task['type']}/{task['short_description']} on {repo_path}")
        print(f"Model: {get_model_name()}, up to {settings['max_attempts']} attempts of {settings['max_tool_steps']} tool steps\n")
        started = time.perf_counter()
        # The recursion limit is sized to the settings so a long run is not cut off by LangGraph's default of 25.
        state = await graph.ainvoke({"task": task}, {"recursion_limit": recursion_limit(settings)})
        minutes = (time.perf_counter() - started) / 60

    # The summary is printed after the servers have shut down, so it is the last thing on screen.
    print(f"\nResult: {state['status'].upper()} after {state['attempt']} attempt(s) in {minutes:.1f} min")
    print(f"Branch: {state['branch']}")
    # Only a passed run has a commit; a blocked run leaves its changes uncommitted on the branch.
    if state["status"] == "passed":
        print(f"Commit: {state['commit_message']}")
    # Reflections show what the model learned between attempts, which is useful evidence for the report.
    for number, reflection in enumerate(state.get("reflections", []), 1):
        print(f"Lesson from attempt {number}: {reflection}")


if __name__ == "__main__":
    # The first line of the module docstring doubles as the command's help text.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task_folder", help="folder containing task.json and a tests folder")
    arguments = parser.parse_args()
    # asyncio.run owns the event loop that the MCP sessions and the graph run on.
    asyncio.run(main(arguments.task_folder))
