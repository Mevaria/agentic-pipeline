"""Run the review gate on a branch the implementer produced.

    python -m pipeline.run_review runs/tasks/20261006-150935 fix/reject-whitespace-title

The task folder is the one the branch was built from, so the reviewer can judge
scope against the spec and the tests. Blocking findings go back to the
implementer on the same branch, up to MAX_REVIEW_ROUNDS reviews.
"""

import argparse
import asyncio
import time
from contextlib import AsyncExitStack

from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import (
    get_context_size,
    get_implementer_settings,
    get_model_name,
    get_review_settings,
    get_server_config,
    get_target_repo_path,
    select_tools,
)
from pipeline.gate import run_gate
from pipeline.reviewer import describe_findings
from pipeline.tasks import load_task
from pipeline.tooling import open_tools


async def main(task_folder, branch):
    """Start the servers and the model, run the gate on the branch, and print every round's verdict."""
    # Imported here so the servers-only code paths do not need Ollama's client installed.
    from langchain_ollama import ChatOllama

    task = load_task(task_folder)
    repo_path = get_target_repo_path()
    server_config = get_server_config()
    # One model instance serves both roles; the prompts, not the weights, make the reviewer and the implementer differ.
    model = ChatOllama(model=get_model_name(), temperature=0, num_ctx=get_context_size())

    async with AsyncExitStack() as exit_stack:
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        implementer_tools = select_tools(all_tools, "implementer")
        print(f"Review of {branch} for {task['type']}/{task['short_description']} on {repo_path}")
        print(f"Model: {get_model_name()}, up to {get_review_settings()['max_review_rounds']} review round(s)\n")
        started = time.perf_counter()
        outcome = await run_gate(repo_path, all_tools, implementer_tools, model, model, task, branch,
                                 get_implementer_settings(), get_review_settings())
        minutes = (time.perf_counter() - started) / 60

    print(f"\nResult: {outcome['status'].upper()} after {len(outcome['rounds'])} review round(s) in {minutes:.1f} min")
    for number, review in enumerate(outcome["rounds"], 1):
        print(f"\nRound {number}: {'approved' if review['approved'] else 'changes requested'}")
        if review["blocking_findings"]:
            print("Blocking:\n" + describe_findings(review["blocking_findings"]))
        if review["non_blocking_findings"]:
            print("Non-blocking:\n" + describe_findings(review["non_blocking_findings"]))
        if review["summary"]:
            print(f"Summary: {review['summary']}")
    for number, fix in enumerate(outcome["fixes"], 1):
        print(f"\nFix {number}: {fix['status']} after {fix['attempt']} attempt(s)"
              + (f", commit: {fix['commit_message']}" if fix["status"] == "passed" else ""))


if __name__ == "__main__":
    # The first line of the module docstring doubles as the command's help text.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task_folder", help="folder containing task.json and the tests the branch was built from")
    parser.add_argument("branch", help="the branch to review")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.task_folder, arguments.branch))
