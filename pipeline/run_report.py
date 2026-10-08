"""Run the review gate on a branch and, if it approves, push the branch and open the pull request.

    python -m pipeline.run_report runs/tasks/20261006-150935 fix/reject-whitespace-title

The task folder is the one the branch was built from. The GitHub MCP server
must be configured: GITHUB_MCP_SERVER and GITHUB_PERSONAL_ACCESS_TOKEN in .env.
"""

import argparse
import asyncio
import time
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import (
    get_context_size,
    get_github_server_config,
    get_implementer_settings,
    get_model_name,
    get_reporter_settings,
    get_review_settings,
    get_server_config,
    get_target_repo_path,
    select_tools,
)
from pipeline.gate import run_gate
from pipeline.reporter import report
from pipeline.reviewer import describe_findings
from pipeline.tasks import load_task
from pipeline.tooling import open_tools


async def main(task_folder, branch):
    """Start every server including GitHub's, run the gate, then the reporter, and print the outcome."""
    # Imported here so the servers-only code paths do not need Ollama's client installed.
    from langchain_ollama import ChatOllama

    task = load_task(task_folder)
    repo_path = get_target_repo_path()
    # The GitHub server is added only here; the other runners never start it.
    server_config = {**get_server_config(), **get_github_server_config()}
    model = ChatOllama(model=get_model_name(), temperature=0, num_ctx=get_context_size())
    # A generated task folder is named by its run id, which also names the record.
    run_id = Path(task_folder).name

    async with AsyncExitStack() as exit_stack:
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        implementer_tools = select_tools(all_tools, "implementer")
        print(f"Review and report of {branch} for {task['type']}/{task['short_description']} on {repo_path}\n")
        started = time.perf_counter()
        outcome = await run_gate(repo_path, all_tools, implementer_tools, model, model, task, branch,
                                 get_implementer_settings(), get_review_settings())
        record = await report(repo_path, all_tools, model, task, branch, outcome["review"], run_id, get_reporter_settings())
        minutes = (time.perf_counter() - started) / 60

    print(f"\nGate: {outcome['status'].upper()} after {len(outcome['rounds'])} review round(s)")
    print(f"Reporter: {record['outcome'].upper()} in {minutes:.1f} min")
    if record["steps"]["pull_request"]:
        print(f"Pull request: {record['steps']['pull_request']['url']}")
    if record["error"]:
        print(f"Error: {record['error']}")
    if outcome["review"]["non_blocking_findings"]:
        print("Non-blocking findings in the pull request body:\n" + describe_findings(outcome["review"]["non_blocking_findings"]))
    print(f"Record: {get_reporter_settings()['records_path'] / (run_id + '.json')}")


if __name__ == "__main__":
    # The first line of the module docstring doubles as the command's help text.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task_folder", help="folder containing task.json and the tests the branch was built from")
    parser.add_argument("branch", help="the branch to review and report")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.task_folder, arguments.branch))
