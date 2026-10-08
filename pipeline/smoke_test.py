"""Smoke test for the model and the MCP servers.

Run before building the agents, to find out whether the chosen model can call
tools reliably through this setup.

    python -m pipeline.smoke_test             # servers and model
    python -m pipeline.smoke_test --no-model  # servers only

Each check gives the model a small task, runs the tools it asks for, and passes
only if the model used the required tool and the final answer contains
the expected fact.
"""

import argparse
import asyncio
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import get_model_name, get_server_config, get_target_repo_path, select_tools
from pipeline.tooling import innermost_error

# Model turns allowed per check before it is marked as failed for never answering.
MAX_STEPS = 5


def build_checks():
    """Each check names a tool the model must use, so a lucky guess cannot pass."""
    # The expected facts are specific to the reading_list repository: app.py exists, its suite has 6 tests,
    # and add_book returns 400 without a title.
    return [
        {
            "name": "single tool call",
            "task": "Show me the files and folders in the repository.",
            "required_tool": "list_directory",
            "expected": "app.py",
        },
        {
            "name": "tool result round trip",
            "task": "Run the test suite and tell me how many tests passed.",
            "required_tool": "run_tests",
            "expected": "6",
        },
        {
            "name": "argument filling",
            "task": (
                "Read app.py and tell me the HTTP status code that add_book "
                "returns when the request has no title."
            ),
            "required_tool": "read_text_file",
            "expected": "400",
        },
    ]


async def run_check(model, tools, system_prompt, check):
    """Let the model work on one task, executing its tool calls, and grade the answer."""
    tools_by_name = {tool.name: tool for tool in tools}
    messages = [SystemMessage(system_prompt), HumanMessage(check["task"])]
    # Every tool the model calls is recorded, so the grade can require the right one.
    calls_made = []
    started = time.perf_counter()
    try:
        # A hand-rolled ReAct loop: ask the model, run what it asks for, repeat until it answers in text.
        for _ in range(MAX_STEPS):
            response = await model.ainvoke(messages)
            messages.append(response)
            # No tool calls means the model has given its final answer.
            if not response.tool_calls:
                answer = str(response.content)
                used_required_tool = check["required_tool"] in calls_made
                # Both conditions must hold: the right tool was used and the answer contains the fact.
                passed = used_required_tool and check["expected"].lower() in answer.lower()
                return passed, calls_made, answer, time.perf_counter() - started
            for tool_call in response.tool_calls:
                calls_made.append(tool_call["name"])
                tool = tools_by_name.get(tool_call["name"])
                # A hallucinated tool name is a failure worth seeing, not something to skip quietly.
                if tool is None:
                    raise RuntimeError(f"Model called an unknown tool: {tool_call['name']}")
                # Passing the whole tool_call makes the adapter return a ToolMessage with the matching call id.
                messages.append(await tool.ainvoke(tool_call))
        return False, calls_made, f"No final answer after {MAX_STEPS} steps", time.perf_counter() - started
    # Any error, including the unknown-tool one above, is reported as that check failing rather than ending the run.
    except Exception as error:
        return False, calls_made, f"Error: {error}", time.perf_counter() - started


async def check_servers(client, server_names):
    """Connect to each server on its own, so a failure names the server responsible."""
    all_tools = []
    failed = []
    for name in server_names:
        try:
            # get_tools with a server name starts only that server, so a failure is attributable.
            tools = await client.get_tools(server_name=name)
        except BaseException as error:  # MCP failures arrive as exception groups
            failed.append(name)
            print(f"[FAIL] {name}: {summarise_error(error)}")
            continue
        all_tools.extend(tools)
        print(f"[OK]   {name}: {len(tools)} tools")
    return all_tools, failed


def summarise_error(error):
    """Dig the innermost message out of nested exception groups."""
    # The same unwrapping the orchestrator uses for a crashed stage.
    return innermost_error(error)


async def main(use_model):
    """Check every server connects and the allowlist resolves, then optionally run the model checks."""
    repo_path = get_target_repo_path()
    server_config = get_server_config()
    client = MultiServerMCPClient(server_config)
    all_tools, failed = await check_servers(client, list(server_config))
    # Stop with the exact launch command for each failed server, so it can be run by hand for its own error.
    if failed:
        print(f"\n{len(failed)} server(s) failed to start: {', '.join(failed)}.")
        print("Run the failing server's command by hand to see its own error message:")
        for name in failed:
            settings = server_config[name]
            print(f"  {name}: {settings['command']} {' '.join(settings['args'])}")
        return
    print(f"Connected. {len(all_tools)} tools available across all servers.")
    # Resolving the allowlist here catches a renamed tool before any agent depends on it.
    tools = select_tools(all_tools, "implementer")
    print(f"Implementer allowlist resolved: {len(tools)} tools.")
    if not use_model:
        return

    # Imported here so the servers-only path does not need Ollama's client installed.
    from langchain_ollama import ChatOllama

    model_name = get_model_name()
    # The model gets the implementer's allowlist, the same tools the real agent will have.
    model = ChatOllama(model=model_name, temperature=0).bind_tools(tools)
    system_prompt = (
        f"You work on the repository at {repo_path}. "
        "Always use absolute paths inside that folder. "
        "Use tools to find facts; do not guess."
    )
    print(f"\nModel: {model_name}\n")
    results = []
    for check in build_checks():
        passed, calls, answer, seconds = await run_check(model, tools, system_prompt, check)
        results.append(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {check['name']} ({seconds:.1f}s)")
        print(f"  tools called: {calls or 'none'}")
        # The answer is cut to 300 characters so a verbose model does not swamp the summary.
        print(f"  answer: {answer[:300]}\n")
    print(f"{sum(results)} of {len(results)} checks passed.")


if __name__ == "__main__":
    # The first line of the module docstring doubles as the command's help text.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-model", action="store_true", help="only check that the servers connect")
    arguments = parser.parse_args()
    asyncio.run(main(use_model=not arguments.no_model))
