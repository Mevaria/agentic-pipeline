"""Implementer agent.

Changes code until the given tests pass, or stops as blocked after a fixed
number of attempts. The graph mixes model steps with code steps:

    prepare  (code)   branch from the base branch, or continue on an existing branch, and add the given tests
    agent    (model)  ReAct loop: reason, call a tool, observe the result
    tools    (code)   execute the tool calls the model asked for
    verify   (code)   run the tests independently and check the tests were not edited
    reflect  (model)  Reflexion: explain the failure, carried into the next attempt
    finish   (code)   format the changed files, then commit with a message the model writes
    blocked  (code)   stop once the attempt limit is reached

Branching, test runs used for the decision, formatting and committing are done
in code, so the model cannot skip them, fake them, or do them at the wrong time.
A task may carry "branch" and "review_findings": then the implementer continues
on that branch to address a reviewer's blocking findings instead of starting
a new one.
"""

import json
import re
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

from pipeline.tooling import call_tool, latest_tool_results, scrub_secrets, traced_tool_node

# Folder inside the target repository that holds the tests; everything in it is protected from the model.
TESTS_DIR = "tests"
# Characters of each tool result written to the log, so a run shows what the tools said without flooding it.
TOOL_LOG_LIMIT = 200
# Commit types accepted by the Conventional Commits check in finish.
COMMIT_TYPES = "feat|fix|refactor|test|docs|chore|perf|style|build|ci"
# "<type>(<optional scope>): <description>" on one line, with the description kept under 72 characters.
COMMIT_PATTERN = re.compile(rf"^({COMMIT_TYPES})(\([a-z0-9-]+\))?: \S.{{0,70}}$")
# Characters of test output handed to the reflection step, so a long traceback does not fill the context.
TEST_OUTPUT_LIMIT = 2000


class ImplementerState(TypedDict, total=False):
    """State carried through the graph.

    total=False lets prepare fill the fields in: the run starts with only the
    task. messages uses the add_messages reducer, so a node returns the
    messages to append (or RemoveMessage to clear) rather than the whole list.
    """

    # The task as loaded by run_implementer: type, short_description, spec and the tests mapping.
    task: dict
    # Branch created by prepare, where all changes land.
    branch: str
    # 1-based attempt counter, incremented by reflect.
    attempt: int
    # One reflection per failed attempt, fed into the next attempt's prompt.
    reflections: list
    # Path to original content for every file in the tests folder, used by verify to detect tampering.
    protected_files: dict
    # The conversation for the current attempt; cleared and rebuilt at the start of each attempt.
    messages: Annotated[list, add_messages]
    # The result dict from run_tests, as decided by verify.
    test_result: dict
    # "running", then "passed" or "blocked".
    status: str
    # The Conventional Commits message used by finish.
    commit_message: str
    # Whatever the git server reported back from the commit.
    commit_result: str
    # What the formatter did in finish: the files it rewrote and any whitespace problem left.
    format_result: dict


def implementer_system_prompt(repo_path):
    """Return the system prompt that sets the model's role and rules for every attempt."""
    return (
        f"You are the implementer on a small Python Flask project at {repo_path}.\n"
        "Your job is to change the code so that the tests pass.\n"
        "Rules:\n"
        f"- Use absolute paths inside {repo_path}.\n"
        "- Read a file before you change it.\n"
        "- write_file replaces the whole file, so always write the complete file content.\n"
        f"- Never create, edit or delete anything in the {TESTS_DIR} folder. "
        "The tests define what done means.\n"
        "- After changing code, call run_tests to check your work.\n"
        "- When the tests pass, reply with a one-sentence summary and no tool calls."
    )


def attempt_prompt(task, reflections):
    """The task as given to the model at the start of each attempt."""
    # The model is told which files must pass, not their contents; it can read them with its tools.
    test_files = ", ".join(task["tests"]) or "none"
    prompt = f"Task:\n{task['spec']}\n\nTests that must pass: {test_files}"
    # In continue mode the branch already holds an earlier change; the reviewer's findings say what to fix.
    if task.get("review_findings"):
        prompt += (
            "\n\nYour earlier change on this branch was reviewed. Fix these findings without breaking the tests:\n"
            f"{task['review_findings']}"
        )
    # Reflexion: lessons from earlier attempts are the only memory that survives between attempts.
    if reflections:
        lessons = "\n".join(f"- Attempt {number}: {text}" for number, text in enumerate(reflections, 1))
        prompt += f"\n\nEarlier attempts failed. Lessons from them:\n{lessons}"
    return prompt


def parse_listing(listing):
    """Turn list_directory output into file names, skipping folders."""
    # The Filesystem server prints one entry per line, prefixed with "[FILE] " or "[DIR] ".
    return [line[len("[FILE] "):].strip() for line in listing.splitlines() if line.startswith("[FILE] ")]


def parse_branches(listing):
    """Turn git_branch output into a set of branch names, dropping the marker on the current branch."""
    # The git server marks the checked-out branch with "*", which is not part of its name.
    return {line.replace("*", "").strip() for line in listing.splitlines() if line.strip()}


def build_implementer(repo_path, all_tools, agent_tools, model, settings, log=print):
    """Build the implementer graph.

    all_tools: every MCP tool, used by pipeline code at fixed points.
    agent_tools: the implementer's allowlist, the only tools the model can call.
    settings: base_branch, max_attempts, max_tool_steps.
    """
    repo_path = Path(repo_path)
    # The MCP servers take the repository as a plain string argument.
    repo = str(repo_path)
    tools_by_name = {tool.name: tool for tool in all_tools}
    # Binding the allowlist is what lets the model emit tool calls, and only for these tools.
    model_with_tools = model.bind_tools(agent_tools)
    system_prompt = implementer_system_prompt(repo)

    def path_in_repo(relative_path):
        """Return the absolute path of a repository-relative path, as the Filesystem server requires."""
        return str(repo_path / relative_path)

    async def git(name, **arguments):
        """Call a Git server tool; every one of them takes the repository path as repo_path."""
        return await call_tool(tools_by_name, name, {"repo_path": repo, **arguments})

    async def files(name, **arguments):
        """Call a Filesystem server tool with the given arguments."""
        return await call_tool(tools_by_name, name, arguments)

    def fresh_attempt_messages(task, reflections):
        """Return the messages that start an attempt: clear the history, then system prompt and task."""
        return [
            # add_messages treats this as "delete everything so far", so each attempt starts with a clean context.
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            SystemMessage(system_prompt),
            HumanMessage(attempt_prompt(task, reflections)),
        ]

    async def prepare(state):
        """Code node: check the repo is clean, create or continue the branch, add the given tests and record protected files."""
        task = state["task"]
        status = await git("git_status")
        # Starting on a dirty tree would mix someone else's edits into this run's commit.
        if "working tree clean" not in status:
            raise RuntimeError(f"The target repository has uncommitted changes:\n{status}")
        existing = parse_branches(await git("git_branch", branch_type="local"))

        if task.get("branch"):
            # Continue mode: the review gate sends the implementer back to the branch it already built.
            branch = task["branch"]
            if branch not in existing:
                raise RuntimeError(f"Cannot continue on branch {branch}: it does not exist")
            await git("git_checkout", branch_name=branch)
            log(f"[prepare] continuing on branch {branch}")
        else:
            await git("git_checkout", branch_name=settings["base_branch"])
            # Branch names follow <type>/<short-description>, as CONVENTIONS.md in the target repository requires.
            base_name = f"{task['type']}/{task['short_description']}"
            # git_create_branch reports success even when the branch exists, so pick a free name first:
            # feat/x, then feat/x-2, feat/x-3 and so on.
            branch, suffix = base_name, 2
            while branch in existing:
                branch, suffix = f"{base_name}-{suffix}", suffix + 1
            await git("git_create_branch", branch_name=branch, base_branch=settings["base_branch"])
            await git("git_checkout", branch_name=branch)
            log(f"[prepare] on branch {branch}")

        # The given tests are written before the model starts, so it works against them from the first step.
        for relative_path, content in task["tests"].items():
            await files("write_file", path=path_in_repo(relative_path), content=content)
        # Snapshot every file in the tests folder, existing and given, so verify can detect and undo any edit.
        protected = {}
        for name in parse_listing(await files("list_directory", path=path_in_repo(TESTS_DIR))):
            relative_path = f"{TESTS_DIR}/{name}"
            protected[relative_path] = await files("read_text_file", path=path_in_repo(relative_path))
        log(f"[prepare] {len(task['tests'])} given test file(s) added, {len(protected)} test file(s) protected")

        return {
            "branch": branch,
            "attempt": 1,
            "reflections": [],
            "protected_files": protected,
            "messages": fresh_attempt_messages(task, []),
            "status": "running",
        }

    async def agent(state):
        """Model node: one ReAct step, producing either tool calls or a final answer."""
        # What the tools answered since the last step, so a run's log shows refusals and errors, not only call names.
        for result in latest_tool_results(state["messages"]):
            log(f"[tool] {result.name} ({result.status}): {scrub_secrets(result.content)[:TOOL_LOG_LIMIT]}")
        response = await model_with_tools.ainvoke(state["messages"])
        # Log the tool names so a run can be followed without printing the whole conversation.
        for tool_call in response.tool_calls:
            log(f"[agent] attempt {state['attempt']}: {tool_call['name']}")
        return {"messages": [response]}

    def route_after_agent(state):
        """Send the model's tool calls to the tools node, or go to verify when it stops or hits the step limit."""
        last = state["messages"][-1]
        # Every AIMessage in the current attempt is one step, since the history is cleared between attempts.
        steps = sum(isinstance(message, AIMessage) for message in state["messages"])
        if last.tool_calls and steps < settings["max_tool_steps"]:
            return "tools"
        # Reaching the limit with calls still pending means the model never finished; verify judges what it has.
        if last.tool_calls:
            log(f"[agent] attempt {state['attempt']}: step limit reached")
        return "verify"

    async def verify(state):
        """Code node: restore any edited test file, then run the tests and record the result."""
        tampered = []
        for relative_path, original in state["protected_files"].items():
            try:
                current = await files("read_text_file", path=path_in_repo(relative_path))
            # A deleted test file raises on read; treating it as None makes it count as tampered below.
            except Exception:
                current = None
            if current != original:
                tampered.append(relative_path)
                # Put the original back so the test run below checks the real tests.
                await files("write_file", path=path_in_repo(relative_path), content=original)
        # This run, not any run the model made itself, is the one that decides pass or fail.
        result = json.loads(await call_tool(tools_by_name, "run_tests", {}))
        # Editing the tests is a failed attempt even if the restored tests happen to pass.
        if tampered:
            result["passed"] = False
            result["tampered"] = tampered
            log(f"[verify] test files were modified and have been restored: {tampered}")
        log(f"[verify] attempt {state['attempt']}: {'passed' if result['passed'] else 'failed'} ({result.get('summary', '')})")
        return {"test_result": result}

    def route_after_verify(state):
        """Go to finish on a pass, reflect while attempts remain, or blocked once they run out."""
        if state["test_result"]["passed"]:
            return "finish"
        if state["attempt"] < settings["max_attempts"]:
            return "reflect"
        return "blocked"

    async def reflect(state):
        """Model node: Reflexion. Ask why the attempt failed and start the next attempt with that lesson."""
        result = state["test_result"]
        # Only the end of the output is sent, which is where pytest puts the failures and the summary.
        problem = f"Test output:\n{result.get('output', '')[-TEST_OUTPUT_LIMIT:]}"
        if result.get("tampered"):
            problem += f"\n\nThe attempt also edited protected test files, which is not allowed: {result['tampered']}"
        earlier = "\n".join(f"- {text}" for text in state["reflections"]) or "none"
        # The plain model is used here, without tools, so the reflection is text rather than more tool calls.
        response = await model.ainvoke([
            SystemMessage(
                "You review a failed attempt to change code. In two to four sentences, "
                "explain why it failed and what to do differently next time. Do not write code."
            ),
            HumanMessage(f"Task:\n{state['task']['spec']}\n\n{problem}\n\nEarlier lessons:\n{earlier}"),
        ])
        reflection = str(response.content).strip()
        reflections = state["reflections"] + [reflection]
        log(f"[reflect] {reflection}")
        return {
            "attempt": state["attempt"] + 1,
            "reflections": reflections,
            # The conversation is rebuilt from scratch with the lessons, so the failed attempt's context is gone.
            "messages": fresh_attempt_messages(state["task"], reflections),
        }

    async def finish(state):
        """Code node: format the changed files, have the model write a commit message, validate it, and commit."""
        task = state["task"]
        # The formatter owns layout: trailing whitespace, final newlines, quotes and blank lines are fixed here,
        # on the changed files outside the tests folder, so no reviewer ever has to police them.
        format_result = json.loads(await call_tool(tools_by_name, "format_code", {"fix": True}))
        if format_result.get("reformatted"):
            log(f"[finish] formatted {format_result['reformatted']}")
            # Formatting is meant to preserve behaviour; the tests are run again so that is checked, not assumed.
            recheck = json.loads(await call_tool(tools_by_name, "run_tests", {}))
            if not recheck["passed"]:
                log(f"[finish] tests failed after formatting; leaving the branch uncommitted ({recheck.get('summary', '')})")
                return {"status": "blocked", "format_result": format_result, "test_result": recheck}
        if format_result.get("error"):
            log(f"[finish] formatter error, committing unformatted: {format_result['error']}")
        status = await git("git_status")
        # In continue mode the branch already has a commit for the feature; this commit is about the revision.
        revising = bool(task.get("review_findings"))
        if revising:
            subject = ("This commit revises an earlier change on the same branch to address these findings. Describe "
                       f"what this revision changes, not the original feature:\n{task['review_findings']}\n\nTask:\n{task['spec']}")
        else:
            subject = f"Task:\n{task['spec']}"
        # The model writes the message; the code decides whether it is acceptable and does the commit.
        response = await model.ainvoke([
            SystemMessage(
                "Write one commit message in Conventional Commits format: "
                "type: short description, lowercase, under 72 characters, no trailing period. "
                "Reply with the message only."
            ),
            HumanMessage(f"Type: {task['type']}\n{subject}\n\nChanged files:\n{status}"),
        ])
        # Keep the first line only, after stripping whitespace and the backticks small models like to wrap it in.
        message = str(response.content).strip().strip("`").strip().splitlines()[0] if response.content else ""
        # A message that breaks the convention is replaced, not committed, so the history stays clean.
        if not COMMIT_PATTERN.match(message):
            log(f"[finish] model's commit message was not valid, using a fallback: {message!r}")
            description = task["short_description"].replace("-", " ")
            message = f"{task['type']}: {'revise ' if revising else ''}{description}"
        # Stage everything, including the given tests, so the branch carries the tests that define the change.
        await git("git_add", files=["."])
        commit_result = await git("git_commit", message=message)
        log(f"[finish] {commit_result}")
        return {"status": "passed", "commit_message": message, "commit_result": commit_result, "format_result": format_result}

    async def blocked(state):
        """Code node: record that the attempt limit was reached. Nothing is committed."""
        log(f"[blocked] tests still failing after {state['attempt']} attempt(s); changes left uncommitted on {state['branch']}")
        return {"status": "blocked"}

    graph = StateGraph(ImplementerState)
    graph.add_node("prepare", prepare)
    graph.add_node("agent", agent)
    # The traced node runs the calls in the last AIMessage like ToolNode, turns a failing tool into an error
    # message the model sees, and logs every call with its result to the run's tool log.
    graph.add_node("tools", traced_tool_node(agent_tools))
    graph.add_node("verify", verify)
    graph.add_node("reflect", reflect)
    graph.add_node("finish", finish)
    graph.add_node("blocked", blocked)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "agent")
    # The ReAct loop: agent -> tools -> agent, until the router sends the attempt to verify.
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "verify"])
    graph.add_edge("tools", "agent")
    # The attempt loop: verify -> reflect -> agent, until a pass or the attempt limit.
    graph.add_conditional_edges("verify", route_after_verify, ["finish", "reflect", "blocked"])
    graph.add_edge("reflect", "agent")
    graph.add_edge("finish", END)
    graph.add_edge("blocked", END)
    return graph.compile()


def recursion_limit(settings):
    """Enough graph steps for every attempt to use its full tool budget."""
    # Each tool step costs two graph steps (agent and tools); the extra four cover verify, reflect and the routing.
    per_attempt = 2 * settings["max_tool_steps"] + 4
    # The final ten cover prepare, finish or blocked, and some slack.
    return settings["max_attempts"] * per_attempt + 10
