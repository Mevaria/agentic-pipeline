"""Spec and test author agent.

Turns a plain-English change request or bug report into a task folder the
implementer can run: a scoped spec and tests, written before any
implementation exists. The graph mixes model steps with code steps:

    prepare   (code)   refuse a dirty target, check out the base branch, start the conversation
    author    (model)  ReAct loop: read the code, write the spec and the tests
    tools     (code)   execute the tool calls the model asked for; a clarification ends the attempt
    check     (code)   fail-first check: every new test must fail on an assertion
    revise    (code)   restart the author with the check's findings, up to the revision cap
    finish    (code)   the task folder is ready for the implementer
    clarify, blocked, not_reproducible (code)  the other ways a run ends

The check is done in code, so the model cannot skip it or judge its own tests.
The author never has write access to the target repository: it reads the code
through the Filesystem server and writes only into its own task folder through
the dev tools server. If the request cannot be specified as given, the author
calls request_clarification with a reason and one question instead of writing
anything, and the run ends with that question so the developer can rerun with
a clearer request. That tool lives here rather than on an MCP server, because
it has no side effect: it only signals the graph.
"""

import json
import shutil
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode

from pipeline.tasks import load_task
from pipeline.tooling import call_tool, latest_tool_results, record_tool_results, scrub_secrets

# Folder inside the target repository that holds its tests, which the author must read but never write.
TESTS_DIR = "tests"
# Characters of a test's check detail passed back to the model, enough for the failure line.
DETAIL_LIMIT = 300
# Characters of each tool result written to the log, so a run shows what the tools said without flooding it.
TOOL_LOG_LIMIT = 200
# Task type under which tests that pass mean the reported bug could not be reproduced.
BUG_TYPE = "fix"
# The only reasons the author may give for asking a question instead of writing a spec.
CLARIFICATION_REASONS = {
    "behaviour_not_testable": "the request gives no observable outcome to test",
    "contradicts_existing": "the request conflicts with existing code or tests",
    "unknown_reference": "the request refers to something that does not exist in the code",
    "too_broad": "the request is too large for one change",
    "duplicate_of_open_pr": "an open pull request already implements a near-identical request",
}


@tool
def request_clarification(
    reason: Literal["behaviour_not_testable", "contradicts_existing", "unknown_reference", "too_broad", "duplicate_of_open_pr"],
    question: str,
) -> str:
    """Ask the developer one question instead of writing a spec, when the request cannot be specified as given.

    reason: behaviour_not_testable (no observable outcome to test), contradicts_existing
    (conflicts with existing code or tests), unknown_reference (refers to something not in
    the code), too_broad (too large for one change) or duplicate_of_open_pr (an open pull
    request listed in the request already implements a near-identical request). question:
    the one question whose answer would let you write the spec, or for a duplicate, whether
    to revise that pull request instead. Call this at most once, then reply with a one-sentence summary.
    """
    # The call itself is the signal; the check node reads its arguments from the conversation.
    return "Clarification recorded. Reply with a one-sentence summary and no further tool calls."


def clarification_request(messages):
    """Return (reason, question) from a valid request_clarification call in the attempt, or None."""
    # Only calls with a listed reason count; an invalid reason was already refused by the tool's schema,
    # and the model saw that error, so it is not treated as a request.
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["name"] == "request_clarification" and call["args"].get("reason") in CLARIFICATION_REASONS:
                    return call["args"]["reason"], str(call["args"].get("question", "")).strip()
    return None


def clarification_answered(messages):
    """Return True when the latest tool results include a request_clarification call the tool accepted."""
    # A call with a bad reason comes back with status "error", and must not end the attempt.
    return any(
        result.name == "request_clarification" and result.status != "error"
        for result in latest_tool_results(messages)
    )


class AuthorState(TypedDict, total=False):
    """State carried through the graph.

    total=False lets prepare fill the fields in: the run starts with only the
    request. messages uses the add_messages reducer, so a node returns the
    messages to append (or RemoveMessage to clear) rather than the whole list.
    """

    # The change request or bug report, as plain text.
    request: str
    # Where the author's tools write task.json and the tests for this run.
    task_folder: str
    # 0 on the first attempt, incremented by revise.
    revision: int
    # The check's findings for each revision, as text given to the model.
    feedback: list
    # The conversation for the current attempt; cleared and rebuilt at the start of each attempt.
    messages: Annotated[list, add_messages]
    # The latest result from check_tests_fail, or a stand-in error when nothing could be checked.
    check_result: dict
    # The task as loaded from the folder once a spec exists.
    task: dict
    # "running", then "ready", "needs_clarification", "blocked" or "not_reproducible".
    status: str
    # The clarifying question and its reason, when the run ends with needs_clarification.
    question: str
    clarification_reason: str


def author_system_prompt(repo_path):
    """Return the system prompt that sets the author's role and rules for every attempt."""
    return (
        f"You are the spec and test author on a small Python Flask project at {repo_path}.\n"
        "You receive a change request or a bug report. Your job is to write a scoped spec and "
        "tests that define the change, before anyone implements it. You never change the application code.\n"
        "Rules:\n"
        f"- First read the code: list {repo_path}, then read app.py and the files in its {TESTS_DIR} folder, "
        "so your tests fit the app and match the style of the existing tests.\n"
        f"- Use absolute paths inside {repo_path} when reading.\n"
        "- Call write_task_spec once, with all three arguments: task_type is feat for new behaviour or fix for a "
        "bug, short_description is a slug like delete-book, and spec says exactly what must change and what "
        "must stay the same.\n"
        "- Call write_task_test for each test file, with a new file name. The tests must fail now and pass "
        "once the change is made. Reach the app only through its Flask test client, assert the status code "
        "before reading a response body, and never import a name that does not exist yet.\n"
        "- For a bug report, the first test reproduces the bug exactly as reported.\n"
        "- If the request cannot be specified as given, write nothing and call request_clarification "
        "with the reason and one question.\n"
        "- If the request lists an open pull request from a past run that already implements a near-identical "
        "request, write nothing and call request_clarification with reason duplicate_of_open_pr, naming that "
        "pull request and asking whether to revise it instead.\n"
        "- When the spec and tests are written, reply with a one-sentence summary and no tool calls."
    )


def request_prompt(request):
    """The request as given to the model on the first attempt."""
    return f"Request:\n{request}"


def describe_findings(result):
    """Turn a check result into one line per test, or the check's error when it could not run."""
    if result.get("error"):
        return result["error"]
    # Each line names the test, its status and why, so the model knows which file to rewrite and how.
    return "\n".join(
        f"- {entry['test']}: {entry['status']}. {entry['detail'][:DETAIL_LIMIT]}"
        for entry in result.get("tests", [])
    )


def revision_prompt(request, task, findings):
    """The request, what was written so far, and the check's findings, for the next attempt."""
    prompt = f"{request_prompt(request)}\n\n"
    # The earlier spec and tests are repeated in full, since the conversation that wrote them is gone.
    if task:
        prompt += f"Your spec so far (type {task['type']}, {task['short_description']}):\n{task['spec']}\n\n"
        for path, content in task["tests"].items():
            prompt += f"Your test file {Path(path).name}:\n{content}\n\n"
    prompt += (
        "The tests were run on the current code before any change. Every test must fail on an assertion, "
        "because the change is not made yet. These did not:\n"
        f"{findings}\n\n"
        "Fix this: rewrite the affected test file with write_task_test using the same file name, or "
        "write the spec and tests if they are missing. Tests marked passes test behaviour that already exists. "
        "Tests marked broken fail for the wrong reason, such as an import or a typo."
    )
    return prompt


def build_author(repo_path, all_tools, agent_tools, model, settings, task_folder, log=print):
    """Build the author graph.

    all_tools: every MCP tool, used by pipeline code at fixed points.
    agent_tools: the author's allowlist, the only tools the model can call.
    settings: base_branch, max_revisions, max_tool_steps.
    task_folder: where the author's tools write for this run, TASKS_PATH/RUN_ID.
    """
    repo = str(Path(repo_path))
    task_folder = Path(task_folder)
    tools_by_name = {tool.name: tool for tool in all_tools}

    async def git(name, **arguments):
        """Call a Git server tool; every one of them takes the repository path as repo_path."""
        return await call_tool(tools_by_name, name, {"repo_path": repo, **arguments})

    # The clarification tool is defined in this module, not on a server, so it is added to the allowlist here.
    agent_tools = [*agent_tools, request_clarification]
    # Binding the allowlist is what lets the model emit tool calls, and only for these tools.
    model_with_tools = model.bind_tools(agent_tools)
    system_prompt = author_system_prompt(repo)

    def fresh_attempt_messages(prompt):
        """Return the messages that start an attempt: clear the history, then system prompt and task."""
        return [
            # add_messages treats this as "delete everything so far", so each attempt starts with a clean context.
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            SystemMessage(system_prompt),
            HumanMessage(prompt),
        ]

    async def prepare(state):
        """Code node: put the target on the base branch, so the author reads the code the check runs against."""
        status = await git("git_status")
        # Uncommitted changes would make the author describe code that is not on any branch.
        if "working tree clean" not in status:
            raise RuntimeError(f"The target repository has uncommitted changes:\n{status}")
        # The fail-first check runs on the base branch; whatever branch happened to be checked out must not
        # be what the author reads, or it could specify against code that is not in the base.
        await git("git_checkout", branch_name=settings["base_branch"])
        log(f"[prepare] on branch {settings['base_branch']}, task folder {task_folder}")
        return {
            "task_folder": str(task_folder),
            "revision": 0,
            "feedback": [],
            "messages": fresh_attempt_messages(request_prompt(state["request"])),
            "status": "running",
        }

    async def author(state):
        """Model node: one ReAct step, producing either tool calls or a final answer."""
        # What the tools answered since the last step, so a run's log shows refusals and errors, not only call names.
        for result in latest_tool_results(state["messages"]):
            log(f"[tool] {result.name} ({result.status}): {scrub_secrets(result.content)[:TOOL_LOG_LIMIT]}")
        # The full calls and results go to the run's tool log, so the evidence never depends on what was printed.
        record_tool_results(state["messages"])
        response = await model_with_tools.ainvoke(state["messages"])
        # Log the tool names so a run can be followed without printing the whole conversation.
        for tool_call in response.tool_calls:
            log(f"[author] revision {state['revision']}: {tool_call['name']}")
        return {"messages": [response]}

    def route_after_author(state):
        """Send the model's tool calls to the tools node, or go to check when it stops or hits the step limit."""
        last = state["messages"][-1]
        # Every AIMessage in the current attempt is one step, since the history is cleared between attempts.
        steps = sum(isinstance(message, AIMessage) for message in state["messages"])
        if last.tool_calls and steps < settings["max_tool_steps"]:
            return "tools"
        # Reaching the limit with calls still pending means the model never finished; check judges what it has.
        if last.tool_calls:
            log(f"[author] revision {state['revision']}: step limit reached")
        return "check"

    def route_after_tools(state):
        """Go straight to check once a clarification was accepted, so the model gets no further turns."""
        if clarification_answered(state["messages"]):
            # Logged here because the author node, which normally logs tool results, is skipped.
            log(f"[tools] request_clarification accepted; ending the attempt")
            return "check"
        return "author"

    async def check(state):
        """Code node: load what the author wrote and run the fail-first check on its tests."""
        # A valid request_clarification call ends the run, whatever else was written.
        clarification = clarification_request(state["messages"])
        if clarification:
            reason, question = clarification
            # Anything written before the question is incomplete by the author's own account, and a folder
            # that looks usable must not be left for the implementer. The question is kept in the state.
            if task_folder.exists():
                shutil.rmtree(task_folder, ignore_errors=True)
                log(f"[check] removed the partial task folder {task_folder}")
            log(f"[check] the author asked for clarification ({reason})")
            return {"status": "needs_clarification", "clarification_reason": reason, "question": question}
        # No spec and no clarification means the author wrote nothing useful, which is fed back as a failed attempt.
        if not (task_folder / "task.json").exists():
            return {"check_result": {"all_red": False, "tests": [],
                                     "error": "No spec was written. Call write_task_spec, then write_task_test."}}
        task = load_task(task_folder)
        if not task["tests"]:
            return {"task": task, "check_result": {"all_red": False, "tests": [],
                                                   "error": "No test file was written. Call write_task_test."}}
        # The check runs in code, in a throwaway worktree, and classifies each test by what it raised.
        result = json.loads(await call_tool(tools_by_name, "check_tests_fail", {"tests": task["tests"]}))
        summary = ", ".join(f"{entry['test'].split('::')[-1]}={entry['status']}" for entry in result.get("tests", []))
        log(f"[check] revision {state['revision']}: {'all red' if result.get('all_red') else 'not all red'} ({summary or result.get('error', '')})")
        return {"task": task, "check_result": result}

    def route_after_check(state):
        """Finish on all red, end on a question, revise while revisions remain, otherwise stop."""
        if state.get("status") == "needs_clarification":
            return "clarify"
        result = state["check_result"]
        if result.get("all_red"):
            return "finish"
        if state["revision"] < settings["max_revisions"]:
            return "revise"
        # Out of revisions. For a bug report whose every test passes, the bug could not be reproduced.
        task = state.get("task") or {}
        tests = result.get("tests", [])
        if task.get("type") == BUG_TYPE and tests and all(entry["status"] == "passes" for entry in tests):
            return "not_reproducible"
        return "blocked"

    def revise(state):
        """Code node: start the next attempt with the request, what was written, and the check's findings."""
        findings = describe_findings(state["check_result"])
        log(f"[revise] starting revision {state['revision'] + 1}")
        return {
            "revision": state["revision"] + 1,
            "feedback": state["feedback"] + [findings],
            # The conversation is rebuilt from scratch, so the failed attempt's reads and writes do not fill the context.
            "messages": fresh_attempt_messages(revision_prompt(state["request"], state.get("task"), findings)),
        }

    def finish(state):
        """Code node: every test is red, so the task folder is ready for the implementer."""
        task = state["task"]
        log(f"[finish] {task['type']}/{task['short_description']} with {len(task['tests'])} test file(s) ready in {task_folder}")
        return {"status": "ready"}

    def clarify(state):
        """Code node: the author asked a question; the run ends so the developer can answer by rerunning."""
        log(f"[clarify] {state['clarification_reason']}: {state['question']}")
        return {"status": "needs_clarification"}

    def blocked(state):
        """Code node: the revision cap was reached without every test being red."""
        log(f"[blocked] tests still not all red after {state['revision'] + 1} attempt(s)")
        return {"status": "blocked"}

    def not_reproducible(state):
        """Code node: every reproduction test for a bug report passed on the current code."""
        log(f"[not_reproducible] every test passed on the current code after {state['revision'] + 1} attempt(s)")
        return {"status": "not_reproducible"}

    graph = StateGraph(AuthorState)
    graph.add_node("prepare", prepare)
    graph.add_node("author", author)
    # ToolNode runs the calls in the last AIMessage; handle_tool_errors turns a failing tool into an error
    # message the model sees, instead of an exception that ends the run.
    graph.add_node("tools", ToolNode(agent_tools, handle_tool_errors=True))
    graph.add_node("check", check)
    graph.add_node("revise", revise)
    graph.add_node("finish", finish)
    graph.add_node("clarify", clarify)
    graph.add_node("blocked", blocked)
    graph.add_node("not_reproducible", not_reproducible)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "author")
    # The ReAct loop: author -> tools -> author, until the router sends the attempt to check,
    # or an accepted clarification ends it from the tools side.
    graph.add_conditional_edges("author", route_after_author, ["tools", "check"])
    graph.add_conditional_edges("tools", route_after_tools, ["author", "check"])
    # The revision loop: check -> revise -> author, until all red, a question, or the cap.
    graph.add_conditional_edges("check", route_after_check, ["finish", "clarify", "revise", "blocked", "not_reproducible"])
    graph.add_edge("revise", "author")
    for terminal in ("finish", "clarify", "blocked", "not_reproducible"):
        graph.add_edge(terminal, END)
    return graph.compile()


def recursion_limit(settings):
    """Enough graph steps for every attempt to use its full tool budget."""
    # Each tool step costs two graph steps (author and tools); the extra four cover check, revise and the routing.
    per_attempt = 2 * settings["max_tool_steps"] + 4
    # The first attempt plus one per allowed revision, and ten more for prepare, the terminal node and slack.
    return (settings["max_revisions"] + 1) * per_attempt + 10
