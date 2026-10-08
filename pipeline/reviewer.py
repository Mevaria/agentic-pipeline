"""Security and scope reviewer.

Judges one branch's change against the target repository's review checklist
and returns a structured verdict. The graph mixes code steps with model steps:

    gather    (code)   check out the branch, collect the diff, the changed files, a fresh test
                       run, the security scan, a formatting report and the checklist
    reviewer  (model)  read the evidence and call submit_review with findings and a summary
    tools     (code)   execute the submit_review call
    remind    (code)   one nudge if the model answered without calling the tool
    verdict   (code)   validate the findings, merge in the code-decided ones, derive the outcome

The reviewer never sees the implementer's reasoning, only the evidence code
gathers. It has exactly one tool, submit_review, defined here because it has
no side effect: it only hands the findings to the graph. The verdict is not
something the model states; code derives approve or request changes from the
findings, so a verdict can never contradict them. Tests passing and the
security scan's blocking decision are settled by code before the model runs,
and formatting is handled by the formatter, so the model judges only input
validation, secrets and scope.
"""

import json
import re
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from pipeline.tooling import call_tool, latest_tool_results

# The checklist items the model judges. The other two items, tests and scan severity, are decided by code.
JUDGED_ITEMS = ("input_validation", "secrets", "scope")
# Used when the target has no CONVENTIONS.md, so the reviewer always has a checklist to apply.
DEFAULT_CHECKLIST = (
    "- Input from requests is validated before use.\n"
    "- No secrets, tokens or credentials in the code.\n"
    "- The change stays within the scope of the request."
)
# Characters of each gathered text kept in the prompt, so one large file cannot push the rest out of context.
DIFF_LIMIT = 6000
FILE_LIMIT = 4000
# Characters of each tool result written to the log.
TOOL_LOG_LIMIT = 200
# Marks the file a finding is about when it is not about any file, such as a scan error.
NO_FILE = ""


class Finding(BaseModel):
    """One finding from the reviewer, about one checklist item in one file."""

    item: Literal["input_validation", "secrets", "scope"] = Field(description="The checklist item the finding is about.")
    severity: Literal["blocking", "non_blocking"] = Field(description="blocking only if it must be fixed before merge.")
    file: str = Field(description="The changed file the finding is about, as named in the diff.")
    line: int | None = Field(default=None, description="The line in that file, if there is one.")
    message: str = Field(description="What is wrong and what would fix it, in one or two sentences.")


class ReviewSubmission(BaseModel):
    """The arguments of submit_review: the findings and a summary for the pull request."""

    findings: list[Finding] = Field(description="Every finding; an empty list when the change is clean.")
    summary: str = Field(description="One paragraph for the pull request body describing what was reviewed and found.")


@tool(args_schema=ReviewSubmission)
def submit_review(findings, summary):
    """Hand in the review. Call exactly once, with every finding and a one-paragraph summary.

    findings is a list; use an empty list when the change is clean. Each finding
    names the checklist item (input_validation, secrets or scope), a severity
    (blocking only if it must be fixed before merge), the changed file from the
    diff, the line if any, and a message. Do not state a verdict: it is derived
    from the findings.
    """
    # The call itself is the signal; the verdict node reads its arguments from the conversation.
    return "Review recorded."


def submitted_review(messages):
    """Return the validated ReviewSubmission from the latest accepted submit_review call, or None."""
    accepted = any(result.name == "submit_review" and result.status != "error" for result in latest_tool_results(messages))
    if not accepted:
        return None
    # The accepted call is the last submit_review call in the last AIMessage.
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            for call in reversed(message.tool_calls):
                if call["name"] == "submit_review":
                    return ReviewSubmission.model_validate(call["args"])
            return None
    return None


def changed_files_in(diff):
    """Return the paths named in a unified diff's headers, in order, for files that still exist afterwards."""
    paths = []
    deleted = set()
    for line in diff.splitlines():
        # "diff --git a/path b/path" opens each file's section; the b side is the path after the change.
        if line.startswith("diff --git "):
            paths.append(line.split(" b/", 1)[1].strip() if " b/" in line else line.split()[-1])
        # "+++ /dev/null" means the file was deleted, so there is no content to read.
        elif line.startswith("+++ /dev/null") and paths:
            deleted.add(paths[-1])
    return [path for path in paths if path not in deleted]


def names_changed_file(file, changed):
    """Return True when a finding's file refers to one of the changed paths, however it was spelled."""
    # The model may write a bare name, a relative path or an absolute one; forward slashes make them comparable.
    name = file.replace("\\", "/").strip().lstrip("./")
    return any(name == path or name.endswith("/" + path) or path.endswith("/" + name) for path in changed if name)


def checklist_from(conventions):
    """Return the "Review checklist" section of a CONVENTIONS.md text, or the default checklist."""
    match = re.search(r"## Review checklist\s*\n(.*?)(?:\n## |\Z)", conventions, re.S)
    return match.group(1).strip() if match else DEFAULT_CHECKLIST


def reviewer_system_prompt(checklist):
    """Return the system prompt that sets the reviewer's role and limits."""
    return (
        "You are the security and scope reviewer on a small Python Flask project.\n"
        "You review one change against this checklist from the repository's conventions:\n"
        f"{checklist}\n\n"
        "Two items are already settled by code and shown in the evidence: whether the tests pass, and "
        "whether the security scan found anything at or above the blocking severity. Formatting is handled "
        "by a formatter. Do not comment on those.\n"
        "Judge only: input from requests is validated before use (input_validation), no secrets or "
        "credentials in the code (secrets), and the change stays within the scope of the request (scope).\n"
        "Rules:\n"
        "- Base every finding on the diff and the changed files shown; name the file as the diff names it.\n"
        "- Mark a finding blocking only if it must be fixed before merge.\n"
        "- Call submit_review exactly once with every finding, or an empty list if the change is clean, "
        "and a one-paragraph summary. Do not state a verdict; it is derived from the findings.\n"
        "- Do not write code and do not use any other tool."
    )


def evidence_prompt(evidence):
    """Return the human message laying out everything the reviewer may judge from."""
    task = evidence["task"]
    tests = evidence["test_record"]
    scan = evidence["scan"]
    text = f"Request type: {task['type']}\nSpec:\n{task['spec']}\n\n"
    for path, content in task["tests"].items():
        text += f"Test file that defines done, {path}:\n{content}\n\n"
    text += f"Test run by the reviewer: {'passed' if tests.get('passed') else 'FAILED'} ({tests.get('summary', '')})\n"
    text += (
        f"Security scan: {'BLOCKING' if scan.get('blocking') else 'nothing blocking'}, "
        f"{len(scan.get('blocking_code_findings', []))} blocking and "
        f"{len(scan.get('non_blocking_code_findings', []))} non-blocking code findings\n\n"
    )
    text += f"Diff against {evidence['base_branch']}:\n{evidence['diff'][:DIFF_LIMIT]}\n\n"
    for path, content in evidence["changed_files"].items():
        text += f"Full content of {path} after the change:\n{content[:FILE_LIMIT]}\n\n"
    text += "Review the change and call submit_review."
    return text


def scan_findings(scan):
    """Turn a run_security_scan result into reviewer findings, blocking exactly where the scan said so."""
    findings = []
    for severity, key in (("blocking", "blocking_code_findings"), ("non_blocking", "non_blocking_code_findings")):
        for item in scan.get(key, []):
            findings.append({"source": "scan", "item": "security", "severity": severity, "file": item.get("file") or NO_FILE,
                             "line": item.get("line"), "message": f"{item.get('test_id')}: {item.get('message')}"})
    for severity, key in (("blocking", "blocking_dependency_findings"), ("non_blocking", "non_blocking_dependency_findings")):
        for item in scan.get(key, []):
            findings.append({"source": "scan", "item": "dependency", "severity": severity, "file": "requirements.txt",
                             "line": None, "message": f"{item.get('package')} {item.get('version')}: {item.get('id')}"})
    # A scan that could not run is blocking by the server's rules; it is carried as a finding so the reason is visible.
    for error in scan.get("errors", []):
        findings.append({"source": "scan", "item": "security", "severity": "blocking", "file": NO_FILE, "line": None, "message": error})
    return findings


def format_findings(report):
    """Turn a check-only format_code result into non-blocking findings about formatting left on the branch."""
    findings = [
        {"source": "formatter", "item": "formatting", "severity": "non_blocking", "file": path, "line": None,
         "message": "the file is not formatted; the formatter would rewrite it"}
        for path in report.get("would_reformat", [])
    ]
    findings.extend(
        {"source": "formatter", "item": "formatting", "severity": "non_blocking", "file": item["file"],
         "line": item["line"], "message": f"{item['code']}: {item['message']}"}
        for item in report.get("findings", [])
    )
    if report.get("error"):
        findings.append({"source": "formatter", "item": "formatting", "severity": "non_blocking", "file": NO_FILE,
                         "line": None, "message": report["error"]})
    return findings


class ReviewerState(TypedDict, total=False):
    """State carried through the graph."""

    # The task the change was made for: type, short_description, spec and the tests mapping.
    task: dict
    # The branch under review.
    branch: str
    # Everything gathered by code: diff, changed_files, test_record, scan, format_report, checklist, base_branch.
    evidence: dict
    # The conversation; cleared and rebuilt by gather.
    messages: Annotated[list, add_messages]
    # How many times the model was nudged to call submit_review.
    reminders: int
    # The outcome: approved, blocking_findings, non_blocking_findings, summary, no_verdict, plus the evidence.
    result: dict


def build_reviewer(repo_path, all_tools, model, settings, log=print):
    """Build the reviewer graph.

    all_tools: every MCP tool, used by code to gather the evidence.
    model: the chat model; it is bound to submit_review only.
    settings: base_branch, max_tool_steps.
    """
    repo_path = Path(repo_path)
    repo = str(repo_path)
    tools_by_name = {tool.name: tool for tool in all_tools}
    agent_tools = [submit_review]
    model_with_tools = model.bind_tools(agent_tools)

    async def git(name, **arguments):
        """Call a Git server tool; every one of them takes the repository path as repo_path."""
        return await call_tool(tools_by_name, name, {"repo_path": repo, **arguments})

    async def dev_tools(name, **arguments):
        """Call a dev tools server tool and parse its JSON result."""
        return json.loads(await call_tool(tools_by_name, name, arguments))

    async def read_file(relative_path):
        """Read a file from the target through the Filesystem server, or None if it cannot be read."""
        try:
            return await call_tool(tools_by_name, "read_text_file", {"path": str(repo_path / relative_path)})
        except Exception:
            return None

    async def gather(state):
        """Code node: check out the branch and collect every piece of evidence the reviewer may use."""
        status = await git("git_status")
        if "working tree clean" not in status:
            raise RuntimeError(f"The target repository has uncommitted changes:\n{status}")
        await git("git_checkout", branch_name=state["branch"])
        # With a clean tree, a diff against the base branch is exactly the branch's change.
        diff = await git("git_diff", target=settings["base_branch"])
        changed = changed_files_in(diff)
        changed_files = {}
        for path in changed:
            content = await read_file(path)
            if content is not None:
                changed_files[path] = content
        # The test run and the scan are the reviewer's own, not the implementer's claims.
        test_record = await dev_tools("run_tests")
        scan = await dev_tools("run_security_scan")
        format_report = await dev_tools("format_code", fix=False)
        conventions = await read_file("CONVENTIONS.md")
        checklist = checklist_from(conventions or "")
        evidence = {
            "task": state["task"], "branch": state["branch"], "base_branch": settings["base_branch"],
            "diff": diff, "changed_files": changed_files, "test_record": test_record, "scan": scan,
            "format_report": format_report, "checklist": checklist,
        }
        log(f"[gather] {len(changed)} changed file(s), tests {'passed' if test_record.get('passed') else 'FAILED'}, "
            f"scan {'blocking' if scan.get('blocking') else 'clear'}, "
            f"{len(format_report.get('would_reformat', []))} file(s) unformatted")
        return {
            "evidence": evidence,
            "reminders": 0,
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), SystemMessage(reviewer_system_prompt(checklist)),
                         HumanMessage(evidence_prompt(evidence))],
        }

    async def reviewer(state):
        """Model node: one turn, which should end in a submit_review call."""
        for result in latest_tool_results(state["messages"]):
            log(f"[tool] {result.name} ({result.status}): {str(result.content)[:TOOL_LOG_LIMIT]}")
        response = await model_with_tools.ainvoke(state["messages"])
        for tool_call in response.tool_calls:
            log(f"[reviewer] {tool_call['name']}")
        return {"messages": [response]}

    def route_after_reviewer(state):
        """Run the tool calls, nudge once if the model only talked, or give up after the step limit."""
        last = state["messages"][-1]
        steps = sum(isinstance(message, AIMessage) for message in state["messages"])
        if last.tool_calls and steps < settings["max_tool_steps"]:
            return "tools"
        if not last.tool_calls and state["reminders"] < 1:
            return "remind"
        return "verdict"

    def remind(state):
        """Code node: one reminder that the review only counts once submit_review is called."""
        log("[remind] the reviewer answered without calling submit_review")
        return {"reminders": state["reminders"] + 1,
                "messages": [HumanMessage("Your review only counts once you call submit_review. Call it now, "
                                          "with an empty findings list if the change is clean.")]}

    def route_after_tools(state):
        """Go to verdict once submit_review was accepted; otherwise let the model correct its call."""
        return "verdict" if submitted_review(state["messages"]) else "reviewer"

    def verdict(state):
        """Code node: validate the model's findings, add the code-decided ones, and derive the outcome."""
        evidence = state["evidence"]
        changed = list(evidence["changed_files"])
        submission = submitted_review(state["messages"])
        model_findings = []
        if submission is not None:
            for finding in submission.findings:
                entry = {"source": "model", **finding.model_dump()}
                # The downgrade rule: a blocking finding must name a file the diff touches, or it cannot block.
                if entry["severity"] == "blocking" and not names_changed_file(entry["file"], changed):
                    entry["severity"] = "non_blocking"
                    entry["message"] += " (downgraded: the file is not part of this change)"
                model_findings.append(entry)
        findings = model_findings + scan_findings(evidence["scan"]) + format_findings(evidence["format_report"])
        # The reviewer's own test run failing is blocking whatever the implementer reported.
        if not evidence["test_record"].get("passed"):
            findings.append({"source": "tests", "item": "tests", "severity": "blocking", "file": NO_FILE, "line": None,
                             "message": f"the tests fail on the branch: {evidence['test_record'].get('summary', '')}"})
        # No verdict means the change cannot be judged, which fails closed.
        no_verdict = submission is None
        if no_verdict:
            findings.append({"source": "reviewer", "item": "review", "severity": "blocking", "file": NO_FILE, "line": None,
                             "message": "the reviewer did not submit a review"})
        blocking = [finding for finding in findings if finding["severity"] == "blocking"]
        non_blocking = [finding for finding in findings if finding["severity"] != "blocking"]
        result = {
            "approved": not blocking,
            "blocking_findings": blocking,
            "non_blocking_findings": non_blocking,
            "summary": submission.summary if submission else "",
            "no_verdict": no_verdict,
            "branch": state["branch"],
            "test_record": evidence["test_record"],
            "scan": evidence["scan"],
            "diff": evidence["diff"],
        }
        log(f"[verdict] {'approved' if result['approved'] else 'changes requested'}: "
            f"{len(blocking)} blocking, {len(non_blocking)} non-blocking")
        return {"result": result}

    graph = StateGraph(ReviewerState)
    graph.add_node("gather", gather)
    graph.add_node("reviewer", reviewer)
    graph.add_node("tools", ToolNode(agent_tools, handle_tool_errors=True))
    graph.add_node("remind", remind)
    graph.add_node("verdict", verdict)
    graph.add_edge(START, "gather")
    graph.add_edge("gather", "reviewer")
    graph.add_conditional_edges("reviewer", route_after_reviewer, ["tools", "remind", "verdict"])
    graph.add_edge("remind", "reviewer")
    graph.add_conditional_edges("tools", route_after_tools, ["verdict", "reviewer"])
    graph.add_edge("verdict", END)
    return graph.compile()


def recursion_limit(settings):
    """Enough graph steps for the model to use its full turn budget plus gather, remind and verdict."""
    return 2 * settings["max_tool_steps"] + 10


def describe_findings(findings):
    """Return one line per finding, as given to the implementer or written into a pull request."""
    lines = []
    for finding in findings:
        where = finding["file"] + (f":{finding['line']}" if finding.get("line") else "") if finding.get("file") else "general"
        lines.append(f"- [{finding['item']}] {where}: {finding['message']}")
    return "\n".join(lines)
