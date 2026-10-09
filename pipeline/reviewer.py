"""Security and scope reviewer.

Judges one branch's change against the target repository's review checklist
and returns a structured verdict. The graph mixes code steps with model steps:

    gather    (code)   check out the branch, collect the diff against the merge base, the changed
                       files, a fresh test run, the security scan, a formatting report and the
                       checklist as it is on the base branch
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

from pipeline.tooling import call_tool, latest_tool_results, record_tool_results, scrub_secrets

# The checklist items the model judges. The other two items, tests and scan severity, are decided by code.
JUDGED_ITEMS = ("input_validation", "secrets", "information_exposure", "scope")
# Used when the target has no CONVENTIONS.md, so the reviewer always has a checklist to apply.
DEFAULT_CHECKLIST = (
    "- Input from requests is validated before use.\n"
    "- No secrets, tokens or credentials in the code.\n"
    "- No route exposes internal configuration, environment or debug information.\n"
    "- The change stays within the scope of the request."
)
# The questions the model answers for every new or changed route, in both the review and the security pass.
ROUTE_QUESTIONS = (
    "For each new or changed route in the diff, answer these three questions and report any problem as a finding:\n"
    "1. What can a caller now reach that they could not before? (scope, or information_exposure if it is internal)\n"
    "2. Does the response include internal data such as configuration, environment, stack traces or file paths? "
    "(information_exposure)\n"
    "3. Is any request input used before it is validated? (input_validation)"
)
# The question about the tests, asked only in the full review pass, which sees the spec and the test files.
TEST_QUESTIONS = (
    "For each test file shown, check every assertion against the spec: do the expected status codes and response "
    "bodies match what the spec requires? A test that expects something the spec does not say, such as 500 where "
    "the spec requires 404, is a blocking tests_match_spec finding on that test file, named as the diff names it."
)
# Names of the model passes: the full review, and the optional security-only second call.
REVIEW_PASS, SECURITY_PASS = "review", "security"
# Characters of each gathered text kept in the prompt, so one large file cannot push the rest out of context.
DIFF_LIMIT = 6000
FILE_LIMIT = 4000
# Characters of each tool result written to the log.
TOOL_LOG_LIMIT = 200
# Marks the file a finding is about when it is not about any file, such as a scan error.
NO_FILE = ""


class Finding(BaseModel):
    """One finding from the reviewer, about one checklist item in one file."""

    item: Literal["input_validation", "secrets", "information_exposure", "scope", "tests_match_spec"] = Field(
        description="The checklist item the finding is about."
    )
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
    names the checklist item (input_validation, secrets, information_exposure or
    scope), a severity (blocking only if it must be fixed before merge), the
    changed file from the diff, the line if any, and a message. Do not state a
    verdict: it is derived from the findings.
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


def matching_changed_file(file, changed):
    """Return the changed path a finding's file refers to, however it was spelled, or None."""
    # The model may write a bare name, a relative path or an absolute one; forward slashes make them comparable.
    name = file.replace("\\", "/").strip().lstrip("./")
    if not name:
        return None
    for path in changed:
        if name == path or name.endswith("/" + path) or path.endswith("/" + name):
            return path
    return None


def changed_lines_in(diff):
    """Return {path: set of line numbers} for the lines a unified diff adds or changes, numbered in the new file."""
    lines = {}
    path = None
    new_line = 0
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[1].strip() if " b/" in line else line.split()[-1]
            lines.setdefault(path, set())
        # A hunk header "@@ -a,b +c,d @@" says the new file's numbering restarts at c.
        elif line.startswith("@@ ") and path is not None:
            match = re.search(r"\+(\d+)", line)
            new_line = int(match.group(1)) if match else 0
        elif path is None or line.startswith(("---", "+++")):
            continue
        # Added lines are the change; context lines only advance the count; removed lines have no new number.
        elif line.startswith("+"):
            lines[path].add(new_line)
            new_line += 1
        elif line.startswith(" ") or line == "":
            new_line += 1
    return lines


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
        "Judge the remaining items: input validated before use (input_validation), no secrets or credentials "
        "in the code (secrets), no route exposing internal configuration, environment or debug information "
        "(information_exposure), the change staying within the scope of the request (scope), and every test "
        "asserting what the spec requires (tests_match_spec).\n"
        f"{ROUTE_QUESTIONS}\n"
        f"{TEST_QUESTIONS}\n"
        "Rules:\n"
        "- Base every finding on the diff and the changed files shown; name the file as the diff names it.\n"
        "- Mark a finding blocking only if it must be fixed before merge.\n"
        "- Call submit_review exactly once with every finding, or an empty list if the change is clean, "
        "and a one-paragraph summary. Do not state a verdict; it is derived from the findings.\n"
        "- Do not write code and do not use any other tool."
    )


def security_system_prompt():
    """Return the system prompt for the optional security-only pass, which asks the route questions and nothing else."""
    return (
        "You are a security reviewer on a small Python Flask project. You see one change: its diff and the "
        "full content of the changed files.\n"
        f"{ROUTE_QUESTIONS}\n"
        "Rules:\n"
        "- Report only security problems: information_exposure, input_validation or secrets. Scope, tests and "
        "formatting are judged elsewhere.\n"
        "- Base every finding on the diff and the changed files shown; name the file as the diff names it.\n"
        "- Mark a finding blocking only if it must be fixed before merge.\n"
        "- Call submit_review exactly once with every finding, or an empty list if there is none, and a "
        "one-sentence summary. Do not write code and do not use any other tool."
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
    # The conversation; cleared and rebuilt by gather and by next_pass.
    messages: Annotated[list, add_messages]
    # How many times the model was nudged to call submit_review in the current pass.
    reminders: int
    # The pass the model is in: "review", or "security" for the optional second call.
    pass_name: str
    # Passes still to run after the current one.
    remaining_passes: list
    # pass name -> the validated submission as a dict, or None when that pass produced no verdict.
    submissions: dict
    # The model's part of the verdict: validated findings, summary, no_verdict; what a remote reviewer returns.
    model_verdict: dict
    # The outcome: approved, blocking_findings, non_blocking_findings, summary, no_verdict, plus the evidence.
    result: dict


async def gather_evidence(repo_path, tools_by_name, task, branch, settings, log=print):
    """Check out the branch and collect every piece of evidence the reviewer may judge from.

    This is the only part of a review that needs the MCP servers and the target
    checkout. The model pass and the verdict are a pure function of what it
    returns, which is what lets them run behind an A2A boundary.
    """
    repo_path = Path(repo_path)
    repo = str(repo_path)

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

    status = await git("git_status")
    if "working tree clean" not in status:
        raise RuntimeError(f"The target repository has uncommitted changes:\n{status}")
    await git("git_checkout", branch_name=branch)
    # The diff is taken against the merge base, so commits added to the base branch after this branch
    # was created do not show up as if the branch had removed them.
    diff_result = await dev_tools("diff_against_base")
    if diff_result.get("error"):
        raise RuntimeError(diff_result["error"])
    diff = diff_result["diff"]
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
    # The checklist is read from the base branch, not the branch under review, so a change cannot
    # remove the rule it is judged by, and a branch older than a new rule is still judged by it.
    conventions = await dev_tools("read_file_on_base", path="CONVENTIONS.md")
    checklist = checklist_from(conventions.get("content") or "")
    evidence = {
        "task": task, "branch": branch, "base_branch": settings["base_branch"],
        "diff": diff, "changed_files": changed_files, "test_record": test_record, "scan": scan,
        "format_report": format_report, "checklist": checklist,
    }
    log(f"[gather] {len(changed)} changed file(s), tests {'passed' if test_record.get('passed') else 'FAILED'}, "
        f"scan {'blocking' if scan.get('blocking') else 'clear'}, "
        f"{len(format_report.get('would_reformat', []))} file(s) unformatted")
    return evidence


def first_pass_state(evidence, settings):
    """Return the state fields that start the review pass over the given evidence."""
    return {
        "evidence": evidence,
        "reminders": 0,
        "pass_name": REVIEW_PASS,
        # The security-only pass is a second model call with the same evidence and only the route questions.
        "remaining_passes": [SECURITY_PASS] if settings.get("security_pass") else [],
        "submissions": {},
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), SystemMessage(reviewer_system_prompt(evidence["checklist"])),
                     HumanMessage(evidence_prompt(evidence))],
    }


def model_verdict(submissions, evidence):
    """Reduce the model passes' submissions to validated findings, a summary and whether any pass gave no verdict.

    This is the part of the verdict that depends only on what the model said
    and the evidence it saw, so it is what a remote reviewer sends back. The
    downgrade and line rules are applied here.
    """
    changed = list(evidence["changed_files"])
    changed_lines = changed_lines_in(evidence["diff"])
    findings = []
    for pass_name, submission in submissions.items():
        for finding in (submission or {}).get("findings", []):
            entry = {"source": "model" if pass_name == REVIEW_PASS else pass_name, **finding}
            path = matching_changed_file(entry["file"], changed)
            # The downgrade rule: a blocking finding must name a file the diff touches, or it cannot block.
            if entry["severity"] == "blocking" and path is None:
                entry["severity"] = "non_blocking"
                entry["message"] += " (downgraded: the file is not part of this change)"
            # A line number the diff did not add would mislead the reader of the pull request, so it is dropped.
            if entry.get("line") is not None and entry["line"] not in changed_lines.get(path, set()):
                entry["line"] = None
            findings.append(entry)
    # No verdict from any pass means the change cannot be judged, which fails closed.
    missing = [pass_name for pass_name, submission in submissions.items() if submission is None]
    no_verdict = bool(missing) or REVIEW_PASS not in submissions
    # The pull request summary comes from the review pass; the security pass adds its sentence only when it found something.
    review = submissions.get(REVIEW_PASS) or {}
    security = submissions.get(SECURITY_PASS) or {}
    summary = review.get("summary", "")
    if security.get("findings"):
        summary = f"{summary} Security pass: {security.get('summary', '')}".strip()
    return {"findings": findings, "summary": summary, "no_verdict": no_verdict,
            "missing_passes": missing or ([REVIEW_PASS] if REVIEW_PASS not in submissions else [])}


def assemble_result(verdict, evidence, branch, log=print):
    """Combine the model's verdict with the code-decided findings from the evidence into the review result."""
    findings = list(verdict["findings"]) + scan_findings(evidence["scan"]) + format_findings(evidence["format_report"])
    # The reviewer's own test run failing is blocking whatever the implementer reported.
    if not evidence["test_record"].get("passed"):
        findings.append({"source": "tests", "item": "tests", "severity": "blocking", "file": NO_FILE, "line": None,
                         "message": f"the tests fail on the branch: {evidence['test_record'].get('summary', '')}"})
    if verdict["no_verdict"]:
        findings.append({"source": "reviewer", "item": "review", "severity": "blocking", "file": NO_FILE, "line": None,
                         "message": f"the reviewer did not submit a review ({', '.join(verdict.get('missing_passes') or [REVIEW_PASS])} pass)"})
    blocking = [finding for finding in findings if finding["severity"] == "blocking"]
    non_blocking = [finding for finding in findings if finding["severity"] != "blocking"]
    result = {
        "approved": not blocking,
        "blocking_findings": blocking,
        "non_blocking_findings": non_blocking,
        "summary": verdict["summary"],
        "no_verdict": verdict["no_verdict"],
        "branch": branch,
        "test_record": evidence["test_record"],
        "scan": evidence["scan"],
        "diff": evidence["diff"],
    }
    log(f"[verdict] {'approved' if result['approved'] else 'changes requested'}: "
        f"{len(blocking)} blocking, {len(non_blocking)} non-blocking")
    return result


def build_reviewer(repo_path, all_tools, model, settings, log=print, provided_evidence=False):
    """Build the reviewer graph.

    all_tools: every MCP tool, used by code to gather the evidence.
    model: the chat model; it is bound to submit_review only.
    settings: base_branch, max_tool_steps, security_pass.
    provided_evidence: True to start from evidence given in the input state instead of gathering it,
        which is how the A2A reviewer service runs the same graph without any repository access.
    """
    tools_by_name = {tool.name: tool for tool in all_tools}
    agent_tools = [submit_review]
    model_with_tools = model.bind_tools(agent_tools)

    async def gather(state):
        """Code node: check out the branch, collect the evidence, and start the review pass."""
        evidence = await gather_evidence(repo_path, tools_by_name, state["task"], state["branch"], settings, log)
        return first_pass_state(evidence, settings)

    def start(state):
        """Code node: start the review pass from the evidence the caller provided."""
        return first_pass_state(state["evidence"], settings)

    async def reviewer(state):
        """Model node: one turn, which should end in a submit_review call."""
        for result in latest_tool_results(state["messages"]):
            log(f"[tool] {result.name} ({result.status}): {scrub_secrets(result.content)[:TOOL_LOG_LIMIT]}")
        # The full calls and results go to the run's tool log, so the evidence never depends on what was printed.
        record_tool_results(state["messages"])
        response = await model_with_tools.ainvoke(state["messages"])
        for tool_call in response.tool_calls:
            log(f"[{state['pass_name']}] {tool_call['name']}")
        return {"messages": [response]}

    def route_after_reviewer(state):
        """Run the tool calls, nudge once if the model only talked, or give up after the step limit."""
        last = state["messages"][-1]
        steps = sum(isinstance(message, AIMessage) for message in state["messages"])
        if last.tool_calls and steps < settings["max_tool_steps"]:
            return "tools"
        if not last.tool_calls and state["reminders"] < 1:
            return "remind"
        return "record"

    def remind(state):
        """Code node: one reminder that the review only counts once submit_review is called."""
        log(f"[remind] the {state['pass_name']} pass answered without calling submit_review")
        return {"reminders": state["reminders"] + 1,
                "messages": [HumanMessage("Your review only counts once you call submit_review. Call it now, "
                                          "with an empty findings list if the change is clean.")]}

    def route_after_tools(state):
        """Record the pass once submit_review was accepted; otherwise let the model correct its call."""
        return "record" if submitted_review(state["messages"]) else "reviewer"

    def record(state):
        """Code node: keep the current pass's submission, or None when it produced no verdict."""
        submission = submitted_review(state["messages"])
        return {"submissions": {**state["submissions"], state["pass_name"]: submission.model_dump() if submission else None}}

    def route_after_record(state):
        """Start the next pass if one remains, otherwise derive the verdict."""
        return "next_pass" if state["remaining_passes"] else "verdict"

    def next_pass(state):
        """Code node: start the security-only pass with a fresh conversation over the same evidence."""
        pass_name, *remaining = state["remaining_passes"]
        log(f"[{pass_name}] starting the security-only pass")
        return {
            "pass_name": pass_name,
            "remaining_passes": remaining,
            "reminders": 0,
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), SystemMessage(security_system_prompt()),
                         HumanMessage(evidence_prompt(state["evidence"]))],
        }

    def verdict(state):
        """Code node: reduce the model's submissions to a verdict, then combine it with the code-decided findings."""
        evidence = state["evidence"]
        model_part = model_verdict(state["submissions"], evidence)
        result = assemble_result(model_part, evidence, evidence.get("branch", state.get("branch", "")), log)
        return {"model_verdict": model_part, "result": result}

    graph = StateGraph(ReviewerState)
    # Locally the evidence is gathered from the servers; the A2A service receives it ready-made.
    graph.add_node("gather", start if provided_evidence else gather)
    graph.add_node("reviewer", reviewer)
    graph.add_node("tools", ToolNode(agent_tools, handle_tool_errors=True))
    graph.add_node("remind", remind)
    graph.add_node("record", record)
    graph.add_node("next_pass", next_pass)
    graph.add_node("verdict", verdict)
    graph.add_edge(START, "gather")
    graph.add_edge("gather", "reviewer")
    graph.add_conditional_edges("reviewer", route_after_reviewer, ["tools", "remind", "record"])
    graph.add_edge("remind", "reviewer")
    graph.add_conditional_edges("tools", route_after_tools, ["record", "reviewer"])
    # Each pass ends in record; the security pass, when enabled, starts from there with a fresh conversation.
    graph.add_conditional_edges("record", route_after_record, ["next_pass", "verdict"])
    graph.add_edge("next_pass", "reviewer")
    graph.add_edge("verdict", END)
    return graph.compile()


def recursion_limit(settings):
    """Enough graph steps for every pass to use its full turn budget plus gather, remind, record and verdict."""
    passes = 2 if settings.get("security_pass") else 1
    return passes * (2 * settings["max_tool_steps"] + 6) + 10


def describe_findings(findings):
    """Return one line per finding, as given to the implementer or written into a pull request."""
    lines = []
    for finding in findings:
        where = finding["file"] + (f":{finding['line']}" if finding.get("line") else "") if finding.get("file") else "general"
        lines.append(f"- [{finding['item']}] {where}: {finding['message']}")
    return "\n".join(lines)
