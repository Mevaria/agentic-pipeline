"""Custom MCP server for the dev pipeline.

Runs the target repository's tests, scans it for security issues, pushes the
working branch, and records notifications for the developer.

Every safety rule lives in this code rather than in prompts: the repository
path is fixed by configuration, protected branches can never be pushed, and
the blocking decision for scan findings is computed here, not by a model.

Configuration (environment variables):
    TARGET_REPO_PATH   Absolute path to the repository the pipeline works on.
    BLOCKING_SEVERITY  LOW, MEDIUM or HIGH. Code findings at or above it block.
    BASE_BRANCH        Branch that changes are compared against. Defaults to main.
    PROTECTED_BRANCHES Comma-separated branches that may never be pushed.
    NOTIFICATIONS_LOG  File that outcome notifications are appended to.
    FAIL_FIRST_WORKTREE Folder for the throwaway worktree the fail-first check runs in.
    TASKS_PATH         Folder that generated task folders are written under. Defaults to runs/tasks.
    RUN_ID             Name of this run's task folder under TASKS_PATH. Set by the pipeline per run.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP

# Ranks Bandit's severity labels so a finding can be compared against the threshold.
SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
# Characters of command output kept in a tool result, so one long run cannot flood the model's context.
OUTPUT_LIMIT = 3000
# Seconds a test run or scan may take before it is abandoned and reported as failed.
COMMAND_TIMEOUT = 300
# This folder, which holds the pytest_outcomes plugin, and the pipeline's own root one level up.
PLUGIN_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = PLUGIN_DIR.parent
# Statuses the fail-first check gives a test: failed on an assertion, passed already, or did not run properly.
RED, PASSES, BROKEN = "red", "passes", "broken"
# Folder inside the target repository that holds its tests; generated test files are added to it later.
TESTS_DIR = "tests"
# Task types a generated task may have. They become the branch prefix and the commit type, so they
# match the Conventional Commits types the implementer accepts.
TASK_TYPES = {"feat", "fix", "refactor", "test", "docs", "chore", "perf", "style", "build", "ci"}
# A short description is a lowercase slug: words of letters and digits joined by single hyphens.
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
SLUG_LIMIT = 50
# A generated test file is a bare pytest module name: no folders, no other extension.
TEST_FILE_PATTERN = re.compile(r"^test_[a-z0-9_]+\.py$")
# Used when the pipeline did not set RUN_ID, so a server started by hand still writes somewhere sensible.
DEFAULT_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

# The server itself. Functions decorated with @mcp.tool() are exposed to clients as MCP tools,
# and FastMCP uses each function's docstring as the tool description the model reads.
mcp = FastMCP("dev-tools")


def get_repo_path():
    """Return the target repository path from TARGET_REPO_PATH, checking it is a git repository."""
    repo_path = os.environ.get("TARGET_REPO_PATH")
    # Fail loudly rather than fall back to the current folder, which could be the wrong repository.
    if not repo_path:
        raise RuntimeError("TARGET_REPO_PATH is not set")
    # Resolve symlinks and relative segments so later containment checks compare like with like.
    path = Path(repo_path).resolve()
    if not (path / ".git").exists():
        raise RuntimeError(f"{path} is not a git repository")
    return path


def get_blocking_severity():
    """Return the configured severity threshold, defaulting to MEDIUM, as an upper-case label."""
    # Upper-casing lets .env use "medium" or "MEDIUM" interchangeably.
    severity = os.environ.get("BLOCKING_SEVERITY", "MEDIUM").upper()
    # An unknown label would silently block nothing, so refuse it instead.
    if severity not in SEVERITY_ORDER:
        raise RuntimeError(f"BLOCKING_SEVERITY must be one of {list(SEVERITY_ORDER)}")
    return severity


def get_protected_branches():
    """Return the set of branch names that git_push must refuse, defaulting to main and master."""
    raw = os.environ.get("PROTECTED_BRANCHES", "main,master")
    # Split on commas and drop whitespace and empty entries, so "main, master," is accepted.
    return {branch.strip() for branch in raw.split(",") if branch.strip()}


def get_base_branch():
    """Return the branch that changes are compared against, defaulting to main."""
    return os.environ.get("BASE_BRANCH", "main")


def get_fail_first_worktree():
    """Return the folder for the fail-first check's worktree, defaulting to runs/fail_first_worktree in the pipeline."""
    # A fixed path, not a random temp folder, so a run that crashed leaves it where the next run can find and remove it.
    default = PIPELINE_ROOT / "runs" / "fail_first_worktree"
    return Path(os.environ.get("FAIL_FIRST_WORKTREE", default)).resolve()


def get_task_folder():
    """Return this run's task folder, TASKS_PATH/RUN_ID, defaulting to runs/tasks in the pipeline."""
    # Generated tasks are run output, so they live under the gitignored runs folder, not next to hand-written tasks.
    tasks_path = Path(os.environ.get("TASKS_PATH", PIPELINE_ROOT / "runs" / "tasks")).resolve()
    # One folder per run keeps a revised test from overwriting an earlier run's output.
    return tasks_path / os.environ.get("RUN_ID", DEFAULT_RUN_ID)


def run_command(args, cwd, env=None):
    """Run a command without a shell and return (exit_code, stdout, stderr).

    env replaces the child's environment when given; otherwise the server's own is inherited.
    """
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            # This server talks to its client over stdin, so child processes
            # get an empty input instead of inheriting the protocol channel.
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
        )
    # A hung command is reported as a failure instead of hanging the whole server.
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {COMMAND_TIMEOUT} seconds"
    # A missing executable (for example git not installed) is reported the same way.
    except FileNotFoundError as error:
        return -1, "", f"Command not found: {error}"
    # stdout and stderr can be None when nothing was captured, so they are normalised to strings.
    return result.returncode, result.stdout or "", result.stderr or ""


def tail(text, limit=OUTPUT_LIMIT):
    """Keep the end of long output, where test failures and errors appear."""
    if len(text) <= limit:
        return text
    # The marker tells the reader that earlier output was dropped.
    return "[output truncated]\n" + text[-limit:]


def resolve_inside_repo(repo_path, relative_path):
    """Resolve a path and refuse anything outside the repository."""
    # Resolving collapses ".." segments, so an escape attempt becomes visible as a path outside the repo.
    candidate = (repo_path / relative_path).resolve()
    # The repository root itself is allowed; anything whose ancestors do not include the root is not.
    if candidate != repo_path and repo_path not in candidate.parents:
        raise ValueError(f"Path is outside the repository: {relative_path}")
    return candidate


@mcp.tool()
def run_tests(test_path: str = "") -> dict:
    """Run the repository's pytest suite and report the result.

    Use after every code change. Pass a path relative to the repository root,
    such as "tests/test_app.py", to run one file; leave empty to run them all.
    """
    repo_path = get_repo_path()
    # sys.executable is the interpreter this server runs under, which is the shared .venv,
    # so the target repository's tests run with the same installed packages.
    # -q keeps the output short, --tb=short keeps tracebacks readable, and the cache
    # plugin is disabled so the run leaves no .pytest_cache folder behind in the repository.
    args = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider"]
    if test_path:
        # The path is checked to be inside the repository first, so "../" cannot escape it.
        args.append(str(resolve_inside_repo(repo_path, test_path)))
    exit_code, stdout, stderr = run_command(args, repo_path)
    output = stdout + stderr
    # pytest prints its one-line summary ("3 passed in 0.2s") last, so the last non-empty line is kept as the summary.
    summary_lines = [line for line in output.strip().splitlines() if line.strip()]
    return {
        # pytest exits with 0 only when every collected test passed.
        "passed": exit_code == 0,
        "exit_code": exit_code,
        "summary": summary_lines[-1] if summary_lines else "",
        "output": tail(output),
    }


def scan_code(repo_path):
    """Run Bandit on the repository and return (findings, error).

    Each finding is a flat dict with the file, line, severity, confidence,
    Bandit test id and message. error is None unless Bandit's output could
    not be read, in which case findings is empty and error explains why.
    """
    args = [
        # -r scans recursively, -f json gives machine-readable output, -q silences the progress text.
        sys.executable, "-m", "bandit", "-r", ".", "-f", "json", "-q",
        # Tests are excluded because test code legitimately uses patterns Bandit flags, such as assert,
        # and virtual environments are excluded so third-party code is not scanned as if it were ours.
        "-x", "./tests,./.venv,./venv",
    ]
    # Bandit's exit code is 1 whenever it finds anything, so it is ignored and the JSON is used instead.
    _, stdout, stderr = run_command(args, repo_path)
    try:
        report = json.loads(stdout)
    # Unparseable output means the scan did not run properly; the caller treats this as blocking.
    except ValueError:
        return [], f"Bandit output could not be parsed: {tail(stdout + stderr, 500)}"
    # Only the fields the reviewer needs are kept, under short stable names.
    findings = [
        {
            "file": issue.get("filename"),
            "line": issue.get("line_number"),
            "severity": issue.get("issue_severity"),
            "confidence": issue.get("issue_confidence"),
            "test_id": issue.get("test_id"),
            "message": issue.get("issue_text"),
        }
        for issue in report.get("results", [])
    ]
    return findings, None


def scan_dependencies(repo_path):
    """Run pip-audit on requirements.txt and return (findings, error).

    Each finding names the package, its version, the vulnerability id and the
    versions that fix it. A repository without requirements.txt has nothing
    to audit and returns no findings. error is None unless pip-audit's output
    could not be read.
    """
    requirements = repo_path / "requirements.txt"
    # No requirements file means no pinned dependencies to audit, which is not an error.
    if not requirements.exists():
        return [], None
    # The module name is pip_audit with an underscore, unlike the pip-audit command.
    args = [sys.executable, "-m", "pip_audit", "-r", str(requirements), "-f", "json"]
    # pip-audit exits non-zero when it finds vulnerabilities, so the exit code is ignored in favor of the JSON.
    _, stdout, stderr = run_command(args, repo_path)
    try:
        report = json.loads(stdout)
    # Unparseable output means the audit did not run properly; the caller treats this as blocking.
    except ValueError:
        return [], f"pip-audit output could not be parsed: {tail(stdout + stderr, 500)}"
    findings = []
    # The report lists every dependency; only those with at least one vulnerability become findings.
    for dependency in report.get("dependencies", []):
        for vulnerability in dependency.get("vulns", []):
            findings.append({
                "package": dependency.get("name"),
                "version": dependency.get("version"),
                "id": vulnerability.get("id"),
                "fix_versions": vulnerability.get("fix_versions", []),
            })
    return findings, None


def requirements_changed(repo_path):
    """Return (changed, error) for requirements.txt relative to the base branch.

    Covers committed changes on the branch, uncommitted edits, and a new
    untracked file. If git cannot answer, the change is treated as touching
    requirements.txt so that a failure never hides a vulnerability.
    """
    commands = [
        # Files changed by commits on this branch since it diverged from the base branch.
        # The three-dot form uses the merge base, so later commits on the base branch are not counted.
        ["git", "diff", "--name-only", f"{get_base_branch()}...HEAD"],
        # Files with uncommitted edits, staged or not, compared with the last commit.
        ["git", "diff", "--name-only", "HEAD"],
        # New files git does not know about yet, ignoring anything in .gitignore.
        ["git", "ls-files", "--others", "--exclude-standard"],
    ]
    changed_files = set()
    for args in commands:
        exit_code, stdout, stderr = run_command(args, repo_path)
        # A failed git command, such as an unknown base branch, fails closed: report "changed" and say why.
        if exit_code != 0:
            return True, f"Could not compare against base branch: {tail(stderr, 300)}"
        # Each command prints one path per line; blank lines are skipped.
        changed_files.update(line.strip() for line in stdout.splitlines() if line.strip())
    return "requirements.txt" in changed_files, None


@mcp.tool()
def run_security_scan() -> dict:
    """Scan the repository for security issues in code and dependencies.

    Returns code findings from Bandit, known vulnerabilities in dependencies
    from pip-audit, and a "blocking" flag decided by fixed rules. Code findings
    block at or above the configured severity. Dependency vulnerabilities block
    only when the change edits requirements.txt; otherwise they are reported as
    non-blocking so unrelated work is not held up. Findings that do not block
    should still be mentioned in the pull request.
    """
    repo_path = get_repo_path()
    threshold = get_blocking_severity()
    # The three sources of evidence are gathered first; each returns an error instead of raising.
    code_findings, code_error = scan_code(repo_path)
    dependency_findings, dependency_error = scan_dependencies(repo_path)
    dependencies_touched, diff_error = requirements_changed(repo_path)

    # Rule 1: a code finding blocks when its severity ranks at or above the threshold.
    # An unknown severity label ranks 0 and therefore never blocks.
    blocking_code = [
        finding for finding in code_findings
        if SEVERITY_ORDER.get(finding["severity"], 0) >= SEVERITY_ORDER[threshold]
    ]
    non_blocking_code = [f for f in code_findings if f not in blocking_code]
    # Rule 2: dependency findings block only when this change touched requirements.txt,
    # so a vulnerability that was already there does not hold up unrelated work.
    blocking_dependencies = dependency_findings if dependencies_touched else []
    non_blocking_dependencies = [] if dependencies_touched else dependency_findings
    # Rule 3: any scan that failed to run is itself blocking, so a broken scan cannot pass silently.
    errors = [error for error in (code_error, dependency_error, diff_error) if error]

    return {
        "blocking": bool(blocking_code or blocking_dependencies or errors),
        "blocking_severity": threshold,
        "requirements_changed": dependencies_touched,
        "blocking_code_findings": blocking_code,
        "non_blocking_code_findings": non_blocking_code,
        "blocking_dependency_findings": blocking_dependencies,
        "non_blocking_dependency_findings": non_blocking_dependencies,
        "errors": errors,
    }


def remove_worktree(repo_path, worktree):
    """Remove a worktree folder and its registration in the repository, whether or not git still recognises it."""
    if worktree.exists():
        # --force removes it even with untracked or modified files in it, such as the tests written for a check.
        run_command(["git", "worktree", "remove", "--force", str(worktree)], repo_path)
    # A folder git no longer recognises, for example one left by a run against another repository, is deleted directly.
    if worktree.exists():
        shutil.rmtree(worktree, ignore_errors=True)
    # Drop registrations whose folders are gone, so the same path can be used again.
    run_command(["git", "worktree", "prune"], repo_path)


def classify_outcomes(outcomes, test_files):
    """Turn the plugin's raw records into one entry per test with a status of red, passes or broken.

    A test is red only when its body failed on an assertion or a pytest.fail,
    which includes pytest.raises reporting DID NOT RAISE. A test that passed
    does not test the requested behaviour. Anything else, such as a failed
    import, a failed fixture, a skip or another exception, is broken. A given
    file that produced no test at all is reported as broken too.
    """
    results = []
    for nodeid, phases in outcomes.items():
        # A file that failed to import never produced tests, so the file itself is the entry.
        if "collect" in phases:
            results.append({"test": nodeid, "status": BROKEN, "detail": f"could not be collected: {phases['collect']['message']}"})
            continue
        # The plugin records setup only when it failed or skipped, and then the body never ran.
        setup = phases.get("setup")
        if setup is not None:
            detail = f"setup failed: {setup['message']}" if setup["outcome"] == "failed" else "skipped"
            results.append({"test": nodeid, "status": BROKEN, "detail": detail})
            continue
        call = phases.get("call")
        if call is None:
            continue
        if call["outcome"] == "passed":
            status, detail = PASSES, "passed before any change was made, so it does not test the requested behaviour"
        elif call["outcome"] == "failed" and (call["is_assertion"] or call["is_pytest_fail"]):
            status, detail = RED, call["message"]
        elif call["outcome"] == "failed":
            status, detail = BROKEN, f"raised {call['exception']} instead of failing an assertion: {call['message']}"
        else:
            status, detail = BROKEN, "skipped"
        results.append({"test": nodeid, "status": status, "detail": detail})
    # pytest node ids start with the file's path, so a file with no entry produced no tests.
    for path in test_files:
        if not any(entry["test"].startswith(path) for entry in results):
            results.append({"test": path, "status": BROKEN, "detail": "no tests were collected from this file"})
    return results


@mcp.tool()
def check_tests_fail(tests: dict[str, str]) -> dict:
    """Check that new tests fail on the base branch, before any change is made.

    tests maps a path relative to the repository root, such as "tests/test_x.py",
    to the file's content. The tests run in a throwaway worktree of the base
    branch, so the repository itself is never touched. Every test must fail on
    an assertion to count as red: a test that passes does not test the requested
    behaviour, and a test that errors, on an import, a fixture or a typo, is
    broken. Returns all_red and one entry per test with its status and detail.
    """
    repo_path = get_repo_path()
    base_branch = get_base_branch()
    worktree = get_fail_first_worktree()
    # Whatever a crashed run left behind is cleared first, so the add below starts from a free path.
    remove_worktree(repo_path, worktree)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    # --detach checks out the commit rather than the branch, since a branch cannot be checked out in two places.
    exit_code, _, stderr = run_command(["git", "worktree", "add", "--detach", str(worktree), base_branch], repo_path)
    if exit_code != 0:
        return {"all_red": False, "tests": [], "error": f"Could not create a worktree of '{base_branch}': {tail(stderr, 300).strip()}"}
    # The plugin writes here; it sits next to the worktree, not inside it, so removing the worktree cannot lose it.
    outcomes_file = worktree.parent / "fail_first_outcomes.json"
    try:
        paths = []
        for relative_path, content in tests.items():
            # The same containment check as run_tests, applied to the worktree, so a path cannot escape it.
            target = resolve_inside_repo(worktree, relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            paths.append(str(target))
        environment = {
            **os.environ,
            "PYTEST_OUTCOMES_FILE": str(outcomes_file),
            # The plugin's folder goes first on PYTHONPATH so "-p pytest_outcomes" can import it.
            "PYTHONPATH": os.pathsep.join(part for part in (str(PLUGIN_DIR), os.environ.get("PYTHONPATH", "")) if part),
        }
        args = [
            sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider",
            "-p", "pytest_outcomes",
            # Without this, one file that fails to import stops the whole run and hides every other verdict.
            "--continue-on-collection-errors",
            # Only the new files run; the existing suite is expected to pass and is not the question here.
            *paths,
        ]
        exit_code, stdout, stderr = run_command(args, worktree, env=environment)
        try:
            outcomes = json.loads(outcomes_file.read_text(encoding="utf-8"))
        # No file means pytest never reached the end of the session, for example a bad argument or a crash.
        except (OSError, ValueError):
            return {"all_red": False, "tests": [], "error": f"pytest did not report outcomes: {tail(stdout + stderr, 500)}"}
        results = classify_outcomes(outcomes, [path.replace("\\", "/") for path in tests])
    finally:
        # Always runs, including on an exception above, so a failed check never leaves the worktree behind.
        remove_worktree(repo_path, worktree)
        outcomes_file.unlink(missing_ok=True)
    return {
        # An empty result list is not a pass: at least one test must exist and all of them must be red.
        "all_red": bool(results) and all(entry["status"] == RED for entry in results),
        "tests": results,
        "output": tail(stdout + stderr),
    }


@mcp.tool()
def write_task_spec(type: str, short_description: str, spec: str) -> dict:
    """Save the task's type, short description and spec. Call once, before writing tests.

    type is one of feat, fix, refactor, test, docs, chore, perf, style, build
    or ci. short_description is a lowercase slug such as "delete-book"; it names
    the branch. spec is the scoped description of the change in plain English:
    what must change, what must not, and how the tests decide it is done.
    """
    # Each rule is checked in turn so the error names the first thing to fix.
    if type not in TASK_TYPES:
        return {"saved": False, "error": f"type must be one of {sorted(TASK_TYPES)}, got {type!r}"}
    if not SLUG_PATTERN.match(short_description) or len(short_description) > SLUG_LIMIT:
        return {"saved": False, "error": "short_description must be lowercase words joined by hyphens, "
                                         f"at most {SLUG_LIMIT} characters, like 'delete-book'"}
    if not spec.strip():
        return {"saved": False, "error": "spec must not be empty"}
    task_folder = get_task_folder()
    task_folder.mkdir(parents=True, exist_ok=True)
    # The same shape load_task in run_implementer reads, so the folder can go straight to the implementer.
    task = {"type": type, "short_description": short_description, "spec": spec.strip()}
    (task_folder / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    return {"saved": True, "task_folder": str(task_folder)}


@mcp.tool()
def write_task_test(file_name: str, content: str) -> dict:
    """Save one pytest file for the task. Call once per file; calling again replaces it.

    file_name is a bare module name such as "test_delete_book.py", with no
    folders. It must not be the name of a test file the repository already has,
    because the file will be added to the repository's tests folder. content is
    the complete file. Tests must reach the app only through its test client
    and assert the status code before reading a response body.
    """
    if not TEST_FILE_PATTERN.match(file_name):
        return {"saved": False, "error": "file_name must look like 'test_something.py', with no folders"}
    # Prepare copies generated tests into the repository's tests folder and then protects every file there.
    # A name clash would overwrite an existing suite and protect the overwritten version as if it were original.
    if (get_repo_path() / TESTS_DIR / file_name).exists():
        return {"saved": False, "error": f"{file_name} already exists in the repository's {TESTS_DIR} folder; choose a new name"}
    if not content.strip():
        return {"saved": False, "error": "content must not be empty"}
    # Mirrors the layout of a hand-written task folder, with tests in a tests subfolder next to task.json.
    tests_folder = get_task_folder() / TESTS_DIR
    tests_folder.mkdir(parents=True, exist_ok=True)
    path = tests_folder / file_name
    path.write_text(content, encoding="utf-8")
    return {"saved": True, "path": str(path)}


@mcp.tool()
def git_push() -> dict:
    """Push the current branch to the origin remote.

    Only the branch that is currently checked out is pushed. Protected branches
    such as main are refused, whatever the request.
    """
    repo_path = get_repo_path()
    # The branch comes from git, not from the caller, so the model cannot name a different one.
    exit_code, branch, _ = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_path)
    branch = branch.strip()
    # git prints the literal "HEAD" when the checkout is detached, which has no branch to push.
    if exit_code != 0 or not branch or branch == "HEAD":
        return {"pushed": False, "error": "Could not determine the current branch"}
    if branch in get_protected_branches():
        return {"pushed": False, "error": f"Pushing to protected branch '{branch}' is not allowed"}
    # -u sets the upstream so later pushes and pull request creation know which remote branch this is.
    exit_code, stdout, stderr = run_command(["git", "push", "-u", "origin", branch], repo_path)
    # git writes its progress to stderr, so both streams are kept for the caller.
    return {"pushed": exit_code == 0, "branch": branch, "output": tail(stdout + stderr, 1000)}


@mcp.tool()
def notify_user(
    outcome: Literal["ready_to_review", "blocked"],
    summary: str,
    link: str = "",
) -> dict:
    """Report the outcome of a run to the developer.

    Call exactly once at the end of every run. Use "ready_to_review" when a
    pull request was opened and "blocked" when the work could not be completed.
    """
    # The default is relative to the server's working directory; resolve() makes the returned path absolute.
    log_path = Path(os.environ.get("NOTIFICATIONS_LOG", "runs/notifications.log")).resolve()
    # The runs folder is gitignored and may not exist yet on a fresh checkout.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # UTC keeps the log consistent no matter which machine or timezone the pipeline runs in.
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    label = "READY TO REVIEW" if outcome == "ready_to_review" else "BLOCKED"
    # The link, usually the pull request URL, goes on its own indented line when there is one.
    message = f"[{timestamp}] {label}: {summary}" + (f"\n  {link}" if link else "")
    # Append mode keeps the history of every run rather than only the latest.
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")
    # stdout carries the MCP protocol, so human-readable output goes to stderr.
    print(message, file=sys.stderr)
    return {"delivered": True, "log": str(log_path)}


if __name__ == "__main__":
    # stdio transport: the client launches this script as a subprocess and talks JSON-RPC over its stdin and stdout.
    mcp.run(transport="stdio")
