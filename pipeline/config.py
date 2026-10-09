"""Shared configuration for the pipeline.

Loads settings from .env, defines how each MCP server is launched, and lists
which tools each agent may use. Agents only ever receive the tools on their
allowlist, so least privilege is enforced here rather than in prompts.
"""

import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# The repository root: this file lives in pipeline/, so one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Read .env into the environment once, at import time, so every getter below can use os.environ.
# Variables already set in the shell take precedence over the file.
load_dotenv(PROJECT_ROOT / ".env")

# The npm package for the reference Filesystem MCP server, fetched by npx on first use.
FILESYSTEM_SERVER_PACKAGE = "@modelcontextprotocol/server-filesystem"
# The only GitHub tools the server is started with. Given alone, GITHUB_TOOLS makes the server expose exactly
# these (verified live against v2.0.2), so merge_pull_request is never offered to anything.
GITHUB_TOOLS_ALLOWED = ["create_pull_request", "list_pull_requests", "pull_request_read", "add_issue_comment"]


def get_target_repo_path():
    """Return the target repository path from TARGET_REPO_PATH, checking it is a git repository."""
    repo_path = os.environ.get("TARGET_REPO_PATH")
    # Fail with setup instructions rather than guessing a repository.
    if not repo_path:
        raise RuntimeError("TARGET_REPO_PATH is not set. Copy .env.example to .env and set it.")
    # Resolving gives the servers an absolute path with no symlinks or ".." segments.
    path = Path(repo_path).resolve()
    if not (path / ".git").exists():
        raise RuntimeError(f"{path} is not a git repository")
    return path


def get_model_name():
    """Return the Ollama model name the agents run on, defaulting to gemma4:e4b."""
    return os.environ.get("MODEL_NAME", "gemma4:e4b")


def get_int(name, default):
    """Return the environment variable `name` as an int, or `default` when it is unset."""
    # The default is passed through str() so the same int() conversion applies to both paths.
    value = os.environ.get(name, str(default))
    try:
        return int(value)
    # A typo in .env should stop the run with a clear message rather than a stack trace from deep inside.
    except ValueError:
        raise RuntimeError(f"{name} must be a whole number, got {value!r}")


def get_context_size():
    """Tokens of context the model is given; file contents and test output fill it quickly."""
    # 8192 is the largest value that ran without a CUDA error on the development GPU.
    return get_int("NUM_CTX", 8192)


def get_implementer_settings():
    """Return the implementer's settings: base branch, attempt limit and tool steps per attempt."""
    return {
        # The branch new work branches from and is compared against.
        "base_branch": os.environ.get("BASE_BRANCH", "main"),
        # Attempts before the run stops as blocked.
        "max_attempts": get_int("MAX_ATTEMPTS", 3),
        # Tool calls the model may make within one attempt before verification is forced.
        "max_tool_steps": get_int("MAX_TOOL_STEPS", 12),
    }


def get_author_settings():
    """Return the spec and test author's settings: base branch, revision cap and tool steps per attempt."""
    return {
        # The branch the author reads and the fail-first check runs against; the same one the implementer uses.
        "base_branch": os.environ.get("BASE_BRANCH", "main"),
        # Revisions after the first attempt before the run stops as blocked or not reproducible.
        "max_revisions": get_int("MAX_AUTHOR_REVISIONS", 2),
        # Tool calls the model may make within one attempt before the fail-first check is forced.
        "max_tool_steps": get_int("MAX_TOOL_STEPS", 12),
    }


def get_review_settings():
    """Return the review gate's settings: base branch, review-round cap and tool steps per attempt."""
    return {
        # The branch the change is diffed against, the same one the other agents use.
        "base_branch": os.environ.get("BASE_BRANCH", "main"),
        # Reviews before the gate stops as blocked, counting the first one; each block in between sends
        # the findings to the implementer.
        "max_review_rounds": get_int("MAX_REVIEW_ROUNDS", 2),
        # Model turns the reviewer gets to produce its verdict before the gate fails closed.
        "max_tool_steps": get_int("MAX_TOOL_STEPS", 12),
        # Whether a second, security-only model call runs after the review over the same evidence.
        "security_pass": get_flag("SECURITY_PASS", False),
        # When set, the gate sends each change's evidence to the reviewer service at this URL instead of
        # reviewing in-process. Empty means local.
        "a2a_url": os.environ.get("REVIEWER_A2A_URL", "").strip(),
    }


def get_flag(name, default):
    """Return the environment variable `name` as a boolean: 1, true, yes and on count as True."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_reporter_settings():
    """Return the reporter's settings: base branch, where run records go, and an optional repository override."""
    return {
        # The branch pull requests are opened against.
        "base_branch": os.environ.get("BASE_BRANCH", "main"),
        # One JSON record per run, kept for the developer and for the memory step to index later.
        "records_path": Path(os.environ.get("RECORDS_PATH", PROJECT_ROOT / "runs" / "records")).resolve(),
        # "owner/repo" on GitHub. Normally derived from the target's origin remote; set only to override it.
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        # Branches the cleanup fallback may never delete; the same list the dev tools server reads.
        "protected_branches": {name.strip() for name in os.environ.get("PROTECTED_BRANCHES", "main,master").split(",") if name.strip()},
    }


def get_feedback_settings():
    """Return the feedback loop's settings: revisions allowed per pull request."""
    return {
        # /revise comments acted on per pull request before the loop stops as blocked.
        "max_revisions": get_int("MAX_PR_REVISIONS", 2),
    }


def get_float(name, default):
    """Return the environment variable `name` as a float, or `default` when it is unset."""
    value = os.environ.get(name, str(default))
    try:
        return float(value)
    except ValueError:
        raise RuntimeError(f"{name} must be a number, got {value!r}")


def get_memory_settings():
    """Return the memory settings: collection name, how many past runs to look at, and the distance thresholds."""
    return {
        # The Chroma collection that holds one document per run record.
        "collection": os.environ.get("MEMORY_COLLECTION", "runs"),
        # Past runs fetched per query; only those under the threshold are shown to the author.
        "max_results": get_int("MEMORY_MAX_RESULTS", 3),
        # Cosine distance (0 identical, 1 unrelated) above which a past run is not shown at all. From real queries:
        # the same request scores 0.0, the same topic reworded 0.23 to 0.49, a different topic on the same app
        # 0.44 and up, so 0.40 keeps clear matches and drops the band where the two overlap.
        "distance_threshold": get_float("MEMORY_DISTANCE_THRESHOLD", 0.40),
        # Cosine distance under which a past run with an open pull request counts as a near-identical request.
        "duplicate_threshold": get_float("MEMORY_DUPLICATE_THRESHOLD", 0.20),
    }


def get_chroma_server_config():
    """Launch settings for the Chroma MCP server in persistent mode, from its own virtual environment.

    chroma-mcp pins an old mcp release that cannot share our environment, so it
    lives in .venv-chroma; CHROMA_MCP_SERVER overrides the executable's path and
    CHROMA_DATA_PATH where the database is kept (default runs/chroma).
    """
    default_binary = PROJECT_ROOT / ".venv-chroma" / "Scripts" / "chroma-mcp.exe"
    binary = os.environ.get("CHROMA_MCP_SERVER", str(default_binary))
    if not Path(binary).is_file():
        raise RuntimeError("CHROMA_MCP_SERVER is not set or does not point to chroma-mcp. "
                           "Create .venv-chroma and install chroma-mcp into it, or set the path in .env.")
    data_path = Path(os.environ.get("CHROMA_DATA_PATH", PROJECT_ROOT / "runs" / "chroma")).resolve()
    return {
        "chroma": {
            "command": binary,
            "args": ["--client-type", "persistent", "--data-dir", str(data_path)],
            "transport": "stdio",
            "env": {**os.environ},
        },
    }


def get_github_server_config():
    """Launch settings for the official GitHub MCP server, started with only the allowed tools.

    Requires GITHUB_MCP_SERVER (path to the binary) and GITHUB_PERSONAL_ACCESS_TOKEN
    in the environment; raises with setup instructions when either is missing.
    """
    binary = os.environ.get("GITHUB_MCP_SERVER")
    if not binary or not Path(binary).is_file():
        raise RuntimeError("GITHUB_MCP_SERVER is not set or does not point to github-mcp-server.exe. "
                           "Download the release binary and set its path in .env.")
    if not os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN"):
        raise RuntimeError("GITHUB_PERSONAL_ACCESS_TOKEN is not set. Create a fine-grained token scoped to the "
                           "target repository with Pull requests read and write, and put it in .env.")
    return {
        "github": {
            "command": binary,
            # stdio is the server's subcommand for the transport the other servers use too.
            "args": ["stdio"],
            "transport": "stdio",
            # GITHUB_TOOLS alone limits the server to these tools; GITHUB_TOOLSETS is left unset on purpose,
            # because naming any toolset would add that group's tools on top.
            "env": {**os.environ, "GITHUB_TOOLS": ",".join(GITHUB_TOOLS_ALLOWED)},
        },
    }


def get_tasks_path():
    """Return the folder generated task folders go under, defaulting to runs/tasks in this repository."""
    # The dev tools server reads the same variable with the same default, so both sides agree on the folder.
    return Path(os.environ.get("TASKS_PATH", PROJECT_ROOT / "runs" / "tasks")).resolve()


def new_run_id():
    """Return a run id from the current UTC time, which names the run's task folder."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def find_npx():
    """Return the full path to npx, failing with install instructions when Node.js is missing."""
    # On Windows npx is a .cmd file, which shutil.which resolves to its full path.
    npx = shutil.which("npx")
    if npx is None:
        raise RuntimeError("npx was not found. Install Node.js to run the Filesystem MCP server.")
    return npx


def get_server_config():
    """Launch settings for the three MCP servers, in the format MultiServerMCPClient expects."""
    # The servers take a plain string path on their command lines.
    repo_path = str(get_target_repo_path())
    # Without an explicit environment, MCP servers start with a reduced set of
    # variables. That can drop proxy settings (slowing npx badly) and the home
    # folder git reads your name and email from, so every server gets the full one.
    environment = {**os.environ}
    return {
        # Reference Filesystem server, run through npx. The path argument restricts it to the target repository.
        "filesystem": {
            "command": find_npx(),
            "args": ["-y", FILESYSTEM_SERVER_PACKAGE, repo_path],
            "transport": "stdio",
            "env": environment,
        },
        # Reference Git server, installed in the same .venv, pointed at the target repository.
        "git": {
            "command": sys.executable,
            "args": ["-m", "mcp_server_git", "--repository", repo_path],
            "transport": "stdio",
            "env": environment,
        },
        # This project's own server. It reads TARGET_REPO_PATH and the other settings from the environment.
        "dev_tools": {
            "command": sys.executable,
            "args": [str(PROJECT_ROOT / "servers" / "dev_tools_server.py")],
            "transport": "stdio",
            "env": environment,
        },
    }


# Which tools each agent may call. Anything not listed is never given to that agent.
# Steps that need no judgement, such as creating the branch and committing, are
# done by the pipeline code at fixed points rather than offered to a model.
AGENT_TOOLS = {
    "implementer": [
        # Filesystem server. directory_tree is left out because it floods a small model with .git contents,
        # and edit_file is left out because it needs exact text matches that a small model fumbles.
        "list_directory",
        "read_text_file",
        "write_file",
        # Dev tools server. The model may run tests, but the run that decides pass or fail is done by code.
        "run_tests",
    ],
    "author": [
        # Filesystem server, read-only: the author studies the code but never changes the target repository.
        "list_directory",
        "read_text_file",
        # Dev tools server: the author's only writes go into its own task folder. The fail-first check
        # is not listed, because code runs it at a fixed point; the model cannot judge its own tests.
        "write_task_spec",
        "write_task_test",
        # request_clarification is not an MCP tool; pipeline/author.py defines it and adds it to this list.
    ],
    # The reviewer gets no MCP tool at all. Code gathers the diff, the changed files, the scan and the
    # test record and puts them in its prompt; its only tool, submit_review, is defined in pipeline/reviewer.py.
    "reviewer": [],
    # The reporter is a code step with one model call that writes a paragraph; the model gets no tools.
    # Code calls git_push, notify_user and the GitHub tools itself.
    "reporter": [],
}


def select_tools(all_tools, agent_name):
    """Return only the tools on an agent's allowlist, and fail loudly if any are missing."""
    allowed = AGENT_TOOLS[agent_name]
    by_name = {tool.name: tool for tool in all_tools}
    # A missing tool usually means a server failed to start or was renamed; stop before the agent runs without it.
    missing = [name for name in allowed if name not in by_name]
    if missing:
        raise RuntimeError(f"Tools missing for {agent_name}: {missing}")
    # Returned in allowlist order so the model sees the tools in a stable order.
    return [by_name[name] for name in allowed]
