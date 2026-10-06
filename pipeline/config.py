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
