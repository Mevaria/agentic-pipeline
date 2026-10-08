"""Live test of the official GitHub MCP server's configuration.

Starts the real server binary with the pipeline's launch settings and checks
the tools it exposes. It needs GITHUB_MCP_SERVER and GITHUB_PERSONAL_ACCESS_TOKEN
in the environment (normally from .env) and is skipped, with a message saying
why, on a machine without them. Listing tools does not call GitHub's API.
"""

import asyncio
import os

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import GITHUB_TOOLS_ALLOWED, get_github_server_config

# Both must be set, or the server cannot start; the test says so instead of failing.
CONFIGURED = bool(os.environ.get("GITHUB_MCP_SERVER")) and bool(os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN"))


@pytest.mark.skipif(not CONFIGURED, reason="GITHUB_MCP_SERVER and GITHUB_PERSONAL_ACCESS_TOKEN are not both set")
def test_github_server_exposes_only_the_allowed_tools():
    """The GitHub server started from get_github_server_config exposes exactly the allowed tools, so merge_pull_request is never offered."""
    client = MultiServerMCPClient(get_github_server_config())
    tools = asyncio.run(client.get_tools(server_name="github"))
    names = sorted(tool.name for tool in tools)
    # Printed so a run with -s shows exactly what the server offered, as evidence for the report.
    print(f"GitHub server tools: {names}")
    assert names == sorted(GITHUB_TOOLS_ALLOWED)
    assert "merge_pull_request" not in names
