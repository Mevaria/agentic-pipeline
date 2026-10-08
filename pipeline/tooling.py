"""Helpers for calling MCP tools directly from pipeline code.

Agents call tools through the model. Pipeline code also calls some tools itself,
at fixed points where no judgement is needed, such as creating a branch.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

# The JSON-lines file every tool call of the current run is appended to, or None when no run is being traced.
_TRACE_PATH = None


def start_trace(path):
    """Start recording tool calls to a JSON-lines file; returns True if this call started it, False if one was active."""
    global _TRACE_PATH
    if _TRACE_PATH is not None:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _TRACE_PATH = path
    return True


def stop_trace():
    """Stop recording tool calls."""
    global _TRACE_PATH
    _TRACE_PATH = None


def current_trace():
    """Return the path of the active tool log, or None."""
    return _TRACE_PATH


def trace_tool(source, name, arguments, result=None, error=None):
    """Append one tool call to the active tool log: who called it, the arguments, and the result or error, scrubbed."""
    if _TRACE_PATH is None:
        return
    entry = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # "code" for calls the pipeline made itself, "model" for calls an agent asked for.
        "source": source,
        "tool": name,
        "arguments": json.loads(scrub_secrets(json.dumps(arguments, default=str))),
        "result": scrub_secrets(result) if result is not None else None,
        "error": scrub_secrets(error) if error is not None else None,
    }
    with _TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def record_tool_results(messages):
    """Write the latest model-requested tool calls and their results to the tool log, matching results to calls by id."""
    results = latest_tool_results(messages)
    if not results:
        return
    # The AIMessage before the results holds the calls, with the arguments the model chose.
    calls = {}
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            calls = {call["id"]: call for call in message.tool_calls}
            break
    for result in results:
        call = calls.get(result.tool_call_id, {})
        content = tool_text(result.content)
        trace_tool("model", result.name or call.get("name"), call.get("args", {}),
                   result=None if result.status == "error" else content, error=content if result.status == "error" else None)

# Environment variables whose values must never appear in a log or a record.
SECRET_NAME_PATTERN = re.compile(r"TOKEN|SECRET|PASSWORD|API_KEY", re.I)
# Shorter values are not treated as secrets, so a variable like "1" cannot blank out every "1" in a log.
SECRET_MIN_LENGTH = 8


def scrub_secrets(text):
    """Return text with the value of every secret-looking environment variable replaced by <redacted>."""
    scrubbed = str(text)
    for name, value in os.environ.items():
        if SECRET_NAME_PATTERN.search(name) and len(value) >= SECRET_MIN_LENGTH and value in scrubbed:
            scrubbed = scrubbed.replace(value, "<redacted>")
    return scrubbed


def innermost_error(error):
    """Return "Type: message" for the innermost exception inside nested exception groups.

    MCP sessions run in anyio task groups, so an error inside a graph reaches
    the caller wrapped in one or more ExceptionGroups; this digs it out.
    """
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    # The adapter raises a generic "could not list tools" error; the cause underneath says why the server died.
    if error.__context__ is not None and "tools" in str(error):
        error = error.__context__
        while isinstance(error, BaseExceptionGroup) and error.exceptions:
            error = error.exceptions[0]
    return f"{type(error).__name__}: {error}"


def latest_tool_results(messages):
    """Return the ToolMessages produced for the most recent AIMessage, in order."""
    results = []
    # Walk back from the end; the tool results sit after the AIMessage that asked for them.
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            break
        if isinstance(message, ToolMessage):
            results.insert(0, message)
    return results


def tool_text(result):
    """Flatten an MCP tool result into plain text.

    MCP tools return a list of content blocks; text blocks are joined together.
    """
    # Some adapters already hand back a plain string.
    if isinstance(result, str):
        return result
    # Content blocks are dicts with a "text" key; anything else is rendered with str() so nothing is lost.
    if isinstance(result, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in result
        )
    return str(result)


async def call_tool(tools_by_name, name, arguments):
    """Call one MCP tool by name and return its result as text."""
    # Naming the tool in the error makes a missing server easy to spot.
    if name not in tools_by_name:
        raise RuntimeError(f"Tool not available: {name}")
    # ainvoke sends the call to the server over the open session and waits for the result.
    try:
        result = tool_text(await tools_by_name[name].ainvoke(arguments))
    except BaseException as error:
        trace_tool("code", name, arguments, error=innermost_error(error))
        raise
    trace_tool("code", name, arguments, result=result)
    return result


async def open_tools(exit_stack, client, server_names):
    """Open one long-lived session per server and load its tools.

    By default the MCP client starts a fresh server process for every tool call.
    Holding a session open for the whole run starts each server once instead.
    The sessions close when exit_stack closes.
    """
    # Imported here so that importing this module does not require the adapter package.
    from langchain_mcp_adapters.tools import load_mcp_tools

    tools = []
    for name in server_names:
        # Entering the session on the exit stack keeps it open until the caller's stack closes.
        session = await exit_stack.enter_async_context(client.session(name))
        # Tools loaded from an open session reuse it for every call instead of spawning a new process.
        tools.extend(await load_mcp_tools(session, server_name=name))
    return tools
