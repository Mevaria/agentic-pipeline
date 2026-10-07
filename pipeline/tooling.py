"""Helpers for calling MCP tools directly from pipeline code.

Agents call tools through the model. Pipeline code also calls some tools itself,
at fixed points where no judgement is needed, such as creating a branch.
"""

from langchain_core.messages import AIMessage, ToolMessage


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
    return tool_text(await tools_by_name[name].ainvoke(arguments))


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
