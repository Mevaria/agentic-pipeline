"""Index every run record into the Chroma collection, or query it to see distances.

    python -m pipeline.index_records                 # index runs/records/*.json
    python -m pipeline.index_records --query "text"  # show the nearest runs and their distances

Indexing is idempotent: a record already in the collection is updated. Used
to backfill records written before memory existed, and to pick thresholds
from real distances.
"""

import argparse
import asyncio
from contextlib import AsyncExitStack

from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import get_chroma_server_config, get_memory_settings, get_reporter_settings
from pipeline.feedback import load_records
from pipeline.memory import index_record, recall
from pipeline.tooling import open_tools


async def main(query):
    """Start the Chroma server alone and either index every run record or run one query."""
    settings = get_memory_settings()
    server_config = get_chroma_server_config()
    async with AsyncExitStack() as exit_stack:
        tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        tools_by_name = {tool.name: tool for tool in tools}
        if query:
            hits = await recall(tools_by_name, settings, query)
            print(f"Nearest runs to: {query!r}\n")
            for hit in hits:
                print(f"{hit['distance']:.3f}  {hit['id']}  {hit['text'].splitlines()[0][:100]}")
            if not hits:
                print("(the collection is empty)")
            return
        # A feedback record carries the run it started, if any, nested under "run"; that run is what gets indexed.
        indexed = 0
        for _, record in load_records(get_reporter_settings()["records_path"]):
            run = record.get("run") if record.get("kind") == "feedback" else record
            if run and await index_record(tools_by_name, settings, run):
                indexed += 1
                print(f"indexed {run['run_id']}: {run.get('outcome')}")
        print(f"\n{indexed} run record(s) indexed into collection {settings['collection']!r}")


if __name__ == "__main__":
    # The first line of the module docstring doubles as the command's help text.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", help="show the nearest runs to this text instead of indexing")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.query))
