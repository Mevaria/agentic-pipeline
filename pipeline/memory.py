"""Run memory on Chroma: index run records, recall similar past runs for the author.

The records folder stays the source of truth; the Chroma collection is derived
from it, one document per run record, with the structured fields kept as
metadata. Code does the indexing and the recall; no model ever gets a Chroma
tool. Recall returns only past runs whose cosine distance to the new request
is under a threshold, because unrelated history is noise for a small model,
and it says when a near-identical request already has an open pull request so
the author can ask the developer whether to revise that one instead.
"""

import json

from pipeline.tooling import call_tool

# Chroma's default space is squared L2; cosine distance (0 identical, 1 unrelated) is easier to threshold.
COLLECTION_METADATA = {"hnsw:space": "cosine"}
# Characters of a document kept in the author's prompt per past run.
EXCERPT_LIMIT = 300


def run_document(record):
    """Turn a run record into (id, text, metadata) for the collection; returns None for a record without a request."""
    # Records written by the standalone review and report runner have no request; the spec stands in for it.
    request = record.get("request") or (record.get("task") or {}).get("spec")
    if not request or not record.get("run_id"):
        return None
    stages = record.get("stages") or {}
    author = stages.get("author") or {}
    implementer = stages.get("implementer") or {}
    reporter = stages.get("reporter") or {}
    gate = stages.get("gate") or {}
    pull_request = (reporter.get("steps") or {}).get("pull_request") or {}
    task = record.get("task") or {}
    # Only the request is embedded, so distance measures how alike two requests are; adding the outcome and
    # branch to the text was measured to push a short request 0.2 away from its own record.
    # Metadata values must be scalars for Chroma; a missing pull request number is stored as 0.
    metadata = {
        "kind": "run",
        "run_id": record.get("run_id", ""),
        "outcome": record.get("outcome") or "",
        "task_type": task.get("type") or "",
        "short_description": task.get("short_description") or "",
        "branch": implementer.get("branch") or "",
        "pull_request": int(pull_request.get("number") or 0),
        "finished": record.get("finished") or "",
        "question": (author.get("question") or "")[:EXCERPT_LIMIT],
        "reason": author.get("reason") or "",
        "review": (gate["rounds"][-1].get("summary", "") if gate.get("rounds") else "")[:EXCERPT_LIMIT],
        "error": (record.get("error") or "")[:EXCERPT_LIMIT],
    }
    return record.get("run_id"), request, metadata


def document_text(result):
    """Return the text of a Chroma tool result, which the server wraps as content blocks or returns as a string."""
    if isinstance(result, str):
        return result
    return "".join(block.get("text", "") for block in result if isinstance(block, dict))


async def ensure_collection(tools_by_name, name):
    """Create the collection with cosine distance if it does not exist yet."""
    listing = await call_tool(tools_by_name, "chroma_list_collections", {})
    # The listing tool returns a JSON list of names or a sentence saying there are none.
    try:
        names = json.loads(listing)
    except ValueError:
        names = []
    if name not in names:
        await call_tool(tools_by_name, "chroma_create_collection", {"collection_name": name, "metadata": COLLECTION_METADATA})


async def index_record(tools_by_name, settings, record):
    """Add or update the run's document in the collection; returns the document id, or None if the record has no request."""
    document = run_document(record)
    if document is None:
        return None
    doc_id, text, metadata = document
    collection = settings["collection"]
    await ensure_collection(tools_by_name, collection)
    # Chroma refuses to add an existing id, so an indexed run is updated instead, which keeps reruns idempotent.
    existing = json.loads(await call_tool(tools_by_name, "chroma_get_documents", {"collection_name": collection, "ids": [doc_id], "include": []}))
    if doc_id in (existing.get("ids") or []):
        await call_tool(tools_by_name, "chroma_update_documents",
                        {"collection_name": collection, "ids": [doc_id], "documents": [text], "metadatas": [metadata]})
    else:
        await call_tool(tools_by_name, "chroma_add_documents",
                        {"collection_name": collection, "ids": [doc_id], "documents": [text], "metadatas": [metadata]})
    return doc_id


async def recall(tools_by_name, settings, request):
    """Return the past runs closest to the request, nearest first, each with its distance, text and metadata."""
    collection = settings["collection"]
    await ensure_collection(tools_by_name, collection)
    count = int(json.loads(await call_tool(tools_by_name, "chroma_get_collection_count", {"collection_name": collection})) or 0)
    if count == 0:
        return []
    reply = json.loads(await call_tool(tools_by_name, "chroma_query_documents", {
        "collection_name": collection, "query_texts": [request], "n_results": min(settings["max_results"], count),
        "include": ["documents", "metadatas", "distances"],
    }))
    hits = []
    for doc_id, text, metadata, distance in zip(reply["ids"][0], reply["documents"][0], reply["metadatas"][0], reply["distances"][0]):
        # Cosine distance for an identical text can come back as a tiny negative number; it is clamped at zero.
        hits.append({"id": doc_id, "distance": max(0.0, round(distance, 4)), "text": text, "metadata": metadata})
    return hits


def relevant_hits(hits, settings):
    """Keep only the hits under the distance threshold; an empty list means nothing is shown to the author."""
    return [hit for hit in hits if hit["distance"] <= settings["distance_threshold"]]


def recall_block(hits, settings, pull_request_states):
    """Return the memory text for the author's prompt, or "" when there is nothing relevant.

    pull_request_states maps a pull request number to "open", "closed" or
    "merged" when known. A hit under the duplicate threshold whose pull request
    is open is stated as a near-identical request with an open pull request,
    which is the author's cue to ask whether to revise it instead.
    """
    if not hits:
        return ""
    lines = ["Past runs of this pipeline that resemble this request, from memory:"]
    for hit in hits:
        metadata = hit["metadata"]
        number = metadata.get("pull_request") or 0
        state = pull_request_states.get(number, "unknown") if number else None
        # The request is the embedded text; the rest of what happened comes from the metadata.
        details = [f"outcome {metadata.get('outcome') or 'unknown'}"]
        if metadata.get("task_type"):
            details.append(f"task {metadata['task_type']}/{metadata.get('short_description', '')}")
        if metadata.get("question"):
            details.append(f"asked ({metadata.get('reason')}): {metadata['question']}")
        if metadata.get("review"):
            details.append(f"review: {metadata['review']}")
        if metadata.get("error"):
            details.append(f"crashed: {metadata['error']}")
        line = f"- Run {hit['id']} (distance {hit['distance']:.2f}): request \"{hit['text'][:EXCERPT_LIMIT]}\"; {'; '.join(details)}"
        if number:
            line += f" [pull request #{number} is {state}]"
        lines.append(line)
        if number and state == "open" and hit["distance"] <= settings["duplicate_threshold"]:
            lines.append(f"  An open pull request #{number} already implements a near-identical request.")
    return "\n".join(lines)
