"""Writing run records.

A record is one JSON file per run, written by whichever runner is at the top:
the orchestrator for a full run, run_report for a standalone review and
report. Stages return their part of the record; only the top-level runner
writes, so a run never produces two competing files.
"""

import json
from pathlib import Path

from pipeline.tooling import scrub_secrets


def write_record(records_path, run_id, record):
    """Write the record as <records_path>/<run_id>.json, with secret values scrubbed, and return the path."""
    folder = Path(records_path)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_id}.json"
    # Scrubbing the serialised text catches a secret wherever it ended up, including inside error messages.
    path.write_text(scrub_secrets(json.dumps(record, indent=2)) + "\n", encoding="utf-8")
    return path
