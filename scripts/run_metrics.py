"""Trajectory metrics for every run, measured from the records and tool logs under runs/records.

    python -m scripts.run_metrics
    python -m scripts.run_metrics --records runs/records

Prints one table per run: tool calls in total and by who asked for them, calls
that failed or were refused, the author's revisions, the implementer's
attempts, the review rounds, the fail-first checks, the outcome and the
duration. Every number comes from the files; nothing is estimated. A tool log
without a record, such as a standalone review, gets a table from the log
alone, with its duration taken from the first and last logged call.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

# Keys in a JSON tool result that, when false, mean the server refused or could not do what was asked.
REFUSAL_FLAGS = ("saved", "pushed", "restored")


def load_log(path):
    """Return the entries of a tool log, skipping lines that are not JSON."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    return entries


def parsed_result(entry):
    """Return the entry's result as a dict when it is JSON for one, otherwise None."""
    try:
        data = json.loads(entry.get("result") or "")
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_refused(entry):
    """True when the tool answered rather than raised, but said no: a false saved, pushed or restored flag, or an error field."""
    data = parsed_result(entry)
    if data is None:
        return False
    if any(data.get(flag) is False for flag in REFUSAL_FLAGS):
        return True
    return bool(data.get("error"))


def minutes_between(start, finish):
    """Return the minutes between two ISO timestamps, or None when either is missing or unreadable."""
    try:
        return round((datetime.fromisoformat(finish) - datetime.fromisoformat(start)).total_seconds() / 60, 1)
    except (TypeError, ValueError):
        return None


def log_metrics(entries):
    """Count what the tool log shows: calls in total, by source, failed, refused, by tool, and the fail-first checks."""
    by_tool = {}
    for entry in entries:
        by_tool[entry.get("tool")] = by_tool.get(entry.get("tool"), 0) + 1
    checks = [parsed_result(entry) or {} for entry in entries if entry.get("tool") == "check_tests_fail"]
    test_runs = [parsed_result(entry) or {} for entry in entries if entry.get("tool") == "run_tests"]
    times = [entry.get("time") for entry in entries if entry.get("time")]
    return {
        "tool calls": len(entries),
        "model calls": sum(entry.get("source") == "model" for entry in entries),
        "code calls": sum(entry.get("source") == "code" for entry in entries),
        "failed calls": sum(entry.get("error") is not None for entry in entries),
        "refused calls": sum(is_refused(entry) for entry in entries),
        "fail-first checks": f"{len(checks)} ({sum(bool(check.get('all_red')) for check in checks)} all red)",
        "test runs": f"{len(test_runs)} ({sum(bool(run.get('passed')) for run in test_runs)} passed)",
        "calls by tool": ", ".join(f"{name} {count}" for name, count in sorted(by_tool.items(), key=lambda item: (-item[1], str(item[0])))),
        "log span (min)": minutes_between(times[0], times[-1]) if len(times) > 1 else None,
    }


def record_metrics(record):
    """Read the stage counts and the duration from a run record (a feedback record's nested run is read the same way)."""
    stages = record.get("stages") or {}
    author = stages.get("author") or {}
    implementer = stages.get("implementer") or {}
    gate = stages.get("gate") or {}
    metrics = {
        "outcome": record.get("outcome"),
        "author revisions": author.get("revisions") if author else None,
        "implementer attempts": implementer.get("attempts") if implementer else None,
        "review rounds": len(gate.get("rounds", [])) if gate else None,
        "gate fixes": len(gate.get("fixes", [])) if gate else None,
        "duration (min)": minutes_between(record.get("started"), record.get("finished")),
    }
    if record.get("issue"):
        metrics["issue"] = f"#{record['issue'].get('number')}"
    pull_request = ((stages.get("reporter") or {}).get("steps") or {}).get("pull_request") or {}
    if pull_request.get("number"):
        metrics["pull request"] = f"#{pull_request['number']}"
    return metrics


def collect(records_path):
    """Pair every record with its tool log and return one {"name", "metrics"} per run, logs without a record included."""
    folder = Path(records_path)
    logs = {path.name: path for path in folder.glob("*-tools.jsonl")}
    runs = []
    for path in sorted(folder.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        # A feedback record wraps the run it triggered, if any; the metrics are that run's, plus the action taken.
        run = record.get("run") if record.get("kind") == "feedback" else record
        metrics = {"kind": record.get("kind") or "run"}
        if record.get("kind") == "feedback":
            metrics["action"] = record.get("action")
            metrics["pull request"] = f"#{record.get('pull_request')}"
        if run:
            metrics.update(record_metrics(run))
            log_name = Path(run.get("tool_log") or "").name or f"{run.get('run_id')}-tools.jsonl"
            if log_name in logs:
                metrics.update(log_metrics(load_log(logs.pop(log_name))))
            else:
                metrics["tool log"] = "none (run predates tool logging)"
        runs.append({"name": path.stem, "metrics": metrics})
    # Logs that no record claimed: standalone reviews and reports, measured from the log alone.
    for name, path in sorted(logs.items()):
        runs.append({"name": name.removesuffix("-tools.jsonl"), "metrics": {"kind": "log only", **log_metrics(load_log(path))}})
    return runs


def render(runs):
    """Return the runs as Markdown, one two-column table per run."""
    blocks = []
    for run in runs:
        rows = "\n".join(f"| {key} | {'' if value is None else value} |" for key, value in run["metrics"].items())
        blocks.append(f"### {run['name']}\n\n| Metric | Value |\n|---|---|\n{rows}\n")
    return "\n".join(blocks)


if __name__ == "__main__":
    # The first line of the module docstring doubles as the command's help text.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", default=str(Path(__file__).resolve().parent.parent / "runs" / "records"), help="the records folder")
    arguments = parser.parse_args()
    print(render(collect(arguments.records)))
