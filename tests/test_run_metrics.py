"""Tests for the trajectory metrics script, on a synthetic record and tool log."""

import json

from scripts.run_metrics import collect, is_refused, log_metrics, render


def entry(source, tool, result=None, error=None, time="2026-10-09T10:00:00+00:00"):
    """One tool log line as trace_tool writes it."""
    return {"time": time, "source": source, "tool": tool, "arguments": {}, "result": result, "error": error}


LOG = [
    entry("code", "git_status", "On branch main, working tree clean", time="2026-10-09T10:00:00+00:00"),
    entry("model", "write_task_test", json.dumps({"saved": False, "error": "test_app.py already exists"})),
    entry("model", "write_task_test", json.dumps({"saved": True, "path": "x"})),
    entry("code", "check_tests_fail", json.dumps({"all_red": False, "tests": []})),
    entry("code", "check_tests_fail", json.dumps({"all_red": True, "tests": []})),
    entry("model", "run_tests", json.dumps({"passed": False, "summary": "1 failed"})),
    entry("code", "run_tests", json.dumps({"passed": True, "summary": "2 passed"})),
    entry("model", "read_text_file", None, error="file not found"),
    entry("model", "submit_review", "Review recorded.", time="2026-10-09T10:03:00+00:00"),
]

RECORD = {
    "run_id": "run-1", "request": "x", "started": "2026-10-09T09:59:00+00:00", "finished": "2026-10-09T10:04:00+00:00",
    "tool_log": "C:/anywhere/run-1-tools.jsonl", "outcome": "ready_to_review", "issue": {"number": 6},
    "stages": {"author": {"status": "ready", "revisions": 1}, "implementer": {"status": "passed", "attempts": 2},
               "gate": {"status": "approved", "rounds": [{}, {}], "fixes": ["passed"]},
               "reporter": {"steps": {"pull_request": {"number": 7, "url": "u"}}}},
}


def test_refusals_are_false_flags_or_error_fields_not_failed_tests():
    """is_refused counts a false saved flag and an error field, but not a test run that merely failed or a plain-text result."""
    assert is_refused(LOG[1]) is True
    assert is_refused(LOG[2]) is False
    assert is_refused(LOG[5]) is False
    assert is_refused(LOG[0]) is False


def test_log_metrics_count_calls_sources_failures_checks_and_span():
    """log_metrics reports totals by source, failed and refused calls, the fail-first checks and test runs, and the span of the log."""
    metrics = log_metrics(LOG)
    assert metrics["tool calls"] == 9 and metrics["model calls"] == 5 and metrics["code calls"] == 4
    assert metrics["failed calls"] == 1 and metrics["refused calls"] == 1
    assert metrics["fail-first checks"] == "2 (1 all red)" and metrics["test runs"] == "2 (1 passed)"
    assert metrics["calls by tool"].startswith("check_tests_fail 2, run_tests 2, write_task_test 2, ")
    assert metrics["log span (min)"] == 3.0


def test_collect_pairs_records_with_logs_and_keeps_orphan_logs(tmp_path):
    """collect joins a record with its log by file name, reads a feedback record through its nested run, and lists a log with no record on its own."""
    (tmp_path / "run-1.json").write_text(json.dumps(RECORD), encoding="utf-8")
    (tmp_path / "run-1-tools.jsonl").write_text("\n".join(json.dumps(line) for line in LOG), encoding="utf-8")
    feedback = {"kind": "feedback", "action": "revised", "pull_request": 7, "run": {**RECORD, "run_id": "run-2", "tool_log": "run-2-tools.jsonl"}}
    (tmp_path / "feedback-7-run-2.json").write_text(json.dumps(feedback), encoding="utf-8")
    (tmp_path / "task-review-run-3-tools.jsonl").write_text(json.dumps(LOG[-1]) + "\n", encoding="utf-8")
    runs = {run["name"]: run["metrics"] for run in collect(tmp_path)}
    assert set(runs) == {"run-1", "feedback-7-run-2", "task-review-run-3"}
    first = runs["run-1"]
    assert first["kind"] == "run" and first["outcome"] == "ready_to_review" and first["issue"] == "#6" and first["pull request"] == "#7"
    assert first["author revisions"] == 1 and first["implementer attempts"] == 2 and first["review rounds"] == 2 and first["gate fixes"] == 1
    assert first["duration (min)"] == 5.0 and first["tool calls"] == 9
    assert runs["feedback-7-run-2"]["action"] == "revised" and runs["feedback-7-run-2"]["tool log"].startswith("none")
    assert runs["task-review-run-3"]["kind"] == "log only" and runs["task-review-run-3"]["tool calls"] == 1
    text = render(collect(tmp_path))
    assert "### run-1\n\n| Metric | Value |" in text and "| tool calls | 9 |" in text
