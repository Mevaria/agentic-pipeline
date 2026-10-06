"""Reading task folders.

A task folder holds task.json (type, short_description, spec) and a tests
folder whose files are added to the target repository's tests folder. Hand-
written tasks live under tasks/ in this repository; the spec and test author
writes generated ones under TASKS_PATH, one folder per run. Both have the same
layout, so the implementer runs either.
"""

import json
from pathlib import Path


def load_task(task_folder):
    """Read a task folder into a dict with type, short_description, spec and a tests mapping.

    The tests mapping goes from the path the file will have inside the target
    repository, such as "tests/test_delete_book.py", to its full content.
    """
    folder = Path(task_folder)
    task = json.loads((folder / "task.json").read_text(encoding="utf-8"))
    tests_folder = folder / "tests"
    # Keys are repository-relative so the implementer can write them straight into the target's tests folder.
    # sorted() keeps the order stable between runs; a task without tests gets an empty mapping.
    task["tests"] = {
        f"tests/{path.name}": path.read_text(encoding="utf-8")
        for path in sorted(tests_folder.glob("*.py"))
    } if tests_folder.exists() else {}
    return task
