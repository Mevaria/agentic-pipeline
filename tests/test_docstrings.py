"""Enforces the project rule that every module, class and function has a docstring."""

from scripts.check_docstrings import find_missing_docstrings


def test_every_definition_has_a_docstring():
    """The docstring check reports nothing, so the rule holds for every Python file in the repository."""
    missing = find_missing_docstrings()
    # Listing the offenders in the assertion message says exactly what to fix when this fails.
    assert missing == [], "Missing docstrings:\n" + "\n".join(
        f"{path}:{line} {kind} {name}" for path, line, kind, name in missing
    )
