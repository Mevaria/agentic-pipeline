"""List every module, class and function in this repository that has no docstring.

    python -m scripts.check_docstrings

Prints one line per missing docstring as "path:line kind name", then a total,
and exits with status 1 when anything is missing so it can gate a commit or a
CI run. tests/test_docstrings.py calls the same check, so pytest enforces it too.
"""

import ast
import sys
from pathlib import Path

# The repository root: this file lives in scripts/, so one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Folders that hold third-party or generated code, which is not ours to document. runs/ holds run output,
# including test files the author wrote for a target repository, which follow that repository's rules.
SKIPPED_FOLDERS = {".venv", "venv", ".git", "__pycache__", ".pytest_cache", "node_modules", "runs"}
# AST node types that can carry a docstring, mapped to the label printed for them.
DOCUMENTED_NODES = {
    ast.Module: "module",
    ast.ClassDef: "class",
    ast.FunctionDef: "function",
    ast.AsyncFunctionDef: "function",
}


def is_skipped(part):
    """Return True for a path part that names a skipped folder, including any virtual environment such as .venv-chroma."""
    return part in SKIPPED_FOLDERS or part.startswith(".venv")


def python_files(root):
    """Yield every .py file under root, skipping virtual environments and caches."""
    for path in sorted(root.rglob("*.py")):
        # Any skipped folder anywhere in the path excludes the file, so nested .venv folders are skipped too.
        if not any(is_skipped(part) for part in path.parts):
            yield path


def missing_docstrings_in(path):
    """Return (line, kind, name) for every undocumented module, class or function in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    missing = []
    # ast.walk visits nested definitions too, so closures inside build_implementer are checked.
    for node in ast.walk(tree):
        kind = DOCUMENTED_NODES.get(type(node))
        if kind is None:
            continue
        # ast.get_docstring returns None when the first statement is not a string literal.
        if ast.get_docstring(node) is None:
            # Modules have no name or line of their own, so they are reported as line 1 of the file.
            line = getattr(node, "lineno", 1)
            name = getattr(node, "name", "<module>")
            missing.append((line, kind, name))
    return missing


def find_missing_docstrings(root=PROJECT_ROOT):
    """Return (path, line, kind, name) for every missing docstring under root, in file and line order."""
    missing = []
    for path in python_files(root):
        # Paths are reported relative to the root so the output is the same on every machine.
        relative = path.relative_to(root).as_posix()
        missing.extend((relative, line, kind, name) for line, kind, name in missing_docstrings_in(path))
    return missing


def main():
    """Print every missing docstring and exit with 1 if there are any."""
    missing = find_missing_docstrings()
    for path, line, kind, name in missing:
        print(f"{path}:{line} {kind} {name}")
    print(f"{len(missing)} missing docstring(s)")
    # A non-zero exit lets a pre-commit hook or CI step fail on the result.
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
