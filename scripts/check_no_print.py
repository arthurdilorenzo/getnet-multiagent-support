"""Reject direct Python console calls in application sources using their AST.

This static guard handles direct calls and imported builtins aliases. It does
not attempt whole-program analysis of dynamic aliases or reflective calls.
"""

import argparse
import ast
from pathlib import Path
import sys
import tokenize


def check_file(path: Path) -> list[str]:
    try:
        with tokenize.open(path) as source:
            tree = ast.parse(source.read(), filename=str(path))
    except (SyntaxError, UnicodeError, OSError) as exc:
        return [f"{path}:{getattr(exc, 'lineno', 1)}: syntax/read error: {exc}"]

    module_names = {"builtins"}
    function_names = {"print"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_names.update(alias.asname or alias.name for alias in node.names if alias.name == "builtins")
        elif isinstance(node, ast.ImportFrom) and node.module == "builtins":
            function_names.update(alias.asname or alias.name for alias in node.names if alias.name == "print")

    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        direct = isinstance(fn, ast.Name) and fn.id in function_names
        builtin = (
            isinstance(fn, ast.Attribute)
            and fn.attr == "print"
            and isinstance(fn.value, ast.Name)
            and fn.value.id in module_names
        )
        if direct or builtin:
            errors.append(f"{path}:{node.lineno}:{node.col_offset + 1}: forbidden console call")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    errors = []
    for path in args.paths:
        if not path.exists():
            errors.append(f"{path}:1: source path does not exist")
            continue
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            if file.suffix == ".py":
                errors.extend(check_file(file))
    for error in errors:
        sys.stderr.write(error + "\n")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
