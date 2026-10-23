"""Exercise the exact CLI used by Actions against temporary Python sources."""

from pathlib import Path
import subprocess
import sys

import pytest


CHECKER = Path(__file__).resolve().parents[1] / "scripts" / "check_no_print.py"


def run_check(directory: Path):
    return subprocess.run(
        [sys.executable, str(CHECKER), str(directory)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_ci_checker_ignores_comments_docstrings_strings_and_non_python(tmp_path):
    (tmp_path / "innocent.py").write_text(
        '# print("comment")\n'
        '"""A docstring documenting print()."""\n'
        'message = "print( is only text"\n'
        'def fingerprint(value):\n    return value\n',
        encoding="utf-8",
    )
    (tmp_path / "sample.js").write_text('print("not Python");', encoding="utf-8")

    result = run_check(tmp_path)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "source",
    [
        'print("forbidden")\n',
        'print\n(42)\nprint(\n    "multiline"\n)\n',
        'import builtins\nbuiltins.print(\n    "forbidden"\n)\n',
        'import builtins as console\nconsole.print("forbidden")\n',
        'from builtins import print as emit\nemit("forbidden")\n',
    ],
)
def test_ci_checker_rejects_executable_console_calls(tmp_path, source):
    (tmp_path / "nested").mkdir()
    target = tmp_path / "nested" / "module.py"
    target.write_text(source, encoding="utf-8")

    result = run_check(tmp_path)

    assert result.returncode == 1
    assert "module.py:" in result.stderr
    assert "console call" in result.stderr


def test_ci_checker_fails_closed_on_invalid_python(tmp_path):
    (tmp_path / "invalid.py").write_text("def broken(:\n", encoding="utf-8")

    result = run_check(tmp_path)

    assert result.returncode == 1
    assert "invalid.py:" in result.stderr
    assert "syntax" in result.stderr.lower()
