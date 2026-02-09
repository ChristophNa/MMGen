import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def test_console_script_is_declared():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {})
    assert scripts.get("mmgen") == "mmgen.cli:main"


def test_console_script_help_exits_zero_when_installed():
    mmgen_cmd = shutil.which("mmgen")
    if mmgen_cmd is None:
        return

    result = subprocess.run([mmgen_cmd, "--help"], capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert "Run MMGen sample generation workflows." in result.stdout


def test_module_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "mmgen.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run MMGen sample generation workflows." in result.stdout


def test_python_main_py_is_unsupported():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
