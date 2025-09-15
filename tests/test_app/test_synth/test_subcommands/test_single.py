import pytest
import os
from src.sm.app.main import app

from typer.testing import CliRunner
runner = CliRunner()
import pathlib
test_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
os.chdir(test_dir)


def test_anon_auto_command():
    os.chdir(test_dir)
    print(os.getcwd())
    result = runner.invoke(
        app,
        [
            "--config",
            "config.yaml",
            "synth",
            "single"
        ],
    )
    assert result.exit_code == 0