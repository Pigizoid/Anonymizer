import pytest
import os
from src.sm.app.main import app

from typer.testing import CliRunner
runner = CliRunner()
test_dir = os.getcwd() + "\\tests"


def test_anon_auto_command():
    os.chdir(test_dir)
    print(os.getcwd())
    result = runner.invoke(
        app,
        [
            "--config",
            "config.yaml",
            "anon",
            "auto"
        ],
    )
    assert result.exit_code == 0