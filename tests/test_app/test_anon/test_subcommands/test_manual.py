import os
import pathlib
from src.sm.app.main import app

from typer.testing import CliRunner

runner = CliRunner()


def test_anon_manual_command():
    print(os.getcwd())
    result = runner.invoke(
        app,
        ["--config", "tests\\config.yaml", "anon", "manual"],
    )
    assert result.exit_code == 0
