import os
import pathlib
from src.sm.app.main import app
from typer.testing import CliRunner

runner = CliRunner()

def test_anon_auto_command():
    print(os.getcwd())
    result = runner.invoke(
        app,
        [
            "--config",
            "tests\\config.yaml",
            "synth",
            "batch",
            "--amount",
            "1000",
            "--batch",
            "100",
        ],
    )
    assert result.exit_code == 0
