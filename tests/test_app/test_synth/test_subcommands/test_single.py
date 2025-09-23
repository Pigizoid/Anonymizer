import os
import pathlib
from src.sm.app.main import app

from typer.testing import CliRunner

runner = CliRunner()

def test_synth_single_command():
    print(os.getcwd())
    result = runner.invoke(
        app,
        ["--config", "tests\\config.yaml", "synth", "single"],
    )
    assert result.exit_code == 0
