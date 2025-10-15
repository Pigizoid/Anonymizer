import os
import pathlib
from src.smoke_mirrors.app.main import app

from typer.testing import CliRunner

runner = CliRunner()

def test_synth_single_command():
    print(os.getcwd())
    result = runner.invoke(
        app,
        ["--config", "tests\\config.yaml", "synth", "single"],
    )
    print(result.output)
    assert result.exit_code == 0
