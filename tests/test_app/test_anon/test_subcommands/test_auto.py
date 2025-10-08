import os
import pathlib
from src.smoke_mirrors.app.main import app

from typer.testing import CliRunner

runner = CliRunner()



def test_anon_auto_command():
    print(os.getcwd())
    result = runner.invoke(
        app,
        ["anon", "auto", "--ingest", "tests\\data.json", "--default", "synth"],
    )
    assert result.exit_code == 0
