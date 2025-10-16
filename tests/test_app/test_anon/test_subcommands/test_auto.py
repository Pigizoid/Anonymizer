import os
from src.smoke_mirrors.app.main import app

from typer.testing import CliRunner

runner = CliRunner()


def test_anon_auto_command():
    print(os.getcwd())
    result = runner.invoke(
        app,
        ["anon", "auto", "--ingest", "tests\\data.json", "--default", "synth"],
    )
    print(result.output)
    assert result.exit_code == 0
