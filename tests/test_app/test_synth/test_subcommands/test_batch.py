import os
from src.smoke_mirrors.app.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_synth_batch_command():
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
    print(result.output)
    assert result.exit_code == 0
