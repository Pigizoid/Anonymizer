import pytest
import os
from src.smoke_mirrors.app.main import app

from typer.testing import CliRunner

runner = CliRunner()


@pytest.mark.parametrize(
    "c,subc",
    [("anon", "auto"), ("anon", "manual"), ("synth", "single"), ("synth", "batch")],
)
def test_main(c, subc):
    print(os.getcwd())
    result = runner.invoke(
        app,
        ["--config", "tests\\config.yaml", f"{c}", f"{subc}"],
    )
    print(result.output)
    assert result.exit_code == 0
