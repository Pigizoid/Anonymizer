import pytest
import os
from src.sm.app.main import app

from typer.testing import CliRunner
runner = CliRunner()

import pathlib
test_dir = pathlib.Path(__file__).resolve().parent.parent
os.chdir(test_dir)

@pytest.mark.parametrize("c,subc",[("anon","auto"),("anon","manual"),("synth","single"),("synth","batch")])
def test_main(c,subc):
    os.chdir(test_dir)
    print(os.getcwd())
    result = runner.invoke(
        app,
        [
            "--config",
            "config.yaml",
            f"{c}",
            f"{subc}"
        ],
    )
    assert result.exit_code == 0