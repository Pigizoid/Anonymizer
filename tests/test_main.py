# test_main.py


from src.sm.main import *


from typer.testing import CliRunner
# from unittest.mock import patch

import os
runner = CliRunner()


test_dir = os.getcwd()+"\\tests"

def test_synthesise():
    os.chdir(test_dir)
    print(os.getcwd())
    result = runner.invoke(
        app,
        [
            "synthesise",
            "--method",
            "mixed",
            "--amount",
            "10",
            "--batch",
            "5",
            "--output",
            "test_out",
            "--cout",
        ],
    )
    assert result.exit_code == 0
    assert "Args:" in result.stdout


def test_anonymise():
    os.chdir(test_dir)
    print(os.getcwd())
    result = runner.invoke(
        app,
        [
            "synthesise",
            "--method",
            "mixed",
            "--amount",
            "10",
            "--batch",
            "5",
            "--output",
            "test_out",
            "--cout",
        ],
    )
    assert result.exit_code == 0
    assert "Args:" in result.stdout

ingest: str = (None,)
method: str = (None,)
amount: int = (None,)
start: int = (None,)
output: str = (None,)
manual: Annotated[typing.Optional[bool], typer.Option("--manual/--no-manual")] = (None,)
cout: Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = (None,)
config: str = None
"""


synth_func()

anon_func()

load_config()

load_folder()

close_folder()

send_to_API()

load_flags()

load_ingest_data()
"""
