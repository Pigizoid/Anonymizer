import pytest
import pathlib
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



from pathlib import *
def windows_path_to_pathlib(path_str) -> Path:
    pure_path = PurePath(str(path_str))
    print(pure_path)
    return_path = Path(*pure_path.parts)
    print(return_path)
    return return_path


def test_bruh():
    string_test = "tests\\schema.py"
    windows_test = PureWindowsPath(string_test)
    posix_test = PurePosixPath(string_test)

    a = windows_path_to_pathlib(string_test)
    b = windows_path_to_pathlib(windows_test)
    c = windows_path_to_pathlib(posix_test)

    assert(a==b==c)

    a = windows_path_to_pathlib(a)
    b = windows_path_to_pathlib(b)
    c = windows_path_to_pathlib(c)

    assert(a==b==c)