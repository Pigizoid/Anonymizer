# test_main.py


from src.sm.main import *


from typer.testing import CliRunner
# from unittest.mock import patch

import os
runner = CliRunner()


test_dir = os.getcwd()+"\\tests"

def test_synth_single():
    os.chdir(test_dir)
    print(os.getcwd())
    result = runner.invoke(
        app,
        [
            "synth",
        ],
    )
    print(result)
    assert result.exit_code == 0
    assert "Args:" in result.stdout

def test_main_config_single():
    os.chdir(test_dir)
    print(os.getcwd())
    result = runner.invoke(
        app,
        [
            "--config",
            "config.yaml",
            "synth",
        ],
    )
    assert result.exit_code == 0
    assert "Args:" in result.stdout
    assert "output='test_output'" in result.stdout

def test_main_config_batch():
    os.chdir(test_dir)
    print(os.getcwd())
    result = runner.invoke(
        app,
        [
            "--config",
            "config.yaml",
            "synth",
            "batch",
        ],
    )
    assert result.exit_code == 0
    assert "Args:" in result.stdout
    assert "amount=8" in result.stdout
    assert "batch=4" in result.stdout

def test_synth_single_method():
    os.chdir(test_dir)
    print(os.getcwd())
    result = runner.invoke(
        app,
        [
            "synth",
            "--method",
            "faker",
        ],
    )
    assert result.exit_code == 0
    assert "Args:" in result.stdout
    assert "method='faker'" in result.stdout

def test_synth_single_output():
    os.chdir(test_dir)
    print(os.getcwd())
    result = runner.invoke(
        app,
        [
            "synth",
            "--output",
            "output_file",
        ],
    )
    assert result.exit_code == 0
    assert "Args:" in result.stdout
    assert "output='output_file'" in result.stdout
    

def test_synth_single_cout():
    os.chdir(test_dir)
    print(os.getcwd())
    result = runner.invoke(
        app,
        [
            "synth",
            "--cout",
        ],
    )
    assert result.exit_code == 0
    assert "Args:" in result.stdout
    assert "cout=True)" in result.stdout
    




def test_synth_batch():
    os.chdir(test_dir)
    print(os.getcwd())
    result = runner.invoke(
        app,
        [
            "synth",
            "batch",
            "--amount",
            "3",
            "--batch",
            "6",
        ],
    )
    assert result.exit_code == 0
    assert "Args:" in result.stdout
    assert "amount=3" in result.stdout
    assert "batch=6" in result.stdout



'''
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
'''






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
