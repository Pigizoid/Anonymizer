import pytest
import pathlib
import os

from tests.schema import test_user as schema_model
from src.smoke_mirrors.app.synth.funcs import synth_func

seed = "random"
methods = ["mixed", "mimesis", "faker"]
amounts = [1, 2]
start_index = 0
ingest = "tests\\data.json"
cout = False

#CWD is Smoke-and-Mirrors
output = "tests\\outputs\\test_synth_out"

field_tests = [
    {"name": "default"},
    {"age": "default"},
    {"email": "default"},
    {"name": "mask"},
    {"name": "perturb"},
    {"name": "synth"},
]

test_list = []
for amount in amounts:
    for method in methods:
        test_list.append(
            (schema_model, method, amount, output, cout, start_index, seed)
        )


@pytest.mark.parametrize(
    "schema_model, method, amount, output, cout, start_index, seed", test_list
)
def test_synth_func(schema_model, method, amount, output, cout, start_index, seed):
    print("CWD:", os.getcwd())
    print("Looking for:", os.path.abspath(f"{output}.json"))
    with open(f"{output}.json", "w") as f:  # clear output
        f.write("")
    synth_func(schema_model, method, amount, output, cout=cout, start_index=start_index, seed=seed)
    with open(f"{output}.json", "r") as f:
        assert len(f.readlines()) != 0
