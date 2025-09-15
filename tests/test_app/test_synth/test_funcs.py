import pytest
import pathlib
import os

from tests.schema import test_user as schema_model
from src.sm.app.synth.funcs import synth_func

seed = "random"
methods = ["mixed", "mimesis", "faker"]
amounts = [1, 2]
start_index = 0
ingest = "data.json"
cout = False

test_dir = pathlib.Path(__file__).resolve().parent.parent.parent
output = "outputs\\test_synth_out"

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
            (schema_model, seed, method, amount, output, start_index, cout)
        )


@pytest.mark.parametrize(
    "schema_model, seed, method, amount, output, start_index, cout", test_list
)
def test_synth_func(schema_model, seed, method, amount, output, start_index, cout):
    print("test_dir", test_dir)
    os.chdir(
        test_dir
    )  # pytest alters cwd during runtime based on relative imports for some reason
    print("CWD:", os.getcwd())
    print("Looking for:", os.path.abspath(f"{output}.json"))
    with open(f"{output}.json", "w") as f:  # clear output
        f.write("")
    synth_func(schema_model, seed, method, amount, output, start_index, cout)
    with open(f"{output}.json", "r") as f:
        assert len(f.readlines()) != 0
