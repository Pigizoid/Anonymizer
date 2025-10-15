import pytest
from pathlib import Path
import os
import json
from src.smoke_mirrors.app.synth.funcs import synth_func
from src.smoke_mirrors.library.jsonschemaclass import JsonSchemaClass
from pathlib import Path
test_schema = Path("tests") / "schema.json"
with open(test_schema,"r") as f:
    schema_model=JsonSchemaClass(json.load(f))

seed = "random"
methods = ["mixed", "mimesis", "faker"]
amounts = [1, 2]
start_index = 0
stdcout = False

#CWD is Smoke-and-Mirrors
output = Path("tests") / "outputs" / "test_synth_out"

test_list = []
for amount in amounts:
    for method in methods:
        test_list.append(
            (schema_model, method, amount, output, stdcout, start_index, seed)
        )


@pytest.mark.parametrize(
    "schema_model, method, amount, output, stdcout, start_index, seed", test_list
)
def test_synth_func(schema_model, method, amount, output, stdcout, start_index, seed):
    print("CWD:", os.getcwd())
    stem = output.stem
    output.with_stem(stem+"_(temp)")
    output.with_suffix(".json")
    print("Looking for:", os.path.abspath(output))
    with open(output, "w") as f:  # clear output
        f.write("")
    synth_func(schema_model, method, amount, output, stdcout=stdcout, start_index=start_index, seed=seed)
    with open(output, "r") as f:
        assert len(f.readlines()) != 0
