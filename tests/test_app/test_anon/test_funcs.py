import pytest
from pathlib import Path
import os
import json

from src.smoke_mirrors.app.anon.funcs import anon_func
from src.smoke_mirrors.app.helper_funcs import load_ingest_data
from src.smoke_mirrors.library.jsonschemaclass import JsonSchemaClass
from pathlib import Path
test_schema = Path("tests") / "schema.json"
with open(test_schema,"r") as f:
    schema_model=JsonSchemaClass(json.load(f))

seed = "random"
methods = ["mixed", "mimesis", "faker"]
amounts = [1, 2]
start_index = 0
ingest_file = Path("tests") / "data.json"
ingest = load_ingest_data(ingest_file)

stdcout = False
manuals = [True, False]
defaults = ["mask", "perturb", "synth"]
field_defaults = ["default", "mask", "perturb", "synth"]
fields_list = ["name", "age", "email"]

output = Path("tests") / "outputs" / "test_anon_out"

field_tests = [
    {"name": "default"},
    {"age": "default"},
    {"email": "default"},
    {"name": "mask"},
    {"name": "perturb"},
    {"name": "synth"},
]
key_anons = [True,False]
performances = [True,False]

test_list = []

for default in defaults:
    for amount in amounts:
        for method in methods:
            for key_anon in key_anons:
                 for manual in manuals:
                    if manual == True:
                        for fields in field_tests:
                            test_list.append(
                                (
                                    schema_model,
                                    seed,
                                    method,
                                    amount,
                                    start_index,
                                    ingest,
                                    stdcout,
                                    manual,
                                    default,
                                    fields,
                                    output,
                                    key_anon,
                                    True
                                )
                            )
                    else:
                        test_list.append(
                            (
                                schema_model,
                                seed,
                                method,
                                amount,
                                start_index,
                                ingest,
                                stdcout,
                                manual,
                                default,
                                {},
                                output,
                                key_anon,
                                True
                            )
                        )



@pytest.mark.parametrize(
    "schema_model, seed, method, amount, start_index, ingest, stdcout, manual, default, fields, output, key_anon, performance",
    test_list,
)
def test_anon_func(
    schema_model,
    seed,
    method,
    amount,
    start_index,
    ingest,
    stdcout,
    manual,
    default,
    fields,
    output:Path,
    key_anon,
    performance
):
    print("CWD:", os.getcwd())
    stem = output.stem
    file_output = output.with_stem(stem+"_(temp)")
    file_output = file_output.with_suffix(".json")
    print("Looking for:", os.path.abspath(file_output))
    with open(file_output, "w") as f:  # clear output
        f.write("")
    anon_func(
        schema_model,
        seed,
        method,
        amount,
        start_index,
        ingest,
        stdcout,
        manual,
        default,
        fields,
        output,
        key_anon,
        performance
    )
    with open(file_output, "r") as f:
        assert len(f.readlines()) != 0
