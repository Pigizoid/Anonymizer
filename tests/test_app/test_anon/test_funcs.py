import pytest
import pathlib
import os
import json

from src.smoke_mirrors.app.anon.funcs import anon_func
from src.smoke_mirrors.app.helper_funcs import load_ingest_data
from src.smoke_mirrors.library.jsonschemaclass import JsonSchemaClass
with open("tests\\schema.json","r") as f:
    schema_model=JsonSchemaClass(json.load(f))

seed = "random"
methods = ["mixed", "mimesis", "faker"]
amounts = [1, 2]
start_index = 0
ingest_file = "tests\\data.json"
ingest = load_ingest_data(ingest_file)

stdcout = False
manuals = [True, False]
defaults = ["mask", "perturb", "synth"]
field_defaults = ["default", "mask", "perturb", "synth"]
fields_list = ["name", "age", "email"]

output = "tests\\outputs\\test_synth_out"

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
    output,
    key_anon,
    performance
):
    print("CWD:", os.getcwd())
    print("Looking for:", os.path.abspath(f"{output}_(temp).json"))
    with open(f"{output}_(temp).json", "w") as f:  # clear output
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
    with open(f"{output}_(temp).json", "r") as f:
        assert len(f.readlines()) != 0
