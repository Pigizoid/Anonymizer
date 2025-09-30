import pytest
import pathlib
import os
import json

from tests.schema import test_user as schema_model
from src.sm.app.anon.funcs import anon_func

seed = "random"
methods = ["mixed", "mimesis", "faker"]
amounts = [1, 2]
start_index = 0
ingest_file = "tests\\data.json"
with open(ingest_file, 'r') as f:
    ingest = json.load(f)
cout = False
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

test_list = []
for fields in field_tests:
    for default in defaults:
        for manual in manuals:
            for amount in amounts:
                for method in methods:
                    for key_anon in key_anons:
                        test_list.append(
                            (
                                schema_model,
                                seed,
                                method,
                                amount,
                                start_index,
                                ingest,
                                cout,
                                manual,
                                default,
                                fields,
                                output,
                                key_anon
                            )
                        )


@pytest.mark.parametrize(
    "schema_model, seed, method, amount, start_index, ingest, cout, manual, default, fields, output, key_anon",
    test_list,
)
def test_anon_func(
    schema_model,
    seed,
    method,
    amount,
    start_index,
    ingest,
    cout,
    manual,
    default,
    fields,
    output,
    key_anon
):
    print("CWD:", os.getcwd())
    print("Looking for:", os.path.abspath(f"{output}.json"))
    with open(f"{output}.json", "w") as f:  # clear output
        f.write("")
    anon_func(
        schema_model,
        seed,
        method,
        amount,
        start_index,
        ingest,
        cout,
        manual,
        default,
        fields,
        output,
        key_anon
    )
    with open(f"{output}.json", "r") as f:
        assert len(f.readlines()) != 0
