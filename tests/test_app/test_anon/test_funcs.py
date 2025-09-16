import pytest
import pathlib
import os

from tests.schema import test_user as schema_model
from sm.app.anon.funcs import anon_func

seed = "random"
methods = ["mixed", "mimesis", "faker"]
amounts = [1, 2]
start_index = 0
ingest = "data.json"
cout = False
manuals = [True, False]
defaults = ["mask", "perturb", "synth"]
field_defaults = ["default", "mask", "perturb", "synth"]
fields_list = ["name", "age", "email"]

test_dir = pathlib.Path(__file__).resolve().parent.parent.parent
output = "outputs\\test_anon_out"

field_tests = [
    {"name": "default"},
    {"age": "default"},
    {"email": "default"},
    {"name": "mask"},
    {"name": "perturb"},
    {"name": "synth"},
]

test_list = []
for fields in field_tests:
    for default in defaults:
        for manual in manuals:
            for amount in amounts:
                for method in methods:
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
                        )
                    )


@pytest.mark.parametrize(
    "schema_model, seed, method, amount, start_index, ingest, cout, manual, default, fields, output",
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
):
    print("test_dir", test_dir)
    os.chdir(
        test_dir
    )  # pytest alters cwd during runtime based on relative imports for some reason
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
    )
    with open(f"{output}.json", "r") as f:
        assert len(f.readlines()) != 0
