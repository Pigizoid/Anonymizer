from src.smoke_mirrors.synthesiser.synthesiser import Synthesiser

import pytest
import re
from src.smoke_mirrors.pre_made_data import default_constr_dict
from faker import Faker

from pydantic import BaseModel

from src.smoke_mirrors.tools.model_funcs import get_model_fields
from src.smoke_mirrors.synthesiser.helper_funcs.matching_fields import recursive_match_fields
from src.smoke_mirrors.synthesiser.helper_funcs.constraints import recursive_get_applied_constraints
from src.smoke_mirrors.synthesiser.synthesiser import print_path

from tests.test_synthesiser_folder.test_helper_funcs.models import generate_test1

class User(BaseModel):
    id: int
    name: str

fake = Faker()
synth = Synthesiser()


def test_synthesise_empty():
    result = synth.synthesise(User, amount=0)
    assert result == []


def test_synthesise_amount():
    result = synth.synthesise(User, amount=3)
    assert len(result) == 3
    assert all([isinstance(x, User) for x in result])


"""
def test_synthesise_seed():
    result1 = synth.synthesise(User, amount=2, seed=0)
    result2 = synth.synthesise(User, amount=2, seed=0)
    assert result1 == result2
"""


def test_synthesise_recursive_dict():
    data = synth.synthesise_recursive(User, amount=1)
    assert isinstance(data, dict)
    assert "id" in data
    assert "name" in data


def test_progress_prints(capsys):
    synthp = Synthesiser(cout=True)
    synthp.synthesise(User, amount=5)
    captured = capsys.readouterr()
    assert "Completed:" in captured.out




# ----- constraint generator tests



def test_generate_from_constraints():
    generate_path = "test[100].List(0)[10].Dict(Right)[10].Annotated"
    constraints = default_constr_dict.copy()
    constraints["annotation"] = str
    constraints["pattern"] = r"^a$"
    return_value = synth.generate_from_constraints("test", constraints, generate_path)
    assert return_value == "a"
    data_pool = synth.outputpooling
    assert (
        len(data_pool[generate_path]) == 10000 - 1
    )  # -1 because the pop method was run
    assert all(
        [re.search(r"a", text).group() == text for text in data_pool[generate_path]]
    )


val_types = [bool, int, float, complex, bytes, str]


@pytest.mark.parametrize("val_type", [(val_type) for val_type in val_types])
def test_generate_from_constraints_alternate(val_type):
    generate_path = "test[100].List(0)[10].Dict(Right)[10]"
    constraints = default_constr_dict.copy()
    constraints["annotation"] = val_type
    constraints["pattern"] = None
    return_value = synth.generate_from_constraints("test", constraints, generate_path)
    assert isinstance(return_value, val_type)


val_types = [bool, int, float, complex, bytes, str]


@pytest.mark.parametrize("val_type", [(val_type) for val_type in val_types])
def test_apply_constraints(val_type):
    generate_path = "test(name)[100].List(0)[10].Dict(Right)[10]"
    constraints = default_constr_dict.copy()
    constraints["annotation"] = val_type
    constraints["pattern"] = None
    value = fake.name()
    match_name = "name"
    return_value = synth.apply_constraints(
        value, constraints, match_name, generate_path, 10000, 100
    )
    assert isinstance(return_value, val_type)

# ----- constraint generator tests



# ----- synth generator tests

def test_generate_synth_data():
    method = "mixed"

    schema_model = generate_test1
    schema_name = schema_model.__name__
    field_match_pairs = recursive_match_fields(schema_model,method)
    applied_constraints = recursive_get_applied_constraints(
        schema_model
    )
    synthesised_data = {}
    for name in get_model_fields(schema_model).keys():
        generate_path = "" + f"{schema_model.__name__}({name})[1]"

        synthesised_data[name] = synth.generate_synth_data(
            name,
            field_match_pairs[schema_name][name],
            applied_constraints[schema_name][name],
            generate_path,
        )
    assert schema_model(**synthesised_data)


# ----- synth generator tests


# ----- print tests


def test_print_path_simple(capsys):
    path = "[0]"
    elapsed_time = 1.2345
    print_path(path, elapsed_time)

    captured = capsys.readouterr()
    output = captured.out.strip()

    assert "Time taken: 1.23 seconds" in output
    assert path in output


def test_print_path_deeper_path(capsys):
    path = "[0][1][2]"
    elapsed_time = 12.5
    print_path(path, elapsed_time)

    captured = capsys.readouterr()
    output = captured.out

    assert "Time taken: 12.50 seconds" in output
    assert path in output
    assert "        " in output


@pytest.mark.parametrize(
    "path,elapsed,expected",
    [
        ("[3]", 0.0, "Time taken: 0.00 seconds"),
        ("[1][2]", 2.718, "Time taken: 2.72 seconds"),
        ("[9][9][9][9]", 100.1234, "Time taken: 100.12 seconds"),
    ],
)
def test_print_path_parametrized(path, elapsed, expected, capsys):
    print_path(path, elapsed)
    captured = capsys.readouterr()
    output = captured.out

    assert expected in output
    assert path in output
