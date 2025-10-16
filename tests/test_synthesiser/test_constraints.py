import pytest

from src.smoke_mirrors.pre_made_data import all_constr_attribs
from src.smoke_mirrors.tools.model_funcs import get_json_model_fields
from smoke_mirrors.synthesiser.constraints import (
    make_one_string,
    check_generation_constraints,
    get_applied_constraints,
)

from tests.test_synthesiser.models import (
    test_Address,
    test_Address_2,
    Constraints,
    ConstraintsNested,
    Constraints6,
)

import re


def test_make_one_string_basic():
    pattern = r"[a-z]{5}"
    result = make_one_string(pattern)
    assert isinstance(result, str)
    assert re.fullmatch(pattern, result)


test_check_generation_constraints_pass_schemas = [test_Address, test_Address_2]


@pytest.mark.parametrize(
    "schema_model",
    [(schema_item) for schema_item in test_check_generation_constraints_pass_schemas],
)
def test_check_generation_constraints_expect_pass(schema_model):
    for name, field in get_json_model_fields(schema_model).items():
        return_value = check_generation_constraints(name, field)
        assert isinstance(return_value, dict)
        assert set(return_value.keys()) == set(
            [
                "default",
                "annotation",
                "min_length",
                "max_length",
                "pattern",
                "gt",
                "lt",
                "ge",
                "le",
                "multiple_of",
                "origin",
                "args",
                "required",
            ]
        )


def test_check_generation_constraints_expect_alternate():
    schema_model = Constraints6
    return_value = {}
    for name, field in get_json_model_fields(schema_model).items():
        return_value[name] = check_generation_constraints(name, field)
    assert return_value["constr_default"]["default"] == "AAA"
    assert return_value["constr_annotation"]["annotation"] is str
    assert return_value["constr_min_length"]["min_length"] == 5
    assert return_value["constr_max_length"]["max_length"] == 5
    assert return_value["constr_pattern"]["pattern"] == r"^\d{5}(-\d{4})?$"
    assert return_value["constr_gt"]["gt"] == 5
    assert return_value["constr_lt"]["lt"] == 5
    assert return_value["constr_ge"]["ge"] == 5
    assert return_value["constr_le"]["le"] == 5
    assert return_value["constr_multiple_of"]["multiple_of"] == 5
    assert return_value["constr_origin"]["origin"]
    assert return_value["constr_args"]["args"]
    assert return_value["constr_required"]["required"]


val_types = [bool, int, float, complex, bytes, str]


model_tests = [Constraints, ConstraintsNested]


@pytest.mark.parametrize(
    "schema_model", [(schema_model) for schema_model in model_tests]
)
def test_get_applied_constraints(schema_model):
    return_data = get_applied_constraints(schema_model)
    assert isinstance(return_data, dict)
    assert all([isinstance(x, dict) for x in return_data.values()])
    assert all(
        [z in x.keys() for x in return_data.values() for z in all_constr_attribs]
    )
