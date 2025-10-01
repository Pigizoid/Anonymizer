import pytest

from decimal import Decimal

from src.smoke_mirrors.pre_made_data import all_constr_attribs, default_constr_dict
from src.smoke_mirrors.tools.model_funcs import get_json_model_fields
from src.smoke_mirrors.synthesiser.helper_funcs.constraints import make_one_string, check_generation_constraints, make_new_contraints, get_applied_constraints

from tests.test_synthesiser.test_helper_funcs.models import test_Address, test_Address_2, Constraints, ConstraintsNested, Constraints6

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


@pytest.mark.parametrize(
    "applied_constraints,val_type",
    [({"annotation": val_type}, val_type) for val_type in val_types],
)
def test_make_new_contraints(applied_constraints, val_type):
    return_data = make_new_contraints(applied_constraints)
    assert all([x == y for x, y in zip(return_data.keys(), default_constr_dict.keys())])
    assert return_data["annotation"] == val_type


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


