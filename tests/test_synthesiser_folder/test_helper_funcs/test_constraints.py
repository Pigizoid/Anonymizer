import pytest

from pydantic import BaseModel, Field, constr
from typing import Dict, Annotated, List, Union, Literal, Tuple, Set
from decimal import Decimal

from src.smoke_mirrors.pre_made_data import all_constr_attribs, default_constr_dict
from src.smoke_mirrors.tools.model_funcs import get_model_fields
from src.smoke_mirrors.synthesiser.helper_funcs.constraints import make_one_string, make_one_decimal, check_generation_constraints, make_new_contraints, get_applied_constraints, recursive_get_applied_constraints

from tests.test_synthesiser_folder.test_helper_funcs.models import test_Address,test_Address_2,Constraints6, Constraints, ConstraintsNested

import re





def test_make_one_string_basic():
    pattern = r"[a-z]{5}"
    result = make_one_string(pattern)
    assert isinstance(result, str)
    assert re.fullmatch(pattern, result)



def test_make_one_decimal():
    dp = Decimal("0.01")
    scale = Decimal(100)
    min_scaled = Decimal(100)  # represents 1.00
    scaled_mult = Decimal(100)  # multiple of 1.00
    result = make_one_decimal(0, dp, min_scaled, scaled_mult, scale)
    assert result == Decimal("2.00")  # 1 + 1 * 1

test_check_generation_constraints_pass_schemas = [test_Address, test_Address_2]

@pytest.mark.parametrize(
    "schema_model",
    [(schema_item) for schema_item in test_check_generation_constraints_pass_schemas],
)
def test_check_generation_constraints_expect_pass(schema_model):
    for name, field in get_model_fields(schema_model).items():
        return_value = check_generation_constraints(name, field)
        assert isinstance(return_value, dict)
        assert set(return_value.keys()) == set(
            [
                "strip_whitespace",
                "to_upper",
                "to_lower",
                "strict",
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
                "allow_inf_nan",
                "max_digits",
                "decimal_places",
                "origin",
                "args",
                "required",
            ]
        )


def test_check_generation_constraints_expect_alternate():
    schema_model = Constraints6
    return_value = {}
    for name, field in get_model_fields(schema_model).items():
        return_value[name] = check_generation_constraints(name, field)
    assert "strip_whitespace" in return_value["constr_strip_whitespace"]
    assert "to_upper" in return_value["constr_to_upper"]
    assert "to_lower" in return_value["constr_to_lower"]
    assert "strict" in return_value["constr_strict"]
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
    assert return_value["constr_allow_inf_nan"]["allow_inf_nan"]
    assert return_value["constr_max_digits"]["max_digits"] == 5
    assert return_value["constr_decimal_places"]["decimal_places"] == 5
    assert return_value["constr_origin"]["origin"] in [List, list]
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





def test_recursive_get_applied_constraints():
    return_data = recursive_get_applied_constraints(Constraints)
    assert isinstance(return_data, dict)
    assert list(return_data.keys()) == ["Constraints", "ConstraintsNested"]
    assert all([isinstance(x, dict) for x in return_data.values()])
    assert all([isinstance(x, dict) for y in return_data.values() for x in y.values()])
    assert all(
        [
            z in x.keys()
            for y in return_data.values()
            for x in y.values()
            for z in all_constr_attribs
        ]
    )


def test_recursive_get_applied_constraints_alternate():
    return_data = recursive_get_applied_constraints(ConstraintsNested)
    assert isinstance(return_data, dict)
    assert list(return_data.keys()) == ["ConstraintsNested"]
    assert all([isinstance(x, dict) for x in return_data.values()])
    assert all(
        [
            z in x.keys()
            for y in return_data.values()
            for x in y.values()
            for z in all_constr_attribs
        ]
    )
