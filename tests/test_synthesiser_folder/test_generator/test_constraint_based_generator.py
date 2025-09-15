import pytest
import re
from decimal import Decimal
from pydantic import BaseModel, Field, constr
from typing import List, Dict
from src.sm.synthesiser.synthesiser import Synthesiser
from src.sm.pre_made_data import default_constr_dict
from src.sm.synthesiser.generator.constraint_based_generator import (
    make_one_decimal,
    make_one_string,
)
from src.sm.tools.model_funcs import get_model_fields
from faker import Faker

fake = Faker()

synth = Synthesiser()


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


class test_Address(BaseModel):
    street: str
    city: str


class test_Address_2(BaseModel):
    street: str
    city: str
    social_security_number: str
    continent: str


test_check_generation_constraints_pass_schemas = [test_Address, test_Address_2]


@pytest.mark.parametrize(
    "schema_model",
    [(schema_item) for schema_item in test_check_generation_constraints_pass_schemas],
)
def test_check_generation_constraints_expect_pass(schema_model):
    for name, field in get_model_fields(schema_model).items():
        return_value = synth.check_generation_constraints(name, field)
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


class Constraints6(BaseModel):
    constr_strip_whitespace: str = constr(strip_whitespace=True)
    constr_to_upper: str = constr(to_upper=True)
    constr_to_lower: str = constr(to_lower=True)
    constr_strict: str = Field(strict=True)
    constr_default: str = Field(default="AAA")
    constr_annotation: str
    constr_min_length: str = Field(min_length=5)
    constr_max_length: str = Field(max_length=5)
    constr_pattern: str = Field(pattern=r"^\d{5}(-\d{4})?$")
    constr_gt: int = Field(gt=5)
    constr_lt: int = Field(lt=5)
    constr_ge: int = Field(ge=5)
    constr_le: int = Field(le=5)
    constr_multiple_of: int = Field(multiple_of=5)
    constr_allow_inf_nan: int = Field(allow_inf_nan=True)
    constr_max_digits: Decimal = Field(max_digits=5)
    constr_decimal_places: Decimal = Field(decimal_places=5)
    constr_origin: List[str]
    constr_args: Dict[str, str]
    constr_required: str


def test_check_generation_constraints_expect_alternate():
    schema_model = Constraints6
    return_value = {}
    for name, field in get_model_fields(schema_model).items():
        return_value[name] = synth.check_generation_constraints(name, field)
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
