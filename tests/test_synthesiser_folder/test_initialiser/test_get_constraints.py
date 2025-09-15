import pytest

from pydantic import BaseModel, Field, constr
from typing import Dict, Annotated, List, Union, Literal, Tuple, Set
from decimal import Decimal

from src.sm.pre_made_data import all_constr_attribs
from src.sm.synthesiser.synthesiser import Synthesiser

synth = Synthesiser()


class ConstraintsNested(BaseModel):
    test_none: None
    test_basic: str
    test_pattern: str = Field(pattern=r"^a$")
    test_list: List[str]
    test_dict: Dict[Annotated[str, constr(pattern=r"^\d{50}$")], str]
    test_tuple: Tuple[str, str, str]
    test_set: Set[str]
    test_union: Union[str, None]
    test_literal: Literal["1", "2", "3", "4"]
    test_recursive: List[Dict[Annotated[str, constr(pattern=r"^\d{50}$")], List[Tuple[str, str]]]]
    test_list_length: List[Annotated[str, constr(pattern=r"^\d{50}$")]] = Field(min_length=20)
    test_list_length: Dict[Annotated[str, constr(pattern=r"^\d{50}$")], Annotated[str, constr(pattern=r"^\d{50}$")]] = Field(min_length=20)
    test_list_length: Set[Annotated[str, constr(pattern=r"^\d{50}$")]] = Field(min_length=20)

class Constraints(BaseModel):
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
    nested_1: ConstraintsNested
    nested_2: ConstraintsNested


model_tests= [Constraints,ConstraintsNested]
@pytest.mark.parametrize("schema_model",[(schema_model) for schema_model in model_tests])
def test_get_applied_constraints(schema_model):
    return_data = synth.get_applied_constraints(schema_model)
    assert isinstance(return_data,dict)
    assert all([isinstance(x,dict) for x in return_data.values()])
    assert all([z in x.keys() for x in return_data.values() for z in all_constr_attribs])




def test_recursive_get_applied_constraints():
    return_data = synth.recursive_get_applied_constraints(Constraints)
    assert isinstance(return_data,dict)
    assert list(return_data.keys()) == ["Constraints","ConstraintsNested"]
    assert all([isinstance(x,dict) for x in return_data.values()])
    assert all([isinstance(x,dict) for y in return_data.values() for x in y.values()])
    assert all([z in x.keys() for y in return_data.values() for x in y.values() for z in all_constr_attribs])

def test_recursive_get_applied_constraints_alternate():
    return_data = synth.recursive_get_applied_constraints(ConstraintsNested)
    assert isinstance(return_data,dict)
    assert list(return_data.keys()) == ["ConstraintsNested"]
    assert all([isinstance(x,dict) for x in return_data.values()])
    assert all([z in x.keys() for y in return_data.values() for x in y.values() for z in all_constr_attribs])
    