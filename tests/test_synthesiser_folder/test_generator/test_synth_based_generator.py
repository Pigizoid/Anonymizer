import pytest
from pydantic import BaseModel, Field, constr
from typing import (
    List,
    Dict,
    Tuple,
    Set,
    Union,
    Literal,
    Annotated,
)
from src.sm.pre_made_data import default_constr_dict
from src.sm.tools.model_funcs import get_model_fields
from src.sm.synthesiser.synthesiser import Synthesiser
synth = Synthesiser()


val_types = [bool,int,float,complex,bytes,str]
@pytest.mark.parametrize("applied_constraints,val_type",[({"annotation":val_type},val_type) for val_type in val_types])
def test_make_new_contraints(applied_constraints,val_type):
    return_data = synth.make_new_contraints(applied_constraints)
    assert all([x==y for x,y in zip(return_data.keys(),default_constr_dict.keys())])
    assert return_data["annotation"]==val_type


class generate_test1(BaseModel):
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

class generate_test2(BaseModel):
    test_dict_fail: Dict[Annotated[str, constr(pattern=r"^a$")], Annotated[str, constr(pattern=r"^a$")]] = Field(min_length=20)
    # pattern returns "a" which means there isnt enough keys for a dict of length 20

def test_generate_synth_data():
    synth_g = Synthesiser("mixed")

    schema_model = generate_test1
    schema_name = schema_model.__name__
    synth_g.field_match_pairs = synth_g.recursive_match_fields(schema_model)
    synth_g.applied_constraints = synth_g.recursive_get_applied_constraints(
        schema_model
    )
    synthesised_data = {}
    for name in get_model_fields(schema_model).keys():
        generate_path = "" + f"{schema_model.__name__}({name})[1]"

        synthesised_data[name] = synth_g.generate_synth_data(
            name,
            synth_g.field_match_pairs[schema_name][name],
            synth_g.applied_constraints[schema_name][name],
            generate_path,
        )
    assert schema_model(**synthesised_data)

'''
def test_generate_synth_data_fail():
    synth_g = Synthesiser("mixed")

    schema_model = generate_test2
    schema_name = schema_model.__name__
    synth_g.field_match_pairs = synth_g.recursive_match_fields(schema_model)
    synth_g.applied_constraints = synth_g.recursive_get_applied_constraints(
        schema_model
    )
    synthesised_data = {}
    with pytest.raises(Exception):
        for name in get_model_fields(schema_model).keys():
            generate_path = "" + f"{schema_model.__name__}({name})[1]"

            synthesised_data[name] = synth_g.generate_synth_data(
                name,
                synth_g.field_match_pairs[schema_name][name],
                synth_g.applied_constraints[schema_name][name],
                generate_path,
            )


val_types = [bool,int,float,complex,bytes,str]
list_of_tests = [(f"test_{val_type}",val_type) for val_type in val_types]
@pytest.mark.parametrize("field_name,field_type",list_of_tests)
def generate_single_value(field_name, field_type):
        return_value = synth.generate_single_value(field_name, field_type)
        assert isinstance(return_value,field_type)
'''
