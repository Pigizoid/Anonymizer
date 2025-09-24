from typing import Dict, Any, get_origin, get_args
from decimal import Decimal, ROUND_HALF_UP
import exrex
from pydantic import BaseModel, Field
from sm.library.jsonschema import JsonSchemaClass
import inspect
from sm.pre_made_data import all_constr_attribs, default_constr_dict
from sm.tools.model_funcs import get_json_model_data, get_json_model_fields


def make_one_string(pattern):
    return exrex.getone(pattern)

def make_one_decimal(x:int, decimal_precision:float, min_scaled:float, scaled_mult:float, scale:float) -> Decimal:
    """
    Inputs:\n
        index for data pool selection (the amount of scaled mult to add)
        decimal precision
        minimum value (scaled)
        multiple of (scaled): multiple of e.g. mo = 3 -> 300
        scale

        decimal_places = e.g. 2
        scale = 10 ** decimal_places
        decimal_precision = 10 ** (decimal_places * -1)
    Outputs:\n
        Decimal value
    """
    potential_val = (min_scaled + (x + 1) * scaled_mult) / scale
    if potential_val == potential_val.quantize(
        decimal_precision, rounding=ROUND_HALF_UP
    ):
        return potential_val
    else:
        return None

def check_generation_constraints(name:str, field:dict) -> Dict[str, Any]:
    """
    Inputs:\n
        field name
        field data
    outputs:\n
        constraints=
        {
        *all_constr_attribs,
        "required",
        "origin", #through get_origin(annotation)
        "args",   #through get_args(annotation)
        }
    """
    cout = True
    if cout:
        print("__")
        print(f"	Name:{name}")
        print(f"	Field:{field}")
    input("field pause...")
    constraints = {}
    constraints["required"] = field["required"]
    for attr in all_constr_attribs:
        return_val = getattr(field, attr, None)
        constraints[attr] = return_val

    annotation = constraints["annotation"]
    constraints["origin"] = get_origin(annotation)
    constraints["args"] = get_args(annotation)
    if cout:
        print(f"Constraints:{constraints}")
    input("field pause2...")

    return constraints

def make_new_contraints(applied_constraints:Dict[str,Any]) -> Dict[str,Any]:
    """
    Inputs:\n
        applied constraints
    Outputs:\n
        copy of defaults, updated with input constraints
    fast method for code reusage\n
    """
    new_applied_constraints = default_constr_dict.copy()
    new_applied_constraints.update(applied_constraints)
    return new_applied_constraints

def get_applied_constraints(schema_model:JsonSchemaClass) -> Dict[str, Dict[str, Any]]:
    """
    Inputs:\n
        schema model
    Outputs:\n
        constraints for each field in the schema = {field name:constraints}
    """
    applied_constraints = {}

    for name, field in get_json_model_fields(schema_model).items():
        applied_constraints[name] = check_generation_constraints(name, field)

    return applied_constraints
