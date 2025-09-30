from typing import Dict, Any, get_origin, get_args
from decimal import Decimal, ROUND_HALF_UP
import exrex
from pydantic import BaseModel, Field
from smoke_mirrors.library.jsonschemaclass import JsonSchemaClass
import inspect
from smoke_mirrors.pre_made_data import all_constr_attribs, default_constr_dict
from smoke_mirrors.tools.model_funcs import get_json_model_data, get_json_model_fields, infer_json_type, infer_json_args


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

field_attr_map = {
    "strip_whitespace":None,
    "to_upper":None,
    "to_lower":None,
    "strict":None,
    "default":["default"],
    "annotation":["type"],
    "min_length":["minLength","minProperties","minItems"],
    "max_length":["maxLength","maxProperties","maxItems"],
    "pattern":["pattern"],
    "gt":["exclusiveMinimum"],
    "lt":["exclusiveMaximum"],
    "ge":["minimum"],
    "le":["maximum"],
    "multiple_of":["multipleOf"],
    "allow_inf_nan":None,
    "max_digits":None,
    "decimal_places":None
}

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
    cout = False
    if cout:
        print("__")
        print(f"	Name:{name}")
        print(f"	Field:{field}")
    constraints = {}
    if "required" in field:
        constraints["required"] = field["required"]
    else:
        constraints["required"] = True
    for attr,map_attr in field_attr_map.items():
        if map_attr != None:
            for m_attr in map_attr:
                if m_attr in field:
                    return_val = field[m_attr]
                    break
                else:
                    return_val = None
            constraints[attr] = return_val
        else:
            constraints[attr] = None
    data_type = infer_json_type(field)
    data_Args = infer_json_args(field)
    annotation = data_type
    constraints["annotation"] = annotation
    constraints["origin"] = field
    constraints["args"] = data_Args
    if cout:
        print(f"Constraints:{constraints}")

    '''
    {
        'items': {
            'patternProperties': {
                '^\\d{3}(-\\d{6})?$': {
                    'items': {
                        'pattern': '^\\d{5}(-\\d{4})?$', 
                        'type': 'string'
                    }, 
                    'type': 'array'
                }
            }, 
            'type': 'object'
        }, 
        'title': 'Zip Code', 
        'type': 'array', 
        'required': True
    }
    '''

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
