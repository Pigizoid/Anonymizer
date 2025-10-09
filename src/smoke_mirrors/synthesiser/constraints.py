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


field_attr_map = {
    "default":["default"],
    "min_length":["minLength","minProperties","minItems"],
    "max_length":["maxLength","maxProperties","maxItems"],
    "pattern":["pattern"],
    "gt":["exclusiveMinimum"],
    "lt":["exclusiveMaximum"],
    "ge":["minimum"],
    "le":["maximum"],
    "multiple_of":["multipleOf"],
}

def check_generation_constraints(name: str, field: dict) -> Dict[str, Any]:
    """
    Inputs:\n
        field name
        field data
    outputs:\n
        constraints=
        {
        *all_constr_attribs,
        "required",
        "origin",
        "args",
        }
    """
    if field is None:
        raise Exception("Field input was None")
    constraints = {
        attr: next((field[k] for k in keys if k in field), None)
        for attr, keys in field_attr_map.items()
    }
    constraints.update({
        "required": field.get("required", True),
        "annotation": infer_json_type(field),
        "args": infer_json_args(field),
        "origin": field,
    })
    return constraints

def get_applied_constraints(schema_model:JsonSchemaClass) -> Dict[str, Dict[str, Any]]:
    """
    Inputs:\n
        schema model
    Outputs:\n
        constraints for each field in the schema = {field name:constraints}
    """
    applied_constraints = {}

    if schema_model.contents["type"] != "object":
        applied_constraints[schema_model.__name__] = check_generation_constraints(schema_model.__name__, schema_model.contents)
    else:
        fields = get_json_model_fields(schema_model).items()
        for name, field in fields:
            applied_constraints[name] = check_generation_constraints(name, field)

    return applied_constraints
