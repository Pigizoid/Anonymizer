from pydantic import BaseModel, create_model

from ...tools.model_funcs import get_model_fields

import ast


def subset_model(schema_model, field_names) -> BaseModel:
    """Inputs a pydantic schema model and a list of field names
    Ouputs a newly created model with the name "new_schema_model"
    containing only the fields in the list of field names"""
    fields = {
        name: (field.annotation, field.default)
        for name, field in get_model_fields(schema_model).items()
        if name in field_names
    }
    return create_model("new_schema_model", **fields)


def guess_type(value) -> type:
    """Inputs a value and determines its type and outputs the type"""
    for caster in (bool, int, float, complex, bytes, tuple, list, set, dict):
        if isinstance(value,caster):
            return caster
    if value == "true":
        return True
    elif value == "false":
        return False
    elif value == "null" or value == None:
        return None
    try:
        return type(ast.literal_eval(value))
    except:
        return str


def new_model(data, field_names) -> BaseModel:
    """Inputs a dict of data and a list of field_names
    Ouputs a newly created schema with the name "new_schema_model"
    containing the fields in the data that match the field names
    and each field has its data type automatically added"""
    fields = {
        name: (guess_type(content))
        for name, content in data.items()
        if name in field_names
    }
    return create_model("new_schema_model", **fields)
