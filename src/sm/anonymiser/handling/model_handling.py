from pydantic import BaseModel, create_model

from sm.tools.model_funcs import get_model_fields


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
    if value is None:
        return type(None)
    try:
        if value.lower() in {"true", "false"}:
            return bool
    except:
        pass
    try:
        int(value)
        return int
    except:
        pass
    try:
        float(value)
        return float
    except:
        pass
    try:
        complex(value)
        return complex
    except:
        pass
    if isinstance(value, bytes):
        return bytes
    elif isinstance(value, tuple):
        return tuple
    elif isinstance(value, list):
        return list
    elif isinstance(value, set):
        return set
    elif isinstance(value, frozenset):
        return frozenset
    elif isinstance(value, dict):
        return dict
    else:
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
