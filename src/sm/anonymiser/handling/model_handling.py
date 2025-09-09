from pydantic import BaseModel, create_model

from sm.tools.model_funcs import get_model_fields



def subset_model(schema_model, field_names) -> BaseModel:
    fields = {
        name: (field.annotation, field.default)
        for name, field in get_model_fields(schema_model).items()
        if name in field_names
    }
    return create_model("new_schema_model", **fields)

def guess_type(value) -> type:
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
    fields = {
        name: (guess_type(content))
        for name, content in data.items()
        if name in field_names
    }
    return create_model("new_schema_model", **fields)
