from pydantic import BaseModel, create_model
from sm.tools.model_funcs import get_model_fields
from typing import List, Any, Dict


def subset_model(schema_model:BaseModel, field_names:List[str]) -> BaseModel:
    """
    Inputs:\n
        schema model
        list of field names
    Outputs:\n
        schema model with name "new_schema_model" that contanins only field names in the list of field names
        inferences output data types from the input schema
    """
    fields = {
        name: (field.annotation, field.default)
        for name, field in get_model_fields(schema_model).items()
        if name in field_names
    }
    return create_model("new_schema_model", **fields)


def new_model(data:Dict[str,Any], field_names:List[str]) -> BaseModel:
    """
    Inputs:\n
        dict of data = {field_name:content}
        list of field names
    Outputs:\n
        schema model with name "new_schema_model" that contanins only field names in the list of field names
        inferences output data types from the input data
    """
    fields = {
        name: (type(content))
        for name, content in data.items()
        if name in field_names
    }
    return create_model("new_schema_model", **fields)
