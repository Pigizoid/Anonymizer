from pydantic import BaseModel
from sm.library.jsonschema import JsonSchemaClass
from pydantic.fields import FieldInfo
from typing import Union, Type, Dict, Any, List

ModelLike = Union[Type[BaseModel], BaseModel]


def get_model_fields(schema_model: ModelLike) -> Dict[str, Any]:
    if isinstance(schema_model, type) and issubclass(schema_model, BaseModel):
        model_cls = schema_model
    elif isinstance(schema_model, BaseModel):
        model_cls = type(schema_model)
    else:
        raise TypeError("Schema must be a pydantic BaseModel class or instance")

    model_fields: Dict[str, FieldInfo] = getattr(model_cls, "model_fields", {})
    return model_fields


def get_model_data(model) -> List[list]:
    model_data = []
    for field_name, model_field in get_model_fields(model).items():
        model_data.append([field_name, model_field])
    return model_data


def get_json_model_fields(schema_model:JsonSchemaClass) -> Dict[str,Any]:
    model_fields: Dict[str, Dict[str,Any]] = schema_model.fields
    return model_fields

def get_json_model_data(json_model):
    model_data = []
    for field_name, model_field in get_json_model_fields(json_model).items():
        model_data.append([field_name, model_field])
    return model_data

def infer_json_type(property:Dict[str,Any]):
    if "type" in property:
        data_type = property["type"]
        if data_type == "null":
            return None
        elif data_type == "boolean":
            return bool
        elif data_type == "object":
            return dict
        elif data_type == "array":
            if isinstance(property["items"],list):
                return tuple
            else:
                return list
        elif data_type == "number":
            return float
        elif data_type == "integer":
            return int
        elif data_type == "string":
            return str
        else:
            raise Exception(f"Unhandled json type '{data_type}'")
    elif "$ref" in property:
        return JsonSchemaClass
    elif "anyOf" in property:
        types_list = [infer_json_type(p) for p in property["anyOf"]]
        return Union[*types_list]
    elif "oneOf" in property:
        types_list = [infer_json_type(p) for p in property["oneOf"]]
        return Union[*types_list]
    elif "allOf" in property:
        return property["allOf"]




