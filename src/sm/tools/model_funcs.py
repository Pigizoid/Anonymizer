from pydantic import BaseModel
from sm.library.jsonschema import JsonSchemaClass
from pydantic.fields import FieldInfo
from typing import Union, Type, Dict, Tuple, List, Set, Any, Literal

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
    if not(isinstance(property,dict)):
        return property
    if "enum" in property:
        return Literal
    elif "const" in property:
        return Literal
    elif "type" in property:
        data_type = property["type"]
        if data_type == "null":
            return None
        elif data_type == "boolean":
            return bool
        elif data_type == "object":
            return dict
        elif data_type == "array":
            if "uniqueItems" in property and property["uniqueItems"] == True:
                return set
            else:
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
        elif isinstance(data_type,list):
            return [infer_json_type(t) for t in data_type]
        else:
            raise Exception(f"Unhandled json type '{data_type}'")
    elif "$ref" in property:
        return JsonSchemaClass
    elif "anyOf" in property:
        return Union
    elif "oneOf" in property:
        return Union
    elif "allOf" in property:
        return property["allOf"]
    else: # direct value e.g. "hello" or 100
        return type(property)

def infer_json_args(property:Dict[str,Any]):
    if not(isinstance(property,dict)):
        return property
    if "enum" in property:
        data_args = [p for p in property["enum"]]
    elif "items" in property:
        data_args = [property["items"]]
    elif "anyOf" in property:
        data_args = [p for p in property["anyOf"]]
    elif "oneOf" in property:
        data_args = [p for p in property["oneOf"]]
    elif "allOf" in property:
        data_args = [p for p in property["allOf"]]
    elif infer_json_type(property) == dict:
        data_args = [{"type":"string"},{"type":"string"}]
        if "additionalProperties" in property:
            if isinstance(property["additionalProperties"],dict):
                data_args[1] = property["additionalProperties"]
        elif "patternProperties" in property:
            pattern_properties = property["patternProperties"]
            data_args[0] = {"pattern":list(pattern_properties.keys())[0],"type":"string"}
            data_args[1] = list(pattern_properties.values())[0]

    else:
        data_args = []
    return data_args

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

