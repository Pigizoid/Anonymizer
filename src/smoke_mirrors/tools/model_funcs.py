from pydantic import BaseModel
from smoke_mirrors.library.jsonschemaclass import JsonSchemaClass
from pydantic.fields import FieldInfo
from typing import Union, Type, Dict, Tuple, List, Set, Any, Literal, Generator, get_origin, get_args
import inspect

ModelLike = Union[Type[BaseModel], BaseModel]

def flatten_types(t) -> Generator[type, None, None]:
    origin = get_origin(t)
    args = get_args(t)

    if origin is Union:
        for arg in args:
            if arg is not type(None):
                yield from flatten_types(arg)
    elif origin is not None:
        for arg in args:
            yield from flatten_types(arg)
    else:
        yield t

def is_model(t):
    return inspect.isclass(t) and issubclass(t, BaseModel)


def flatten_json_types(property: dict) -> Generator[type, None, None]:
    """
    Recursively yield all base types from a JSON Schema property.
    Handles anyOf, oneOf, allOf, items, prefixItems, enum, const, etc.
    """
    base_type = infer_json_type(property)
    args = infer_json_args(property)

    if base_type is Union:
        for arg in args:
            yield from flatten_json_types(arg)
    elif base_type in (list, tuple, set):
        for arg in args:
            yield from flatten_json_types(arg)
    elif base_type is JsonSchemaClass:
        yield property
    elif base_type is Literal:
        for val in args:
            yield type(val)
    else:
        yield base_type

def is_json_model(t):
    return (infer_json_type(t) is JsonSchemaClass)
    

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
    elif len(property) == 0:
        return str
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
                if "prefixItems" in property:
                    return tuple
                elif "items" in property:
                    if isinstance(property["items"],list):
                        return tuple
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
    elif "prefixItems" in property:
        data_args = property["prefixItems"]
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
        if "propertyNames" in property:
            property_names = {k:v for k,v in property["propertyNames"].items()}
            property_names.update({"type":"string"})
            data_args[0] = property_names
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

