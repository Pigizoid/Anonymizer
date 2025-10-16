from pydantic import BaseModel
from smoke_mirrors.library.jsonschemaclass import JsonSchemaClass
from pydantic.fields import FieldInfo
from typing import Union, Type, Dict, Tuple, List, Set, Any, Literal, Generator, get_origin, get_args
import inspect
import re
from copy import deepcopy
from jsonschema import validate

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

data_types_dict = {
    "null":None,
    "boolean":bool,
    "object":dict,
    "number":float,
    "integer":int,
    "string":str
}

def infer_json_type(property:Dict[str,Any]):
    if not(isinstance(property,dict)):
        return property
    elif len(property) == 0:
        return str
    if "enum" in property or "const" in property:
        return Literal 
    elif "type" in property:
        data_type = property["type"]

        if data_type in data_types_dict:
            return data_types_dict[data_type]
        elif isinstance(data_type,list):
            return [infer_json_type(t) for t in data_type]
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
        else:
            raise Exception(f"Unhandled json type '{data_type}'")
    elif "$ref" in property:
        return JsonSchemaClass
    elif "anyOf" in property or "oneOf" in property:
        return Union
    elif "allOf" in property:
        return property["allOf"]
    else: # direct value e.g. "hello" or 100
        return type(property)

def infer_json_args(property:Dict[str,Any]):
    if not(isinstance(property,dict)):
        return property

    for keyword in ("enum", "prefixItems", "anyOf", "oneOf"):
        if keyword in property:
            return property[keyword]
    
    if "items" in property:
        return [property["items"]]

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


def strip_json_schema_ids(schema_model):
    if isinstance(schema_model,list):
        for i,item in enumerate(schema_model):
            schema_model[i] = strip_json_schema_ids(item)
    elif isinstance(schema_model,dict):
        if "$id" in schema_model:
            schema_model.pop("$id")
        for key,value in schema_model.items():
            schema_model[key] = strip_json_schema_ids(value)
    return schema_model

def walk_json_schema_defs(schema_model,definitions=None,path="",name=""):
    if definitions is None:
        definitions = []
    if path == "":
        path = f"[{name}]"
    if isinstance(schema_model,list):
        for i,item in enumerate(schema_model):
            recursed_path = path+f"({i})"
            definitions = walk_json_schema_defs(item,definitions,recursed_path)
    elif isinstance(schema_model,dict):
        for key,value in schema_model.items():
            recursed_path = path+f"[{key}]"
            if key == "$anchor" or key == "$defs" or key == "$id":
                definitions = walk_json_schema_defs(value,definitions=definitions,path=recursed_path)
                if key == "$defs":
                    definitions.append(path+"[$defs]")
                else:
                    definitions.append(path)
            else:
                definitions = walk_json_schema_defs(value,definitions=definitions,path=recursed_path)
    return definitions

def return_json_schema_ref_names(schema_model,references=None):
    if references is None:
        references = []
    if isinstance(schema_model,list):
        for i,item in enumerate(schema_model):
            references = return_json_schema_ref_names(item,references=references)
    elif isinstance(schema_model,dict):
        for key,value in schema_model.items():
            if key == "$ref":
                value = value.split("/")[-1]
                value = value.split("#")[-1]
                references.append(value)
            else:
                references = return_json_schema_ref_names(value,references=references)
    return references

def walk_json_schema_refs(schema_model,references=None,path="",name=""):
    if references is None:
        references = []
    if path == "":
        path = f"[{name}]"
    if isinstance(schema_model,list):
        for i,item in enumerate(schema_model):
            recursed_path = path+f"({i})"
            references = walk_json_schema_refs(item,references=references,path=recursed_path)
    elif isinstance(schema_model,dict):
        for key,value in schema_model.items():
            if key == "$ref":
                references.append(path)
            else:
                recursed_path = path+f"[{key}]"
                references = walk_json_schema_refs(value,references=references,path=recursed_path)
    return references


def ref_pass_model(schema_model:JsonSchemaClass):
    definitions = walk_json_schema_defs(schema_model.contents,name=schema_model.__name__)
    references = walk_json_schema_refs(schema_model.contents,name=schema_model.__name__)
    return (references,definitions)

def ref_pass_openapi_model(schema_model:Dict[str,Any],name:str):
    definitions = [ "[components][schemas]"+path for path in walk_json_schema_defs(schema_model,name=name) ]
    references = [ "[components][schemas]"+path for path in walk_json_schema_refs(schema_model,name=name) ]
    return (references,definitions)


def load_parse_json_schema(schema_model:JsonSchemaClass):
    new_schema_model = deepcopy(schema_model.contents)
    references,definitions = ref_pass_model(schema_model)
    schema_name = schema_model.__name__
    context = {schema_name:schema_model.contents}
    def_id_items = {}

    for definition in definitions:
        tokens = re.findall(r'\[([^\]]+)\]|\((\d+)\)', definition)
        current = context
        last_key = None
        for key, idx in tokens:
            if key:  # dict access
                last_key = key
                current = current[key]
            elif idx:  # list access
                last_key = int(idx)
                current = current[int(idx)]
        if "$id" in current:
            if current["$id"].startswith("http"):
                def_id_items[current["$id"].split("/")[-1]] = last_key
            else:
                def_id_items[current["$id"]] = last_key
    
    for reference in references: # resolve all refs first
        tokens = re.findall(r'\[([^\]]+)\]|\((\d+)\)', reference)
        current = context
        parent = None
        last_key = None
        for key, idx in tokens:
            parent = current
            if key:  # dict access
                last_key = key
                current = current[key]
            elif idx:  # list access
                last_key = int(idx)
                current = current[int(idx)]
        new_val = current["$ref"]
        new_val = new_val.split("/")[-1]
        new_val = new_val.split("#")[-1]
        if new_val in def_id_items.keys():
            new_val = def_id_items[new_val]
        new_val = "#/$defs/"+new_val
        parent[last_key]["$ref"] = new_val

    built_defs = {}
    for definition in definitions:#  make all defs and give all defs to all schemas
        tokens = re.findall(r'\[([^\]]+)\]|\((\d+)\)', definition)
        current = context
        last_key = None
        for key, idx in tokens:
            if key:  # dict access
                last_key = key
                current = current[key]
            elif idx:  # list access
                last_key = int(idx)
                current = current[int(idx)]
        if "$id" in current:
            new_def = deepcopy(current)
            new_def["title"] = last_key
            built_defs[new_def["title"]] = new_def
        if "$anchor" in current:
            new_def = deepcopy(current)
            new_def["title"] = current["$anchor"]
            built_defs[new_def["title"]] = new_def
        if "$id" not in current and "$anchor" not in current: # $defs
            for name,schema in current.items():
                built_defs[name] = schema
    built_defs[schema_name] = strip_json_schema_ids(new_schema_model)
    new_schema_model = strip_json_schema_ids(context[schema_name])
    for built_def in built_defs.values():
        if "$defs" in built_def:
            built_def.pop("$defs")
    needed_built_defs = {name:strip_json_schema_ids(content) for name,content in built_defs.items() if name in return_json_schema_ref_names(new_schema_model)}
    new_schema_model["$defs"] = needed_built_defs
    new_schema_model["title"] = schema_name
    if "required" not in new_schema_model:
        new_schema_model["required"] = []

    return_schema = JsonSchemaClass(new_schema_model)
    return return_schema

def load_schemas_from_openapi(schema_model:Dict[str,Any]):
    if "openapi" not in schema_model:
        raise Exception(f"Schema model is not an openapi object")
    new_schema_model = deepcopy(schema_model)
    schemas = [ref_pass_openapi_model(schema,name) for name,schema in new_schema_model["components"]["schemas"].items()]
    schemas = [[reference for schema0 in schemas for reference in schema0[0]],[definition for schema1 in schemas for definition in schema1[1]]]
    context = {"components":new_schema_model["components"]}

    def_id_items = {}

    for definition in schemas[1]:
        tokens = re.findall(r'\[([^\]]+)\]|\((\d+)\)', definition)
        current = context
        last_key = None
        for key, idx in tokens:
            if key:  # dict access
                last_key = key
                current = current[key]
            elif idx:  # list access
                last_key = int(idx)
                current = current[int(idx)]
        if "$id" in current:
            if current["$id"].startswith("http"):
                def_id_items[current["$id"].split("/")[-1]] = last_key
            else:
                def_id_items[current["$id"]] = last_key

    for reference in schemas[0]: # resolve all refs first
        tokens = re.findall(r'\[([^\]]+)\]|\((\d+)\)', reference)
        current = context
        parent = None
        last_key = None
        for key, idx in tokens:
            parent = current
            if key:  # dict access
                last_key = key
                current = current[key]
            elif idx:  # list access
                last_key = int(idx)
                current = current[int(idx)]
        new_val = current["$ref"]
        new_val = new_val.split("/")[-1]
        new_val = new_val.split("#")[-1]
        if new_val in def_id_items.keys():
            new_val = def_id_items[new_val]
        new_val = "#/$defs/"+new_val
        parent[last_key]["$ref"] = new_val
    
    built_defs = {}
    for definition in schemas[1]:#  make all defs and give all defs to all schemas
        tokens = re.findall(r'\[([^\]]+)\]|\((\d+)\)', definition)
        current = context
        last_key = None
        for key, idx in tokens:
            if key:  # dict access
                last_key = key
                current = current[key]
            elif idx:  # list access
                last_key = int(idx)
                current = current[int(idx)]
        if "$id" in current:
            new_def = deepcopy(current)
            new_def["title"] = last_key
            built_defs[new_def["title"]] = new_def
        if "$anchor" in current:
            new_def = deepcopy(current)
            new_def["title"] = current["$anchor"]
            built_defs[new_def["title"]] = new_def
        if "$id" not in current and "$anchor" not in current: # $defs
            for name,schema in current.items():
                built_defs[name] = schema
    for name,schema in new_schema_model["components"]["schemas"].items():
        built_defs[name] = strip_json_schema_ids(schema)
    for built_def in built_defs.values():
        if "$defs" in built_def:
            built_def.pop("$defs")
    for name,schema in new_schema_model["components"]["schemas"].items():
        needed_built_defs = {name:strip_json_schema_ids(content) for name,content in built_defs.items() if name in return_json_schema_ref_names(schema)}
        schema["$defs"] = needed_built_defs
        schema["title"] = name
        if "required" not in schema:
            schema["required"] = []

    return_schemas = {name:strip_json_schema_ids(content) for name,content in new_schema_model["components"]["schemas"].items() }
    
    return return_schemas



def validate_instance_data(instance: Dict, schema: List[Dict],schema_instances=None) -> None:
    if schema_instances is not None:
        try:
            schema_instances[0].validate(instance)
        except:
            schema_instances[1].validate(instance)
    else:
        try:
            validate(instance=instance, schema=schema.contents)
        except:
            validate(instance=instance, schema=schema.sanitised_contents)


