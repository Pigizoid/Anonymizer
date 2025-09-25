from pydantic import BaseModel
from decimal import Decimal
from sm.app.models import SynthesiserConfig, AnonymiserConfig
import importlib.util
import inspect
import requests
import time
import json
from typing import Dict, Any, List, Union
from pathlib import Path
import os
from sm.library.jsonschemaclass import JsonSchemaClass
from jsonschema import validate
from sm.tools.model_funcs import get_model_fields, get_json_model_fields


def make_json_safe(obj):
    if isinstance(obj, (set, list, tuple, frozenset)):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    else:
        return obj


def load_folder(output):
    with open(f"{output}.json", "w") as f:
        f.write("{")


def close_folder(file_path):
    if file_path is not None:
        with open(f"{file_path}.json", "a") as f:
            f.write("\n}\n")


def send_to_API(schema_model, output, data):
    session = requests.Session()
    for entry in data:
        print(entry)
        response = session.post(
            output,
            headers={"Content-Type": "application/json"},
            params={"id_num": ""},
            json=entry,
        )
        # conn.request("POST", "/database/add", payload, headers)
        # response = requests.post(output,json=entry)
        print(response)


def send_batch_to_API(schema_model, output, data):
    start_time = time.time()

    session = requests.Session()
    response = session.post(
        output, headers={"Content-Type": "application/json"}, json=data
    )

    elapsed_time = time.time() - start_time  # end timer
    print(f"Response: {response} | Time taken: {elapsed_time:.2f} seconds")
    return response

# ----- recursive file loading -----
def load_schema_pydantic(schema_path:Path):
    """
    Inputs:\n
        string to the schema path
    Uses importlib to dynamically load and import schema model as "imported_schema_model"
    Orders schemas in schema file alphabetically during import
    Outputs:\n
        pydantic schema BaseModel
    """
    if not (str(schema_path).endswith(".py")):
        schema_path = Path(str(schema_path)+".py")
    try:
        spec = importlib.util.spec_from_file_location("imported_schema_model", schema_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        classes = inspect.getmembers(module, inspect.isclass)
        filtered = [
            {name: cls}
            for name, cls in classes
            if cls.__module__ == "imported_schema_model"
        ]
        if filtered == []:
            raise Exception(f"No pydantic schema in schema file {schema_path}")
        schema_models = filtered # automatically ordered alphabetically
    except:
        return None
    return schema_models

def load_schema_json(schema_path:Path):
    """
    Inputs:\n
        string to the schema path
    Outputs:\n
        JsonSchemaClass (json schema stored in a class with __name__)
    """
    if not (str(schema_path).endswith(".json")):
        schema_path = Path(str(schema_path)+".json")
    try:
        with open(schema_path,"r") as f:
            file_data = json.load(f)
            schema_models = JsonSchemaClass(schema_path.stem,file_data)
    except Exception as e:
        print(f"Exception: {e}")
        return None
    return schema_models

def load_schema(schema_path:Path):
    if str(schema_path).endswith(".py"):
        schema_models = load_schema_pydantic(schema_path)
    elif str(schema_path).endswith(".json"):
        schema_models = load_schema_json(schema_path)
    else:
        return None
    return schema_models

def load_ingest(ingest_path:Path):
    return ingest_path

def load_recursed_path(recursed_path:Path, file_type:str, loading_func):
    if file_type[0] != ".":
        file_type = "."+file_type
    if recursed_path.is_dir():
        # folder case
        result = []
        for item in recursed_path.iterdir():
            return_val = load_recursed_path(item,file_type,loading_func)
            if not return_val:
                continue
            result.append(return_val)
        return {recursed_path.stem: result} if result else None
    elif recursed_path.suffix == file_type:
        # file case
        return {recursed_path.stem: loading_func(recursed_path)}
    else:
        return None
    '''
    {
        'schema.py': <class 'imported_schema_model.Address'>
    }
    all data: {
        'models': [
            {
                'schema1': <class 'imported_schema_model.Address'>
            },
            {
                'schema2': <class 'imported_schema_model.Address'>
            },
            {
                'submodels1': [
                    {
                        'schema1': <class 'imported_schema_model.Address'>
                    },
                    {  
                        'schema2': <class 'imported_schema_model.Address'>
                    }
                ]
            }, 
            {
                'submodels2': [
                    {
                        'schema1': <class 'imported_schema_model.Address'>
                    }, 
                    {
                        'schema2': <class 'imported_schema_model.Address'>
                    }
                ]
            }
        ]
    }
    '''


def schemas_equal(a, b) -> bool:
    """Check if two schemas are equivalent by comparing their fields."""
    if isinstance(a, JsonSchemaClass):
        fields_a = get_json_model_fields(a).items()
    else:
        fields_a = get_model_fields(a).items()

    if isinstance(b, JsonSchemaClass):
        fields_b = get_json_model_fields(b).items()
    else:
        fields_b = get_model_fields(b).items()

    return fields_a == fields_b


def flatten_loaded_schemas(
    schema_models: Union[Dict, List, BaseModel, JsonSchemaClass]
) -> List[Any]:
    result: List[Any] = []

    def _flatten(schema) -> None:
        if schema is None:
            return
        if isinstance(schema, dict):
            for value in schema.values():
                _flatten(value)
        elif isinstance(schema, list):
            for item in schema:
                _flatten(item)
        else:
            if not any(
                (r.__name__ == schema.__name__ and schemas_equal(r, schema))
                for r in result
            ):
                result.append(schema)

    _flatten(schema_models)
    return result
# ----- recursive file loading -----


# ----- ingest handling -----
def load_ingest_data(ingest, start_index=0)-> Dict[str,Any]:
    """
    Inputs:\n
        ingest string and loads the data from file or http
    Uses start_index to offset where to begin loading data (json file start index unimplemented)\n
    Outputs:\n
        file data Dict[str,Any]
    """
    if str(ingest).startswith("http"):
        data = {start_index, requests.get(ingest, params={"id_num": start_index})}
    elif str(ingest).endswith(".json"):
        with open(ingest) as dt_file:
            try:
                data = json.load(dt_file)

                if isinstance(data, list):
                    if not all([isinstance(content, dict) for content in data]):
                        raise Exception(
                            "Data is type of list, expected list entries as type dict"
                        )
                    data = {x: content for x, content in enumerate(data)}
                elif isinstance(data, dict):
                    try:
                        data = {int(key): content for key, content in data.items()}
                    except:
                        raise Exception(
                            "Data is type of dict, expected data to be indexed by int"
                        )

            except Exception as e:
                data = {}
                raise Exception(f"Error loading ingest data: {e}")

    else:
        raise Exception("Unsupported ingest type")
    return data

def find_matching_schema(schema_models:List[Union[BaseModel,JsonSchemaClass]],ingest,ingest_path):
    matched_schemas = []
    first_entry = True
    for key,data_entry in ingest.items():
        for schema_model in schema_models:
            try:
                if type(schema_model) == JsonSchemaClass:
                    if schema_model.fields.keys() == data_entry.keys():
                        try:
                            validate(instance=data_entry, schema=schema_model.contents)
                        except:
                            validate(instance=data_entry, schema=schema_model.sanitised_contents)
                    else:
                        continue
                else:
                    if get_model_fields(schema_model).keys() == data_entry.keys():
                        schema_model(**data_entry)
                    else:
                        continue
                if first_entry == True:
                    matched_schemas.append(schema_model)
                elif schema_model not in matched_schemas:
                    raise Exception(f"Data entry '{key}' in path '{ingest_path}' has mismatched schema validation")
            except:
                continue
        if first_entry == True:
            first_entry = False
    if len(matched_schemas) == 0:
        print(f"Data entry '{key}' in path '{ingest_path}' has no matched schema")
        return schema_models[0]
    return matched_schemas[0] #(return first matched schema)
# ----- ingest handling -----


# ----- recursive handling -----
def get_unique_folder_name(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    parent = base_path.parent
    stem = base_path.name
    counter = 1
    new_path = parent / f"{stem}_(copy)"
    while new_path.exists():
        counter += 1
        new_path = parent / f"{stem}_(copy {counter})"
    return new_path

def recursive_folder_schema_handler(schema_models,command,flags,output_path_name:Path,depth=0,use_parent=False):
    #1. check if output_path_name directory exists (could be nested)
    #2. if it doesnt exist, create it (may have to be created within a sub folder)
    if not os.path.exists(output_path_name):
        os.makedirs(output_path_name)
    elif depth == 0 and use_parent==False:
        output_path_name = get_unique_folder_name(Path(output_path_name))
        os.makedirs(output_path_name)
    for file_path,contents in schema_models.items():
        new_path = Path(os.path.join(output_path_name, file_path))
        if type(contents) is list:
            print(f"{' '*(4*depth)}| path: {file_path}| contents: list|")
            for inner_path in contents:
                recursive_folder_schema_handler(inner_path,command,flags,new_path,depth=depth+1)
        else:
            print(f"{' '*(4*depth)}| path: {file_path}| contents: {type(contents),contents.__name__}| {new_path}")
            if contents == None:
                raise Exception(f"Error in file {file_path}")
            schema_model = contents
            command(schema_model,new_path,flags=flags)
    return None

def recursive_ingest_json_handler(ingests,command,flags,output_path_name:Path,schema_models,depth=0):
    #1. check if output_path_name directory exists (could be nested)
    #2. if it doesnt exist, create it (may have to be created within a sub folder)
    if not os.path.exists(output_path_name):
        os.makedirs(output_path_name)
    elif depth == 0:
        output_path_name = get_unique_folder_name(output_path_name)
        os.makedirs(output_path_name)
    for file_path,contents in ingests.items():
        new_path = Path(os.path.join(output_path_name, file_path))
        if type(contents) is list:
            print(f"{' '*(4*depth)}| path: {file_path}| contents: list|")
            for inner_path in contents:
                recursive_ingest_json_handler(inner_path,command,flags,new_path,schema_models,depth=depth+1)
        else:
            print(f"{' '*(4*depth)}| path: {file_path}| contents: {contents}| {new_path}")
            ingest = load_ingest_data(contents)
            schema_model = find_matching_schema(schema_models,ingest,file_path)
            command(schema_model,new_path,ingest,flags=flags)
    return None
# ----- recursive handling -----

def load_file_path(output):
    if str(output).startswith("http"):
        load_folder("_temp_db_output")
    else:
        if output is not None:
            load_folder(output)

def load_output_path_flag(file_path):
    pathobj = Path(file_path)
    print("p:",pathobj,"s:",file_path)
    if len(pathobj.parts)>1:
        return_file_path =  pathobj.parent / "outputs" / pathobj.name
    else:
        return_file_path = Path("outputs") / pathobj
    return return_file_path


def return_flags(ctx, config_schema:BaseModel)->Dict[str,Any]:
    """
    Key Note: Settings is a dynamically loaded function created by the typer CLI context as ctx\n
    Inputs:\n
        context produced from typer CLI
        config schema from app models "SynthesiserConfig" or "AnonymiserConfig"
    Calls the settings function with applied settings sources in order of importance CLI > yaml > default\n
    with CLI taking priority over config\n
    yaml is loaded from CLI flag\n
    Outputs:\n
        updated and ordered input flags based on config_schema
    """
    settings = ctx.obj["settings"]
    schema_path = ctx.obj["schema_path"]
    schema_type = ctx.obj["schema_type"]
    params = {key: param for key, param in ctx.params.items() if param is not None}
    if config_schema == SynthesiserConfig:
        flags = settings(schema_path=schema_path, schema_type=schema_type, synth=params)
    elif config_schema == AnonymiserConfig:
        flags = settings(schema_path=schema_path, schema_type=schema_type, anon=params)
    else:
        raise Exception(f"Input config schema '{config_schema.__name__}', not in ['SynthesiserConfig','AnonymiserConfig']")
    return flags