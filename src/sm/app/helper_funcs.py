from pydantic import BaseModel
from decimal import Decimal
from sm.app.models import SynthesiserConfig, AnonymiserConfig
import importlib.util
import inspect
import requests
import time
import json
import json_schema_to_pydantic
from typing import Dict, Any


def convert_schema_to_JSON(schema_model):
    if isinstance(schema_model, type) and issubclass(schema_model, BaseModel):
        JSON_schema = schema_model.model_json_schema()

    elif isinstance(schema_model, dict) or isinstance(schema_model, list):
        JSON_schema = schema_model
    else:
        raise Exception("Unhandled schema type")
    return JSON_schema


def normalise_JSON_schema_to_pydantic(JSON_schema):
    pydantic_model = json_schema_to_pydantic.create_model(JSON_schema)
    return pydantic_model


def normalise_schema_to_pydantic(schema_model):
    return normalise_JSON_schema_to_pydantic(convert_schema_to_JSON(schema_model))


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
    import os

    dir_name = "outputs"
    file_path = f"{dir_name}\\{output}"
    if not os.path.isdir(dir_name):
        try:
            os.mkdir(dir_name)
            print(f"Directory '{dir_name}' created successfully")
            with open(f"{file_path}.json", "w") as f:
                f.write("{")
        except PermissionError:
            print(f"Permission denied: Unable to create '{dir_name}'")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        with open(f"{file_path}.json", "w") as f:
            f.write("{")
    return file_path


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


def load_schema(schema_path):
    """
    Inputs:\n
        string to the schema path\n
    Uses importlib to dynamically load and import schema model as "imported_schema_model"\n
    Orders schemas in schema file alphabetically during import (only importing one schema)\n
    Outputs:\n
        pydantic schema BaseModel\n
    """
    if not (schema_path.endswith(".py")):
        schema_path += ".py"

    spec = importlib.util.spec_from_file_location("imported_schema_model", schema_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    classes = inspect.getmembers(module, inspect.isclass)
    filtered = [
        (name, cls)
        for name, cls in classes
        if cls.__module__ == "imported_schema_model"
    ]
    if filtered == []:
        raise Exception(f"No pydantic schema in schema file {schema_path}")
    schema_model = filtered[0][1]  # automatically ordered alphabetically

    return schema_model


def load_file_path(output):
    if output.startswith("http"):
        file_path = load_folder("_temp_db_output")
    else:
        if output is not None:
            file_path = load_folder(output)
        else:
            file_path = None

    return file_path


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
    params = {key: param for key, param in ctx.params.items() if param is not None}
    if config_schema == SynthesiserConfig:
        flags = settings(schema_path=schema_path, synth=params)
    elif config_schema == AnonymiserConfig:
        flags = settings(schema_path=schema_path, anon=params)
    else:
        raise Exception(f"Input config schema '{config_schema.__name__}', not in ['SynthesiserConfig','AnonymiserConfig']")
    return flags


def load_ingest_data(ingest, start_index=0)-> Dict[str,Any]:
    """
    Inputs:\n
        ingest string and loads the data from file or http
    Uses start_index to offset where to begin loading data (json file start index unimplemented)\n
    Outputs:\n
        file data Dict[str,Any]
    """
    if ingest.startswith("http"):
        data = {start_index, requests.get(ingest, params={"id_num": start_index})}
    elif ingest.endswith(".json"):
        with open(ingest) as dt_file:
            try:
                data = json.load(dt_file)

                if isinstance(data, list):
                    if not all([isinstance(content, dict) for content in data]):
                        raise Exception(
                            "Data is type, list, expected list entries as type dict"
                        )
                    data = {x: content for x, content in enumerate(data)}
                elif isinstance(data, dict):
                    try:
                        data = {int(key): content for key, content in data.items()}
                    except:
                        raise Exception(
                            "Data is type, dict, expected data to be indexed by int"
                        )

            except Exception as e:
                data = {}
                raise Exception(f"Error loading ingest data: {e}")

    else:
        raise Exception("Unsupported ingest type")
    return data
