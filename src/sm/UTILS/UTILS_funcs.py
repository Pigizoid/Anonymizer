#UTILS_funcs.py

from pydantic import BaseModel
from decimal import Decimal

try:
    from ..functions.synthesiser import Synthesiser
    from ..functions.anonymiser import Anonymiser
except:
    print("running main directly?")
    from ..functions.synthesiser import Synthesiser
    from ..functions.anonymiser import Anonymiser


import importlib.util
import inspect

import requests

import time

import json

import json_schema_to_pydantic

def convert_schema_to_JSON(schema_model):
    if isinstance(schema_model, type) and issubclass(schema_model,BaseModel):
        JSON_schema = schema_model.model_json_schema()
    
    elif isinstance(schema_model,dict) or isinstance(schema_model,list):
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
        return obj.decode('utf-8', errors='replace')
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


# import time
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
    
    schema_path

    if not (schema_path.endswith(".py")):
        schema_path += ".py"

    spec = importlib.util.spec_from_file_location("imported_schema_model", schema_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    classes = inspect.getmembers(module, inspect.isclass)
    # print(classes)
    filtered = [
        (name, cls)
        for name, cls in classes
        if cls.__module__ == "imported_schema_model"
    ]
    if filtered == []:
        raise Exception(f"No pydantic schema in schema file {schema_path}")
    schema_model = filtered[0][1]  #automatically ordered alphabetically
    
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


def return_flags(ctx,config_schema):
    settings = ctx.obj["settings"]
    schema_path = ctx.obj["schema_path"]
    params = {key:param for key,param in ctx.params.items() if param is not None}
    flags = settings(schema_path=schema_path,synth=config_schema(**params))
    return flags


def load_ingest_data(ingest, amount=1, start=0):
    if ingest.startswith("http"):
        if amount > 1:
            data = []
            for x in range(amount):
                response = requests.get(ingest, params={"id_num": start + x})
                data.append(response.json())
        else:
            response = requests.get(ingest, params={"id_num": start})
            data = response.json()
    elif ingest.endswith(".json"):
        with open(ingest) as dt_file:
            try:
                data = json.load(dt_file)
            except Exception as e:
                print(e)
                data = {}
    else:
        raise Exception("Unsupported ingest type")
    return data


def synth_func(
    schema_model, method, amount, output, start_index=0, cout: bool = False
):
    """
    from line_profiler import LineProfiler
    lp = LineProfiler()
    synth = Synthesiser()
    lp_wrapper = lp(synth.synthesise)
    lp_wrapper(schema_model,method,amount)
    lp.print_stats()
    #"""
    start_time = time.time()
    synth = Synthesiser(method=method)
    dataset = synth.synthesise(
        schema_model, method, amount
    )  # returns as [ data, data, ... ]
    elapsed_time = time.time() - start_time  # end timer
    print(f"Generation | Time taken: {elapsed_time:.2f} seconds")
    flush = []
    if output.startswith("http"):
        request_entries = []

    for index, data in enumerate(dataset):
        if output is not None:
            """
            output_data = {}
            for key, value in dict(data).items():
                output_data[key] = make_json_safe(value)
            """
            if (index + 1 + start_index) != 1:
                front_string = ",\n	"
            else:
                front_string = ""
            json_str = json.dumps(
                data.model_dump(), 
                indent=8, 
                default=lambda v: repr(v)
            )
            flush.append(
                f'{front_string}"{index + start_index}": {json_str}'
            )
            if output.startswith("http"):
                request_entries.append(data)
        else:
            flush.append(f"{index + 1 + start_index}: {data}")
    print("printed")
    if output is not None:
        if output.startswith("http"):
            send_batch_to_API(schema_model, output, request_entries)
        else:
            flush_out = "".join(flush)
            with open(f"{output}.json", "a") as f:
                f.write(flush_out)
    else:
        flush_out = "".join(flush)
    if cout:
        print(flush_out)
    flush.clear()
    if output.startswith("http"):
        request_entries.clear()
    if output is not None and cout:
        print(f"To file_path -> {output}")


def anon_func(
    schema_model, method, amount, start, ingest, cout, manual, fields, output
):
    if amount > 1:
        data = load_ingest_data(ingest, amount=amount, start=start)

        anonymised_data = [
            Anonymiser.anonymise(schema_model, data_item, method, manual, fields)
            for data_item in data
        ]

        for index, data_entry in enumerate(anonymised_data):
            output_data = {}
            for key, value in dict(data_entry).items():
                output_data[key] = make_json_safe(value)
            anonymised_data[index] = output_data

        for x in range(amount):
            if cout:
                L = len(f"		{data}")
                print("_" * L)
                print(f"Input data:\n\t{data[x]}")
                print(f"Anonymised:\n\t{anonymised_data[x]}")
                print("_" * L)

            flush_output = f"{start + x}:{json.dumps(anonymised_data[x], indent=8)}"
            if output is not None:
                with open(f"{output}.json", "a") as f:
                    f.write(flush_output)

    else:
        data = load_ingest_data(ingest, start=start)
        anonymised_data = Anonymiser.anonymise(
            schema_model, data, method, manual, fields
        )

        for index, data_entry in enumerate(anonymised_data):
            output_data = {}
            for key, value in dict(data_entry).items():
                output_data[key] = make_json_safe(value)
            anonymised_data[index] = output_data

        if cout:
            L = len(f"		{data}")
            print("_" * L)
            print(f"Input data:\n\t{data}")
            print(f"Anonymised:\n\t{anonymised_data}")
            print("_" * L)

        flush_output = f"{start}:{json.dumps(anonymised_data, indent=8)}"
        if output is not None:
            with open(f"{output}.json", "a") as f:
                f.write(flush_output)