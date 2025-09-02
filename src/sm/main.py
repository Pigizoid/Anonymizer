#UTILS_main.py




from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import typing
from typing import Optional, Dict, Tuple, List, Any
from decimal import Decimal
from pathlib import Path

from typing_extensions import Annotated

import typer

try:
    from .functions.synthesiser import Synthesiser
    from .functions.anonymiser import Anonymiser
except:
    print("running main directly?")
    from functions.synthesiser import Synthesiser
    from functions.anonymiser import Anonymiser


import importlib.util
import inspect

import requests

import time

import yaml
import json

import json_schema_to_pydantic


app = typer.Typer()

synth_app = typer.Typer()         # sub-app for synth
anon_app = typer.Typer()          # sub-app for anon

app.add_typer(synth_app, name="synth")
app.add_typer(anon_app, name="anon")




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






class SynthesiserConfig(BaseModel):
    method: str = Field(default="mixed")
    amount: int = Field(default=1)
    batch: int = Field(default=0)
    output: str = Field(default="")
    cout: bool = Field(default=False)


class AnonymiserConfig(BaseModel):
    ingest: str = Field(default="")
    method: str = Field(default="mixed")
    amount: int = Field(default=1)
    start: int = Field(default=0)
    output: str = Field(default="")
    cout: bool = Field(default=False)
    manual: bool = Field(default=False)
    fields: typing.Dict[str, str] = Field(default={})


def make_settings_class(config_path: Optional[str]) -> type[BaseSettings]:
    
    def yaml_settings_source() -> Dict[str, Any]:
        
        if not Path(config_path).exists():
            return {}  #returning {} as empty to allow defaults to parse

        try:
            raw = yaml.safe_load(Path(config_path).read_text()) or {}
        except Exception as e:
            print(f"Yaml doesnt exist: {e}")
            return {}
        
        mapping: Dict[str, Any] = {}
        if "schema" in raw:
            mapping["schema_path"] = raw["schema"]
        elif "schema_path" in raw:
            mapping["schema_path"] = raw["schema_path"]
        
        if "synthesiser" in raw:
            mapping["synth"] = raw["synthesiser"]
        elif "synth" in raw:
            mapping["synth"] = raw["synth"]

        if "anonymiser" in raw:
            mapping["anon"] = raw["anonymiser"]
        elif "anonymizer" in raw:
            mapping["anon"] = raw["anonymizer"]
        elif "anon" in raw:
            mapping["anon"] = raw["anon"]

        return mapping
    
    def schema_defaults_source() -> Dict[str, Any]:
        
        synth_defaults = {name: field.default for name, field in SynthesiserConfig.model_fields.items()}
        anon_defaults = {name: field.default for name, field in AnonymiserConfig.model_fields.items()}
        defaults = {
            "schema_path": "schema.py",
            "synth": synth_defaults,
            "anon": anon_defaults
        }
        return defaults
    
    class Settings(BaseSettings):
        schema_path: str
        synth: typing.Optional[SynthesiserConfig]
        anon: typing.Optional[AnonymiserConfig]
    
    def _settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,          # highest priority: values passed into the constructor
            yaml_settings_source,   # next priority: values from YAML file
            schema_defaults_source, # then schema defaults loaded from schema.py
            env_settings,
            file_secret_settings,
        )
    
    Settings.settings_customise_sources = classmethod(_settings_customise_sources)
    return Settings


def return_flags(ctx,config_schema):
    settings = ctx.obj["settings"]
    schema_path = ctx.obj["schema_path"]
    params = {key:param for key,param in ctx.params.items() if param is not None}
    flags = settings(schema_path=schema_path,synth=config_schema(**params))
    return flags

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

# -----
# main app
# -----
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Optional[str] = typer.Option(
        "config.yaml",  #default
        exists=False,         #dont check if path exists before allowing it as option
        file_okay=True,
        dir_okay=False,       #these 3 check its readable and a file
        readable=True,
        help="Path to YAML config file",
    ),
    schema_path: Optional[str] = typer.Option(
        "schema.py",
        exists=False,
    )
):
    Settings = make_settings_class(config)
    ctx.obj = {
        "settings": Settings,
        "schema_path": schema_path,
    }

# -----
# synthesiser commands
# -----
@synth_app.callback(invoke_without_command=True)  #call default with no subcommand
def synth_single(
    ctx: typer.Context,  #contains ctx.config
    method: str = None,
    output: str = None,
    cout: Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
):  
    if ctx.invoked_subcommand is not None: #this allows a default without running extra code
        return
    flags = return_flags(ctx, SynthesiserConfig)
    print(flags)
    schema_model = load_schema(flags.schema_path)
    synth_flags = flags.synth
    output_file_path = load_file_path(synth_flags.output)
    
    synth_func(schema_model, synth_flags.method, 1, output_file_path, cout=synth_flags.cout)
    
    close_folder(output_file_path)

@synth_app.command(name="batch")  #call with "batch" sub command
def synth_batch(
    ctx: typer.Context,  #contains ctx.config
    method: str = None,
    amount: int = None,
    batch: int = None,
    output: str = None,
    cout: Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
):
    flags = return_flags(ctx, SynthesiserConfig)
    print(flags)
    schema_model = load_schema(flags.schema_path)
    synth_flags = flags.synth
    output_file_path = load_file_path(synth_flags.output)
    
    method = synth_flags.method
    amount = synth_flags.amount
    if synth_flags.batch == 0:
        batch = amount  
    else:
        batch = synth_flags.batch
    cout = synth_flags.cout
    batch_index = 0
    for y in range(amount // batch):
        print("Batch: ", y + 1)
        synth_func(
            schema_model,
            method,
            batch,
            output_file_path,
            start_index=batch_index,
            cout=cout,
        )
        batch_index += batch
    if amount - batch_index != 0:
        synth_func(
            schema_model,
            method,
            amount - batch_index,
            output_file_path,
            start_index=batch_index,
            cout=cout,
        )
    
    close_folder(output_file_path)

# -----
# anonymiser commands
# -----
@anon_app.callback(invoke_without_command=True)  #call with no subcommand
def anon_auto(
    ctx: typer.Context,  #contains ctx.config
    ingest: str = None,
    method: str = None,
    output: str = None,
    cout: Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
):
    if ctx.invoked_subcommand is not None: #this allows a default without running extra code
        return
    flags = return_flags(ctx, AnonymiserConfig)
    print(flags)
    schema_model = load_schema(flags.schema_path)
    anon_flags = flags.anon
    output_file_path = load_file_path(anon_flags.output)
    
    if anon_flags.ingest is None:
        raise Exception("Config 'ingest' required")
    
    anon_func(
        schema_model,
        anon_flags.method,
        anon_flags.amount,
        anon_flags.start,
        anon_flags.ingest,
        anon_flags.cout,
        anon_flags.manual,
        anon_flags.fields,
        output_file_path,
    )
    
    close_folder(output_file_path)

@anon_app.command(name="manual")  #call with "manual" sub command
def anon_manual(
    ctx: typer.Context,  #contains ctx.config and ctx.params  (params are the below field
    ingest: str = None,
    method: str = None,
    amount: int = None,
    start: int = None,
    output: str = None,
    cout: Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
    fields: str = typer.Option(None, help="Fields as JSON string"),
):
    if fields:
        try:
            ctx.params["fields"] = json.loads(fields)
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"Invalid JSON for --fields: {e}")
    else:
        ctx.params["fields"] = None
    
    flags = return_flags(ctx, AnonymiserConfig)
    print(flags)
    schema_model = load_schema(flags.schema_path)
    anon_flags = flags.anon
    output_file_path = load_file_path(anon_flags.output)
    
    if anon_flags.ingest is None:
        raise Exception("Config 'ingest' required")
    
    anon_func(
        schema_model,
        anon_flags.method,
        anon_flags.amount,
        anon_flags.start,
        anon_flags.ingest,
        anon_flags.cout,
        anon_flags.manual,
        anon_flags.fields,
        output_file_path,
    )
    
    close_folder(output_file_path)







'''
config = "config.yaml"
Settings = make_settings_class(config)
print(Settings)

flags = {
    "method": None,
    "amount": 10000,
    "batch": 50,
    "output": "outputs.json",
    "cout": False
    }
flags = {key:param for key,param in flags.items() if param is not None}
synth = SynthesiserConfig(**flags)

print(Settings(synth=synth))
'''

'''





sm             

->  synthesise -> single
    -> batch
    
->  anonymise -> auto
    -> manual


sm synthesise *args (default is to use the "single" flag)
sm synthesise batch *args
sm anonymise *args (default is to use the "auto" flag)
sm anonymise manual *args



'''














