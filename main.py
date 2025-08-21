from pydantic import BaseModel, Field
import typing
from typing_extensions import Annotated  # part of pydantic

import typer

from functions.synthesiser import Synthesiser
from functions.anonymiser import Anonymiser


import yaml  # builtin
import json  # builtin


import importlib.util
import inspect  # builtin

import requests

import time


app = typer.Typer()


def synth_func(
    schema_model, method, amount, file_path, output, start_index=0, cout: bool = False
):
    """
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp_wrapper = lp(Synthesiser.synthesise)
    lp_wrapper(schema_model,method,amount)
    lp.print_stats()
    """
    start_time = time.time()
    dataset = Synthesiser.synthesise(
        schema_model, method, amount
    )  # returns as [ data, data, ... ]
    elapsed_time = time.time() - start_time  # end timer
    print(f"Generation | Time taken: {elapsed_time:.2f} seconds")
    flush = []
    if output.startswith("http"):
        request_entries = []

    if output.startswith("http"):
        batch_print_size = amount
    else:
        batch_print_size = max((amount // 10), 1)

    for index, data in enumerate(dataset):
        if file_path is not None:
            output_data = {
                key: str(value) for key, value in dict(data).items()
            }  # convert outputs to string
            if (index + 1 + start_index) != 1:
                front_string = ",\n    "
            else:
                front_string = ""
            flush.append(
                f'{front_string}"{index + start_index}": {json.dumps(output_data, indent=8)}'
            )
            if output.startswith("http"):
                request_entries.append(output_data)
        else:
            flush.append(f"{index + 1 + start_index}: {data}")
        if (index + 1) % batch_print_size == 0:
            if file_path is not None:
                if output.startswith("http"):
                    send_batch_to_API(schema_model, output, request_entries)
                flush_out = "".join(flush)
                with open(f"{file_path}.json", "a") as f:
                    f.write(flush_out)
            else:
                flush_out = "".join(flush)
            if cout:
                print(flush_out)
            flush.clear()
            if output.startswith("http"):
                request_entries.clear()
    if flush != []:
        if file_path is not None:
            if output.startswith("http"):
                send_batch_to_API(schema_model, output, request_entries)
            flush_out = "".join(flush)
            with open(f"{file_path}.json", "a") as f:
                f.write(flush_out)
        else:
            flush_out = "".join(flush)
        if cout:
            print(flush_out)
        flush.clear()
        if output.startswith("http"):
            request_entries.clear()
    if file_path is not None and cout:
        print(f"To file_path -> {file_path}")


def anon_func(
    schema_model, method, amount, start, file_path, ingest, cout, manual, fields, output
):
    if amount > 1:
        data = load_ingest_data(ingest, amount=amount, start=start)

        anonymised_data = [
            Anonymiser.anonymise(schema_model, data_item, method, manual, fields)
            for data_item in data
        ]

        for x in range(amount):
            if cout:
                L = len(f"        {data}")
                print("_" * L)
                print(f"Input data:\n\t{data[x]}")
                print(f"Anonymised:\n\t{anonymised_data[x]}")
                print("_" * L)

            flush_output = f"{start + x}:{json.dumps(anonymised_data[x], indent=8)}"
            if file_path is not None:
                with open(f"{file_path}.json", "a") as f:
                    f.write(flush_output)

    else:
        data = load_ingest_data(ingest, start=start)
        anonymised_data = Anonymiser.anonymise(
            schema_model, data, method, manual, fields
        )

        if cout:
            L = len(f"        {data}")
            print("_" * L)
            print(f"Input data:\n\t{data}")
            print(f"Anonymised:\n\t{anonymised_data}")
            print("_" * L)

        flush_output = f"{start}:{json.dumps(anonymised_data, indent=8)}"
        if file_path is not None:
            with open(f"{file_path}.json", "a") as f:
                f.write(flush_output)


def load_config(config):
    with open(config) as cf_file:
        try:
            # print(yaml.safe_load(cf_file))
            config_data = dict(yaml.safe_load(cf_file))
        except yaml.YAMLError as err:
            print(err)

    try:
        schema_file = config_data["schema"]
    except:
        print("Config 'schema' not defined")
        return 0

    if not (schema_file.endswith(".py")):
        schema_file += ".py"

    spec = importlib.util.spec_from_file_location("imported_schema_model", schema_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    classes = inspect.getmembers(module, inspect.isclass)
    # print(classes)
    filtered = [
        (name, cls)
        for name, cls in classes
        if cls.__module__ == "imported_schema_model"
    ]
    schema_model = filtered[0][1]
    # print(filtered)
    # print(schema_model)
    # User = module.User

    try:
        synthesiser_config = config_data["synthesiser"]
    except:
        synthesiser_config = []

    try:
        anonymiser_config = config_data["anonymiser"]
    except:
        anonymiser_config = []

    return {"schema": schema_model, "s": synthesiser_config, "a": anonymiser_config}


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
    method: str = Field(default="faker")
    amount: int = Field(default=1)
    batch: int = Field(default=0)
    output: str = Field(default="")
    cout: bool = Field(default=False)


class AnonymiserConfig(BaseModel):
    ingest: str = Field(default="")
    method: str = Field(default="faker")
    amount: int = Field(default=1)
    start: int = Field(default=0)
    output: str = Field(default="")
    cout: bool = Field(default=False)
    manual: bool = Field(default=False)
    fields: typing.Dict[str, str] = Field(default={})


def load_flags(func_type, flags):
    if func_type == "s":
        config_schema = SynthesiserConfig
    elif func_type == "a":
        config_schema = AnonymiserConfig

    unchanged = [
        field_name
        for field_name in config_schema.model_fields.keys()
        if flags[field_name] is None
    ]
    flag_data = dict(config_schema(**{}))

    if flags["config"] is not None:
        config_data = load_config(flags["config"])
        schema_model = config_data["schema"]
        if config_data[func_type]:
            flag_data.update(config_data[func_type])
        else:
            if func_type == "s":
                print("Config 'synthesiser' not defined")
            elif func_type == "a":
                print("Config 'anonymiser' not defined")
    else:
        print("Schema auto loader")
        import schema

        schema_model = schema.Address

    cli_args = {
        field_name: flags[field_name]
        for field_name in config_schema.model_fields.keys()
    }
    flag_data.update({k: v for k, v in cli_args.items() if k not in unchanged})

    return (flag_data, schema_model)


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
        raise ("Unsupported ingest type")
    return data


@app.command()
def synthesise(
    method: str = None,
    amount: int = None,
    batch: int = None,
    output: str = None,
    cout: Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
    config: str = None,
):
    flags = {
        "method": method,
        "amount": amount,
        "batch": batch,
        "output": output,
        "cout": cout,
        "config": config,
    }

    synthesiser_data, schema_model = load_flags("s", flags)

    method = synthesiser_data["method"]
    amount = synthesiser_data["amount"]
    batch = synthesiser_data["batch"]
    output = synthesiser_data["output"]
    cout = synthesiser_data["cout"]

    print("Args:", synthesiser_data)

    if output.startswith("http"):
        file_path = load_folder("_temp_db_output")
    else:
        if output is not None:
            file_path = load_folder(output)
        else:
            file_path = None
    ##	load folder		##

    if batch != 0:
        batch_index = 0
        for y in range(amount // batch):
            print("Batch: ", y + 1)
            synth_func(
                schema_model,
                method,
                batch,
                file_path,
                output,
                start_index=batch_index,
                cout=cout,
            )
            batch_index += batch
        if amount - batch_index != 0:
            synth_func(
                schema_model,
                method,
                amount - batch_index,
                file_path,
                output,
                start_index=batch_index,
                cout=cout,
            )
    else:
        synth_func(schema_model, method, amount, file_path, output, cout=cout)

    ##	close folder	##
    close_folder(file_path)


@app.command()
def anonymise(
    ingest: str = None,
    method: str = None,
    amount: int = None,
    start: int = None,
    output: str = None,
    manual: Annotated[
        typing.Optional[bool], typer.Option("--manual/--no-manual")
    ] = None,
    cout: Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
    config: str = None,
):
    fields = None
    flags = {
        "ingest": ingest,
        "method": method,
        "amount": amount,
        "start": start,
        "output": output,
        "manual": manual,
        "cout": cout,
        "config": config,
        "fields": fields,
    }

    anonymiser_data, schema_model = load_flags("a", flags)

    ingest = anonymiser_data["ingest"]
    method = anonymiser_data["method"]
    amount = anonymiser_data["amount"]
    start = anonymiser_data["start"]
    output = anonymiser_data["output"]
    cout = anonymiser_data["cout"]
    manual = anonymiser_data["manual"]
    fields = anonymiser_data["fields"]

    print("Args:", anonymiser_data)

    if output.startswith("http"):
        file_path = load_folder("_temp_db_output")
    else:
        if output is not None:
            file_path = load_folder(output)
        else:
            file_path = None
    ##	load folder		##

    if ingest is None:
        raise ("Config 'ingest' required")

    anon_func(
        schema_model,
        method,
        amount,
        start,
        file_path,
        ingest,
        cout,
        manual,
        fields,
        output,
    )

    ##	close folder	##
    close_folder(file_path)


if __name__ == "__main__":
    app()
