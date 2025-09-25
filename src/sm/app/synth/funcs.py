import json
import time
from sm.app.helper_funcs import send_batch_to_API
from sm.synthesiser.synthesiser import Synthesiser
from sm.synthesiser_json.synthesiser import JsonSynthesiser
from pathlib import Path
from typing import Any, Union
from pydantic import BaseModel
from sm.library.jsonschemaclass import JsonSchemaClass

def synth_func(
    schema_model:Union[BaseModel,JsonSchemaClass],
    method:str,
    amount:int,
    output:Path,
    cout:bool = False,
    start_index:int = 0,
    seed: Union[int,str,None]="random",
):
    """
    Inputs:\n
        schema_model: pydantic schema model or JSON_schema as JsonSchemaClass
        generation method of the methods "mixed","mimesis","faker"
        amount as an int, to generate per schema
        Path the output file (.json added by default)
        cout boolean toggle for verbose printing
        optional starting index used in batching to specify what the output index should start at
        optional generation seed
    Loads data from ingest file\n
    Runs the anonymiser tool\n
    Outputs:\n
        data to the output file and optionally prints output to the screen
    """
    start_time = time.time()
    json_schema_flag = False
    if type(schema_model) == JsonSchemaClass:
        json_schema_flag = True
        synth = JsonSynthesiser(method=method)
    else:
        synth = Synthesiser(method=method)
    dataset = synth.synthesise(
        schema_model, method, amount, seed
    )  # returns as [ data, data, ... ]
    elapsed_time = time.time() - start_time  # end timer
    print(f"Generation | Time taken: {elapsed_time:.2f} seconds")
    flush = []
    if str(output).startswith("http"):
        request_entries = []

    for index, data in enumerate(dataset):
        if output is not None:
            if (index + 1 + start_index) != 1:
                front_string = ",\n	"
            else:
                front_string = ""
            if json_schema_flag:
                json_str = json.dumps(data, indent=4, default=lambda v: str(v))
            else:
                json_str = json.dumps(
                    data.model_dump(), indent=4, default=lambda v: str(v)
                )
            flush.append(f'{front_string}"{index + start_index}": {json_str}')
            if str(output).startswith("http"):
                request_entries.append(data)
        else:
            flush.append(f"{index + 1 + start_index}: {data}")
    if output is not None:
        if str(output).startswith("http"):
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
    if str(output).startswith("http"):
        request_entries.clear()
    if output is not None and cout:
        print(f"To file_path -> {output}")
