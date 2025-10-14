import json
import time
from smoke_mirrors.app.helper_funcs import send_batch_to_API, make_json_safe
from smoke_mirrors.synthesiser.synthesiser import JsonSynthesiser
from pathlib import Path
from typing import Union
from smoke_mirrors.library.jsonschemaclass import JsonSchemaClass
from rich import print

def synth_func(
    schema_model:JsonSchemaClass,
    method:str,
    amount:int,
    output:Path,
    stdcout:bool = False,
    start_index:int = 0,
    seed: Union[int,str,None]="random",
    performance: bool = False
):
    """
    Inputs:\n
        schema_model: pydantic schema model or JSON_schema as JsonSchemaClass
        generation method of the methods "mixed","mimesis","faker"
        amount as an int, to generate per schema
        Path the output file (.json added by default)
        stdcout boolean toggle for verbose printing
        optional starting index used in batching to specify what the output index should start at
        optional generation seed
    Loads data from ingest file\n
    Runs the anonymiser tool\n
    Outputs:\n
        data to the output file and optionally prints output to the screen
    """
    start_time = time.time()
    synth = JsonSynthesiser(method=method,stdcout=stdcout,rich_output=True)
    dataset = synth.synthesise(
        schema_model, method, amount, seed, performance=performance
    )  # returns as [ data, data, ... ]
    elapsed_time = time.time() - start_time  # end timer
    print(f"Generation | Time taken: {elapsed_time:.2f} seconds")
    flush = []
    if str(output).startswith("http"):
        request_entries = []
    if stdcout:
        print(f"Writing to file : {output}")
    for index,data in enumerate(dataset):
        if output is not None:
            if (index == 0 and start_index != 0):
                front_string = ","
            else:
                front_string = ""
            json_str = front_string+json.dumps(data, indent=4, default=lambda v: make_json_safe(v))
            flush.append(json_str)
            if str(output).startswith("http"):
                request_entries.append(data)
        else:
            flush.append(data)
    if output is not None:
        if str(output).startswith("http"):
            send_batch_to_API(schema_model, output, request_entries)
        else:
            flush_out = ",".join(flush)
            stem = output.stem
            stem+="_(temp)"
            output = output.with_stem(stem)
            output = output.with_suffix(".json")
            with open(output, "a") as f:
                f.write(flush_out)
    else:
        flush_out = ",".join(flush)
    flush.clear()
    if str(output).startswith("http"):
        request_entries.clear()
