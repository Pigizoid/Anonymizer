import json
from smoke_mirrors.app.helper_funcs import make_json_safe
from smoke_mirrors.anonymiser import anonymiser
from typing import Any, Dict, Union
from pathlib import Path
from smoke_mirrors.library.jsonschemaclass import JsonSchemaClass
import time
from rich import print

def anon_func(
    schema_model:JsonSchemaClass,
    seed: Union[int,str,None],
    method:str,
    amount:int,
    start_index:int,
    ingest:Dict[str,Any],
    stdcout:bool,
    manual:bool,
    default:str,
    fields:Dict[str,str],
    output:Path,
    key_anon:bool,
    performance:bool
):
    """
    Inputs:\n
        pydantic schema model
        generation seed
        generation method of the methods "mixed","mimesis","faker"
        amount as an int, to generate per data index
        starting index (to optionally skip data indexes in the ingest)
        string to the ingest data (either .json or route http)
        stdcout boolean toggle for verbose printing
        manual boolean toggle for automatic/manual processing modes
        default anonymisation method of the methods "mask","perturb","synth"
        dict of fields = {field_name:method} of the methods "default","mask","perturb","synth"
        Path for the output file (.json added by default)
    Loads data from ingest file\n
    Runs the anonymiser tool\n
    Outputs:\n
        data to the output file and optionally prints output to the screen
    """
    # data comes in as a dict of dicts
    start = time.time()
    anonymised_data = anonymiser.anonymise(
        schema_model, ingest, method, manual, default, fields, amount, seed=seed, key_anon=key_anon, stdcout=stdcout, performance=performance
    )
    end = time.time()
    elapsed_time = end-start
    # data returns as a dict of lists of dicts
    # { index: [model, * amount] }

    flush_output = []
    print(f"Anonymisation | Time taken: {elapsed_time:.2f} seconds")
    if stdcout:
        print(f"Writing to file : {output}")
        print(f"Showing 1 of {len(anonymised_data)}")
        print(f"Input data :\n\t{ingest[0]}")
        print(f"Output data :\n\t{list(anonymised_data.values())[0]}")
    with open(f"{output}.json", "a") as f:
        for _,content in anonymised_data.items():
            if len(content) == 1:
                flush_list = content[0]
            else:
                flush_list = []
                for output_data in content:
                    flush_list.append(output_data)
            flush_output.append(json.dumps(flush_list, indent=4, default= lambda v: make_json_safe(v)))
        flush_output = ",".join(flush_output)
        f.write(flush_output)
