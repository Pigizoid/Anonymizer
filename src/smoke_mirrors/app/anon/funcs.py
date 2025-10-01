import json
from smoke_mirrors.app.helper_funcs import make_json_safe
from smoke_mirrors.anonymiser import anonymiser
from typing import Any, Dict, Union
from pathlib import Path
from smoke_mirrors.library.jsonschemaclass import JsonSchemaClass

def anon_func(
    schema_model:JsonSchemaClass,
    seed: Union[int,str,None],
    method:str,
    amount:int,
    start_index:int,
    ingest:Dict[str,Any],
    cout:bool,
    manual:bool,
    default:str,
    fields:Dict[str,str],
    output:Path,
    key_anon:bool
):
    """
    Inputs:\n
        pydantic schema model
        generation seed
        generation method of the methods "mixed","mimesis","faker"
        amount as an int, to generate per data index
        starting index (to optionally skip data indexes in the ingest)
        string to the ingest data (either .json or route http)
        cout boolean toggle for verbose printing
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
    anonymised_data = anonymiser.anonymise(
        schema_model, ingest, method, manual, default, fields, amount, seed=seed, key_anon=key_anon, cout=cout
    )
    # data returns as a dict of lists of dicts
    # { index: [model, * amount] }

    flush_output = []
    with open(f"{output}.json", "a") as f:
        for index,content in anonymised_data.items():
            if cout:
                print("-" * 60)
                print(f"Input data:\n\t{ingest[index]}")
                print("Output data:")
            flush_list = []
            for idx, x in enumerate(content):
                output_data = x
                if cout:
                    print(
                        f"output {str(idx)}{' ' * (10 - len(str(idx)))}{output_data}"
                    )
                flush_list.append(output_data)
            flush_output.append(flush_list)
        f.write(json.dumps(flush_output, indent=4, default= lambda v: make_json_safe(v)))
