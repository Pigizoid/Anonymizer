import json
from sm.app.helper_funcs import load_ingest_data
from sm.anonymiser.anonymiser import anonymise
from pydantic import BaseModel
from typing import Any, Dict, Union
from pathlib import Path

def anon_func(
    schema_model:BaseModel,
    seed: Union[int,str,None],
    method:str,
    amount:int,
    start_index:int,
    ingest:str,
    cout:bool,
    manual:bool,
    default:str,
    fields:Dict[str,str],
    output:Path
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
    data = load_ingest_data(ingest, start_index=start_index)
    # data comes in as a dict of dicts
    anonymised_data = anonymise(
        schema_model, data, method, manual, default, fields, amount, seed=seed
    )
    # data returns as a dict of lists of dicts
    # { index: [model, * amount] }

    flush_output = {}
    with open(f"{output}.json", "a") as f:
        for index, content in anonymised_data.items():
            if cout:
                print("-" * 60)
                print(f"Input data:\n\t{data[index]}")
                print("Output data:")
            flush_list = []
            for idx, x in enumerate(content):
                if cout:
                    print(
                        f"output {str(idx)}{' ' * (10 - len(str(idx)))}{x.model_dump()}"
                    )
                flush_list.append(x.model_dump())
            flush_output[index] = flush_list
        f.write(json.dumps(flush_output, indent=8))
