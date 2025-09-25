import json
from sm.app.helper_funcs import load_ingest_data, make_json_safe
from sm.anonymiser import anonymiser, anonymiser_json
from pydantic import BaseModel
from typing import Any, Dict, Union
from pathlib import Path
from sm.library.jsonschemaclass import JsonSchemaClass

def anon_func(
    schema_model:Union[BaseModel,JsonSchemaClass],
    seed: Union[int,str,None],
    method:str,
    amount:int,
    start_index:int,
    ingest:Dict[str,Any],
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
    # data comes in as a dict of dicts
    json_schema_flag = False
    if type(schema_model) == JsonSchemaClass:
        anonymised_data = anonymiser_json.anonymise(
            schema_model, ingest, method, manual, default, fields, amount, seed=seed
        )
        json_schema_flag = True
    else:
        anonymised_data = anonymiser.anonymise(
            schema_model, ingest, method, manual, default, fields, amount, seed=seed
        )
    # data returns as a dict of lists of dicts
    # { index: [model, * amount] }

    flush_output = {}
    with open(f"{output}.json", "a") as f:
        for index, content in anonymised_data.items():
            if cout:
                print("-" * 60)
                print(f"Input data:\n\t{ingest[index]}")
                print("Output data:")
            flush_list = []
            for idx, x in enumerate(content):
                if json_schema_flag == True:
                    output_data = x
                else:
                    output_data = x.model_dump()
                if cout:
                    print(
                        f"output {str(idx)}{' ' * (10 - len(str(idx)))}{output_data}"
                    )
                flush_list.append(output_data)
            flush_output[index] = flush_list
        f.write(json.dumps(flush_output, indent=4, default= lambda v: make_json_safe(v)))
