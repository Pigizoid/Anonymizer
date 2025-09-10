import json
from ..helper_funcs import load_ingest_data

from sm.anonymiser.anonymiser import Anonymiser

def anon_func(
    schema_model, seed, method, amount, start_index, ingest, cout, manual, default, fields, output
):
    '''
    Inputs:
        a pydantic schema model
        a generation seed as an int
        a generation method of the methods "mixed","mimesis","faker"
        an amount as an int, to generate per data index
        a starting index (to optionally skip data indexes in the ingest)
        a string path to the ingest file
        a cout boolean toggle for verbose printing
        a manual boolean toggle for automatic/manual processing modes
        a default anonymisation method of the methods "mask","perturb","synth"
        a dict of fields = {field_name:method} of the methods "default","mask","perturb","synth"
        a filename as str for the output file (.json added by default)
    Loads data from ingest file
    Runs the anonymiser tool
    Outputs data to the output file and optionally prints output to the screen
    '''
    data = load_ingest_data(ingest, start_index=start_index)
    # data comes in as a dict of dicts
    anon = Anonymiser()
    anonymised_data = anon.anonymise(
        schema_model, data, method, manual, seed, default, fields, amount
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
                    print(f"output {str(idx)}{' ' * (10 - len(str(idx)))}{x.model_dump()}")
                flush_list.append(x.model_dump())
            flush_output[index] = flush_list
        f.write(json.dumps(flush_output, indent=8))
