import json
from ..helper_funcs import load_ingest_data

from sm.anonymiser.anonymiser import Anonymiser

def anon_func(
    schema_model, seed, method, amount, index, ingest, cout, manual, default, fields, output
):
    data = load_ingest_data(ingest, index=index)
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
            print("-" * 60)
            print(f"Input data:\n\t{data[index]}")
            flush_list = []
            print("Output data:")
            for idx, x in enumerate(content):
                if cout:
                    print(f"output {str(idx)}{' ' * (10 - len(str(idx)))}{x.model_dump()}")
                flush_list.append(x.model_dump())
            flush_output[index] = flush_list
        f.write(json.dumps(flush_output, indent=8))
