import json
import time
from ..helper_funcs import send_batch_to_API

from sm.synthesiser.synthesiser import Synthesiser



def synth_func(schema_model, seed, method, amount, output, start_index=0, cout: bool = False):
    start_time = time.time()
    synth = Synthesiser(method=method)
    dataset = synth.synthesise(
        schema_model, method, amount, seed
    )  # returns as [ data, data, ... ]
    elapsed_time = time.time() - start_time  # end timer
    print(f"Generation | Time taken: {elapsed_time:.2f} seconds")
    flush = []
    if output.startswith("http"):
        request_entries = []

    for index, data in enumerate(dataset):
        if output is not None:
            if (index + 1 + start_index) != 1:
                front_string = ",\n	"
            else:
                front_string = ""
            json_str = json.dumps(
                data.model_dump(), indent=8, default=lambda v: repr(v)
            )
            flush.append(f'{front_string}"{index + start_index}": {json_str}')
            if output.startswith("http"):
                request_entries.append(data)
        else:
            flush.append(f"{index + 1 + start_index}: {data}")
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