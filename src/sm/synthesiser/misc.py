from typing import List
import re

from sm.tools.model_funcs import get_model_fields
    
def print_path(generate_path, elapsed_time):
    pooling_depth = len(list(map(int, re.findall(r"\[(\d+)\]", generate_path))))
    time_str = f"{elapsed_time:.2f}"
    print(
        f"Time taken: {time_str} seconds",
        " " * (10 - max(10, len(time_str))),
        "    " * (pooling_depth - 1),
        generate_path,
    )