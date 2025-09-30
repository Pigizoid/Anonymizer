from pydantic import BaseModel
import json
from pathlib import Path



def convert_to_JSON(schema_model:BaseModel):
    if isinstance(schema_model, type) and issubclass(schema_model, BaseModel):
        JSON_schema = schema_model.model_json_schema()
    else:
        raise Exception("Unhandled schema type")
    return JSON_schema


def write_JSON_schema(schema_model:BaseModel,write_path:Path,flags=None):
    JSON_output = convert_to_JSON(schema_model)
    with open(f"{write_path}.json","w+") as f:
        output_str = json.dumps(JSON_output, indent=8, default=lambda v: str(v))
        f.write(output_str)

