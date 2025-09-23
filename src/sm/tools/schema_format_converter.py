import json_schema_to_pydantic
from pydantic import BaseModel
import json
from pathlib import Path
from sm.app.helper_funcs import load_recursed_path, load_schema, recursive_folder_schema_handler


def convert_to_JSON(schema_model:BaseModel):
    if isinstance(schema_model, type) and issubclass(schema_model, BaseModel):
        JSON_schema = schema_model.model_json_schema()
    else:
        raise Exception("Unhandled schema type")
    return JSON_schema

def convert_to_pydantic(JSON_schema):
    pydantic_model = json_schema_to_pydantic.create_model(JSON_schema)
    return pydantic_model

def write_JSON_schema(schema_model:BaseModel,write_path:Path,flags=None):
    JSON_output = convert_to_JSON(schema_model)
    with open(f"{write_path}.json","w+") as f:
        output_str = json.dumps(JSON_output, indent=8, default=lambda v: str(v))
        f.write(output_str)

def convert_folder_to_JSON(folder_path:Path):
    schema_models = load_recursed_path(folder_path,".py",load_schema)
    recursive_folder_schema_handler(schema_models,write_JSON_schema,[],folder_path)

