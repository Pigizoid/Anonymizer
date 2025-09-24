import typer
from pathlib import Path
from sm.app.helper_funcs import load_recursed_path, load_schema, recursive_folder_schema_handler
from sm.tools.schema_format_converter import write_JSON_schema

json_subcommand = typer.Typer()


def convert_folder_to_JSON(folder_path:Path):
    schema_models = load_recursed_path(folder_path,".py",load_schema)
    print(schema_models)
    schema_models = {f"{folder_path.stem}_(json)":schema_models[folder_path.stem]}
    print(schema_models)
    new_path = folder_path.resolve().parent
    recursive_folder_schema_handler(schema_models,write_JSON_schema,[],new_path,use_parent=True)


@json_subcommand.command(name="pyd_to_json")
def anon_auto_command(
    ctx: typer.Context,
    ingest: str = None
):
    convert_folder_to_JSON(Path(ingest))
    
    
