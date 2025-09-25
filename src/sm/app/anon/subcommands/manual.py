import typer
from typing import Annotated, Optional
from sm.app.anon.funcs import anon_func
from sm.app.helper_funcs import  (
    return_flags, 
    load_recursed_path, 
    load_schema, 
    load_ingest, 
    load_file_path, 
    close_folder, 
    recursive_ingest_json_handler, 
    flatten_loaded_schemas,
    load_output_path_flag
)
from sm.app.models import AnonymiserConfig,Settings
from pathlib import Path

import json

anon_manual_subcommand = typer.Typer()

def anon_manual_func(schema_model,output_file_path,ingest,flags):
    seed = flags.seed
    anon_flags = flags.anon
    load_file_path(anon_flags.output)

    if anon_flags.ingest is None:
        raise Exception("Config 'ingest' required")

    anon_func(
        schema_model,
        seed,
        anon_flags.method,
        anon_flags.amount,
        anon_flags.start,
        ingest,
        anon_flags.cout,
        True,  # manual
        anon_flags.default,
        anon_flags.fields,
        output_file_path,
    )

    close_folder(output_file_path)

@anon_manual_subcommand.command(name="manual")
def anon_manual_command(
    ctx: typer.Context,  # contains ctx.config and ctx.params  (params are the below field
    ingest: str = None,
    method: str = None,
    amount: int = None,
    start: int = None,
    output: str = None,
    cout: Annotated[Optional[bool], typer.Option("--cout/--no-cout")] = None,
    default: Optional[str] = "mask",
    fields: str = typer.Option(None, help="Fields as JSON string"),
):
    """
    A subcommand for the anon command\n
    Inputs:\n
        a path as str to an ingest file
        a method of the methods "mixed","mimesis","faker"
        an amount to generate per data index as an int
        a starting index (to optionally skip data indexes in the ingest)
        a filename as str for the output file (.json added by default)
        cout boolean to toggle verbose printing
        a default anonymisation method of the methods "mask","synth","perturb"
        a json string of type dict = {field_name:method} of the methods "default","mask","synth","perturb":
            json string key and value requires double quotes
            example:  sm --config config.yaml anon manual --fields '{"name":"mask"}'
    Runs the anonymisation tool in manual mode\n
    """
    if default not in ["mask", "synth", "perturb"]:
        raise ValueError(f"Default:'{default}' not in {['mask', 'synth', 'perturb']}")
    if fields:
        try:
            ctx.params["fields"] = json.loads(fields)
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"Invalid JSON for --fields: {e}")
    else:
        ctx.params["fields"] = None

    flags = return_flags(ctx, AnonymiserConfig)
    print(f"Args: {flags}")
    schema_models = load_recursed_path(Path(flags.schema_path),flags.schema_type,load_schema)
    schema_models = flatten_loaded_schemas(schema_models)
    ingests = load_recursed_path(Path(flags.anon.ingest),".json",load_ingest)
    output_path_name = load_output_path_flag(flags.anon.output)
    recursive_ingest_json_handler(ingests,anon_manual_func,flags,output_path_name,schema_models)
