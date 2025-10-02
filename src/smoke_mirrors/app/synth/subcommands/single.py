import typer
from typing import Annotated, Optional
from smoke_mirrors.app.synth.funcs import synth_func
from smoke_mirrors.app.helper_funcs import (
    return_flags, 
    load_recursed_path, 
    load_schema, 
    load_file_path, 
    close_folder, 
    recursive_folder_schema_handler, 
    load_output_path_flag,
    flatten_loaded_schemas
)
from smoke_mirrors.app.models import SynthesiserConfig
from pathlib import Path

synth_single_subcommand = typer.Typer()


def synth_single_func(schema_model,output_file_path,flags):
    seed = flags.seed
    synth_flags = flags.synth
    load_file_path(output_file_path)

    synth_func(
        schema_model,
        synth_flags.method,
        synth_flags.amount,
        output_file_path,
        cout=synth_flags.cout,
        seed=seed,
    )

    close_folder(output_file_path)

@synth_single_subcommand.command(name="single")
def synth_single_command(
    ctx: typer.Context,  # contains ctx.config
    method: str = None,
    output: str = None,
    flat_output: Annotated[Optional[bool], typer.Option("--flat-output/--no-flat-output")] = None,
    cout: Annotated[Optional[bool], typer.Option("--cout/--no-cout")] = None,
):
    """
    A subcommand for the synth command\n
    Inputs:\n
        a method of the methods "mixed","mimesis","faker"
        a filename as str for the output file (.json added by default)
        cout boolean to toggle verbose printing
    Runs the synthesiser tool in single mode, amount=1 batch=1\n
    """
    ctx.params["amount"] = 1
    ctx.params["batch"] = 1
    flags = return_flags(ctx, SynthesiserConfig)
    print(f"Args: {flags}")
    schema_models = load_recursed_path(Path(flags.schema_path),flags.schema_type,load_schema)
    if flat_output:
        schema_models = {schema_model.__name__:schema_model for schema_model in flatten_loaded_schemas(schema_models)}
    output_path_name = load_output_path_flag(flags.synth.output)
    recursive_folder_schema_handler(schema_models,synth_single_func,flags,output_path_name)
