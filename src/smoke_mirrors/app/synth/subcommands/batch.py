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
import json


synth_batch_subcommand = typer.Typer()


def synth_batch_func(schema_model,output_file_path,flags):
    seed = flags.seed
    synth_flags = flags.synth
    load_file_path(output_file_path)

    method = synth_flags.method
    amount = synth_flags.amount
    if synth_flags.batch == 0:
        batch = amount
    else:
        batch = synth_flags.batch
    cout = synth_flags.cout
    performance = synth_flags.performance
    batch_index = 0
    for y in range(amount // batch):
        print(
            f"Batch num: {y + 1} of {amount // batch} | Batch amount: {batch} | Completed: {batch_index}/{amount}  {'-' * 50}"
        )
        synth_func(
            schema_model,
            method,
            batch,
            output_file_path,
            cout=cout,
            start_index=batch_index,
            seed=seed,
            performance=performance,
        )
        batch_index += batch
    if amount - batch_index != 0:
        synth_func(
            schema_model,
            method,
            amount - batch_index,
            output_file_path,
            cout=cout,
            start_index=batch_index,
            seed=seed,
            performance=performance,
        )

    close_folder(output_file_path)


@synth_batch_subcommand.command(name="batch")
def synth_batch_command(
    ctx: typer.Context,  # contains ctx.config
    method: str = None,
    amount: int = None,
    batch: int = None,
    output: str = None,
    flat_output: Annotated[Optional[bool], typer.Option("--flat-output/--no-flat-output")] = None,
    cout: Annotated[Optional[bool], typer.Option("--cout/--no-cout")] = None,
    performance: Annotated[Optional[bool], typer.Option("--performance/--no-performance")] = None,
):
    """
    A subcommand for the synth command\n
    Inputs:\n
        a method of the methods "mixed","mimesis","faker"
        an amount to generate per schema as an int
        a batch amount as an int
            generate "amount" total in "batch" sizes
            e.g. amount=100 batch=50 means 2 batches of 50
        a filename as str for the output file (.json added by default)
        cout boolean to toggle verbose printing
    Runs the synthesiser tool in batch mode\n
    """
    flags = return_flags(ctx, SynthesiserConfig)
    print(f"Args: {flags}")
    schema_models = load_recursed_path(Path(flags.schema_path),flags.schema_type,load_schema)
    if flat_output:
        schema_models = {schema_model.__name__:schema_model for schema_model in flatten_loaded_schemas(schema_models)}
    output_path_name = load_output_path_flag(flags.synth.output)
    recursive_folder_schema_handler(schema_models,synth_batch_func,flags,output_path_name)