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
    flatten_loaded_schemas,
)
from smoke_mirrors.app.models import SynthesiserConfig
from pathlib import Path


synth_batch_subcommand = typer.Typer()


def synth_batch_func(schema_model, output_file_path, flags):
    seed = flags.seed
    synth_flags = flags.synth
    load_file_path(output_file_path)

    method = synth_flags.method
    amount = synth_flags.amount
    if synth_flags.batch == 0:
        batch = amount
    else:
        batch = synth_flags.batch
    stdcout = synth_flags.stdcout
    performance = synth_flags.performance
    realistic = synth_flags.realistic
    batch_index = 0
    aofb = +amount // batch
    totam = aofb + (amount % batch != 0)
    for y in range(amount // batch):
        print(
            f"Batch num: {y + 1} of {totam} | Batch amount: {batch} | Total: {amount}"
        )
        synth_func(
            schema_model,
            method,
            batch,
            output_file_path,
            stdcout=stdcout,
            start_index=batch_index,
            seed=seed,
            performance=performance,
            realistic=realistic,
        )
        batch_index += batch
    if amount - batch_index != 0:
        print(
            f"Batch num: {totam} of {totam} | Batch amount: {amount - batch_index} | Total: {amount}"
        )
        synth_func(
            schema_model,
            method,
            amount - batch_index,
            output_file_path,
            stdcout=stdcout,
            start_index=batch_index,
            seed=seed,
            performance=performance,
            realistic=realistic,
        )

    close_folder(output_file_path)


@synth_batch_subcommand.command(name="batch")
def synth_batch_command(
    ctx: typer.Context,  # contains ctx.config
    method: str = None,
    amount: int = None,
    batch: int = None,
    output: Path = None,
    flat_output: Annotated[
        Optional[bool], typer.Option("--flat-output/--no-flat-output")
    ] = None,
    stdcout: Annotated[Optional[bool], typer.Option("--stdcout/--no-stdcout")] = None,
    performance: Annotated[
        Optional[bool], typer.Option("--performance/--no-performance")
    ] = None,
    realistic: Annotated[
        Optional[bool], typer.Option("--realistic/--no-realistic")
    ] = None,
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
        stdcout boolean to toggle verbose printing
    Runs the synthesiser tool in batch mode\n
    """
    flags = return_flags(ctx, SynthesiserConfig)
    print(f"Args: {flags.model_dump(exclude={'synth', 'anon'})}")
    print(flags.synth)
    schema_models = load_recursed_path(
        flags.schema_path, flags.schema_type, load_schema
    )
    if schema_models is None:
        raise Exception("No schemas loaded from input")
    if flat_output:
        schema_models = {
            schema_model.__name__: schema_model
            for schema_model in flatten_loaded_schemas(schema_models)
        }
    output_path_name = load_output_path_flag(flags.synth.output)
    recursive_folder_schema_handler(
        schema_models, synth_batch_func, flags, output_path_name
    )
