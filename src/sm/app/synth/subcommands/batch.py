import typer
from typing import Annotated, Optional
from sm.app.synth.funcs import synth_func
from sm.app.helper_funcs import return_flags, load_schema, load_file_path, close_folder
from sm.app.models import SynthesiserConfig


synth_batch_subcommand = typer.Typer()


@synth_batch_subcommand.command(name="batch")
def synth_batch_command(
    ctx: typer.Context,  # contains ctx.config
    method: str = None,
    amount: int = None,
    batch: int = None,
    output: str = None,
    cout: Annotated[Optional[bool], typer.Option("--cout/--no-cout")] = None,
):
    """
    A subcommand for the synth command
    Inputs:
        a method of the methods "mixed","mimesis","faker"
        an amount to generate per schema as an int
        a batch amount as an int
            generate "amount" total in "batch" sizes
            e.g. amount=100 batch=50 means 2 batches of 50
        a filename as str for the output file (.json added by default)
        cout boolean to toggle verbose printing
    Runs the synthesiser tool in batch mode
    """
    flags = return_flags(ctx, SynthesiserConfig)
    print(f"Args: {flags}")
    schema_model = load_schema(flags.schema_path)
    seed = flags.seed
    synth_flags = flags.synth
    output_file_path = load_file_path(synth_flags.output)

    method = synth_flags.method
    amount = synth_flags.amount
    if synth_flags.batch == 0:
        batch = amount
    else:
        batch = synth_flags.batch
    cout = synth_flags.cout
    batch_index = 0
    for y in range(amount // batch):
        print(
            f"Batch num: {y + 1} of {amount // batch} | Batch amount: {batch} | Completed: {batch_index}/{amount}  {'-' * 50}"
        )
        synth_func(
            schema_model,
            seed,
            method,
            batch,
            output_file_path,
            start_index=batch_index,
            cout=cout,
        )
        batch_index += batch
    if amount - batch_index != 0:
        synth_func(
            schema_model,
            seed,
            method,
            amount - batch_index,
            output_file_path,
            start_index=batch_index,
            cout=cout,
        )

    close_folder(output_file_path)
