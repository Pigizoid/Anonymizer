import typer
from typing import Annotated, Optional
from ..test_funcs import synth_func
from ...test_helper_funcs import return_flags, load_schema, load_file_path, close_folder
from ...test_models import SynthesiserConfig

synth_single_subcommand = typer.Typer()


@synth_single_subcommand.command(name="single")
def synth_single_command(
    ctx: typer.Context,  # contains ctx.config
    method: str = None,
    output: str = None,
    cout: Annotated[Optional[bool], typer.Option("--cout/--no-cout")] = None,
):
    """
    A subcommand for the synth command
    Inputs:
        a method of the methods "mixed","mimesis","faker"
        a filename as str for the output file (.json added by default)
        cout boolean to toggle verbose printing
    Runs the synthesiser tool in single mode, amount=1 batch=1
    """
    ctx.params["amount"] = 1
    ctx.params["batch"] = 1
    flags = return_flags(ctx, SynthesiserConfig)
    print(f"Args: {flags}")
    schema_model = load_schema(flags.schema_path)
    seed = flags.seed
    synth_flags = flags.synth
    output_file_path = load_file_path(synth_flags.output)

    synth_func(
        schema_model,
        seed,
        synth_flags.method,
        synth_flags.amount,
        output_file_path,
        cout=synth_flags.cout,
    )

    close_folder(output_file_path)
