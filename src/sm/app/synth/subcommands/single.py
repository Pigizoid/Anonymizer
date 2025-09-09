import typer
from typing import Annotated, Optional
from ..funcs import synth_func
from ...helper_funcs import return_flags, load_schema, load_file_path, close_folder
from ...models import SynthesiserConfig

synth_single_subcommand = typer.Typer()



@synth_single_subcommand.command(name="single")  # call default with no subcommand
def synth_single_command(
    ctx: typer.Context,  # contains ctx.config
    method: str = None,
    output: str = None,
    cout: Annotated[Optional[bool], typer.Option("--cout/--no-cout")] = None,
):
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
