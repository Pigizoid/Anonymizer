#app_synth.py

import typing
from typing_extensions import Annotated

import typer

from .helper_funcs import *
from .app_models import SynthesiserConfig

synth_app = typer.Typer()   

# -----
# synthesiser commands
# -----
@synth_app.callback(invoke_without_command=True)  #call default with no subcommand
def synth_single(
    ctx: typer.Context,  #contains ctx.config
    method: str = None,
    output: str = None,
    cout: Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
):  
    if ctx.invoked_subcommand is not None: #this allows a default without running extra code
        return
    ctx.params["amount"] = 1
    ctx.params["batch"] = 1
    flags = return_flags(ctx, SynthesiserConfig)
    print(f"Args: {flags}")
    schema_model = load_schema(flags.schema_path)
    synth_flags = flags.synth
    output_file_path = load_file_path(synth_flags.output)
    
    synth_func(schema_model, synth_flags.method, synth_flags.amount, output_file_path, cout=synth_flags.cout)
    
    close_folder(output_file_path)

@synth_app.command(name="batch")  #call with "batch" sub command
def synth_batch(
    ctx: typer.Context,  #contains ctx.config
    method: str = None,
    amount: int = None,
    batch: int = None,
    output: str = None,
    cout: Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
):
    flags = return_flags(ctx, SynthesiserConfig)
    print(f"Args: {flags}")
    schema_model = load_schema(flags.schema_path)
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
        print("Batch: ", y + 1)
        synth_func(
            schema_model,
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
            method,
            amount - batch_index,
            output_file_path,
            start_index=batch_index,
            cout=cout,
        )
    
    close_folder(output_file_path)
