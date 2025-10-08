import typer
from typing import Annotated, Optional
from smoke_mirrors.app.anon.funcs import anon_func
from smoke_mirrors.app.helper_funcs import  (
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
from smoke_mirrors.app.models import AnonymiserConfig
from pathlib import Path

anon_auto_subcommand = typer.Typer()

def anon_auto_func(schema_model,output_file_path,ingest,flags):
    seed = flags.seed
    anon_flags = flags.anon
    load_file_path(output_file_path)

    if anon_flags.ingest is None:
        raise Exception("Config 'ingest' required")

    anon_func(
        schema_model=schema_model,
        seed=seed,
        method=anon_flags.method,
        amount=anon_flags.amount,
        start_index=0,
        ingest=ingest,
        stdcout=anon_flags.stdcout,
        manual=anon_flags.manual,
        default=anon_flags.default,
        fields=anon_flags.fields,
        output=output_file_path,
        key_anon=anon_flags.key_anon,
        performance=anon_flags.performance
    )

    close_folder(output_file_path)


@anon_auto_subcommand.command(name="auto")
def anon_auto_command(
    ctx: typer.Context,  # contains ctx.config
    ingest: str = None,
    method: str = None,
    output: str = None,
    amount: int = None,
    stdcout: Annotated[Optional[bool], typer.Option("--stdcout/--no-stdcout")] = None,
    default: str = "mask",
    key_anon: Annotated[Optional[bool], typer.Option("--key-anon/--no-key-anon")] = None,
    performance: Annotated[Optional[bool], typer.Option("--performance/--no-performance")] = None,
):
    """
    A subcommand for the anon command\n
    Inputs:\n
        a path as str to an ingest file
        a method of the methods "mixed","mimesis","faker"
        a filename as str for the output file (.json added by default)
        an amount to generate per data index as an int
        stdcout boolean to toggle verbose printing
        a default anonymisation method of the methods "mask","synth","perturb"
    Runs the anonymisation tool in auto mode\n
    """
    if default not in ["mask", "synth", "perturb"]:
        raise ValueError(f"Default:'{default}' not in {['mask', 'synth', 'perturb']}")
    ctx.params["fields"] = {}
    flags = return_flags(ctx, AnonymiserConfig)
    print(f"Args: {flags.dict(exclude={"synth","anon"})}")
    print(flags.anon)
    if flags.schema_path is None:
        schema_models = None
    else:
        schema_models = load_recursed_path(Path(flags.schema_path),flags.schema_type,load_schema)
        schema_models = flatten_loaded_schemas(schema_models)
    ingests = load_recursed_path(Path(flags.anon.ingest),".json",load_ingest)
    output_path_name = load_output_path_flag(flags.anon.output)
    recursive_ingest_json_handler(ingests,anon_auto_func,flags,output_path_name,schema_models)
