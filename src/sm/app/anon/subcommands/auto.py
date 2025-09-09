import typer
from typing import Annotated, Optional
from ..funcs import anon_func
from ...helper_funcs import return_flags, load_schema, load_file_path, close_folder
from ...models import AnonymiserConfig

anon_auto_subcommand = typer.Typer()

@anon_auto_subcommand.command(name="auto")  # call with "auto" sub command
def anon_auto_command(
    ctx: typer.Context,  # contains ctx.config
    ingest: str = None,
    method: str = None,
    output: str = None,
    amount: int = None,
    cout: Annotated[Optional[bool], typer.Option("--cout/--no-cout")] = None,
    default: str = "mask"
):
    ctx.params["start"] = 0
    ctx.params["fields"] = {}
    flags = return_flags(ctx, AnonymiserConfig)
    print(f"Args: {flags}")
    schema_model = load_schema(flags.schema_path)
    seed = flags.seed
    anon_flags = flags.anon
    output_file_path = load_file_path(anon_flags.output)

    if anon_flags.ingest is None:
        raise Exception("Config 'ingest' required")

    anon_func(
        schema_model,
        seed,
        anon_flags.method,
        anon_flags.amount,
        anon_flags.start,
        anon_flags.ingest,
        anon_flags.cout,
        anon_flags.manual,
        anon_flags.default,
        anon_flags.fields,
        output_file_path,
    )

    close_folder(output_file_path)
