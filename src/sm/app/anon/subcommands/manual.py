import typer
from typing import Annotated, Optional
from ..funcs import anon_func
from ...helper_funcs import return_flags, load_schema, load_file_path, close_folder
from ...models import AnonymiserConfig

import json

anon_manual_subcommand = typer.Typer()



@anon_manual_subcommand.command(name="manual")  # call with "manual" sub command
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
    if default not in ["mask","synth","perturb"]:
        raise ValueError(f"Default:'{default}' not in {["mask","synth","perturb"]}")
    if fields:
        try:
            ctx.params["fields"] = json.loads(fields)
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"Invalid JSON for --fields: {e}")
    else:
        ctx.params["fields"] = None

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
        True,  # manual
        anon_flags.default,
        anon_flags.fields,
        output_file_path,
    )

    close_folder(output_file_path)
