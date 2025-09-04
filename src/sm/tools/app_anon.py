# app_anon.py

import typing
from typing_extensions import Annotated

import typer

from .helper_funcs import (
    return_flags,
    load_schema,
    load_file_path,
    anon_func,
    close_folder,
)
from .app_models import AnonymiserConfig
import json

anon_app = typer.Typer()
# -----
# anonymiser commands
# -----


def anon_auto(
    ctx: typer.Context,  # contains ctx.config
    ingest: str = None,
    method: str = None,
    output: str = None,
    amount: int = None,
    cout: Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
):
    ctx.params["start"] = 0
    ctx.params["fields"] = {}
    flags = return_flags(ctx, AnonymiserConfig)
    print(f"Args: {flags}")
    schema_model = load_schema(flags.schema_path)
    anon_flags = flags.anon
    output_file_path = load_file_path(anon_flags.output)

    if anon_flags.ingest is None:
        raise Exception("Config 'ingest' required")

    anon_func(
        schema_model,
        anon_flags.method,
        anon_flags.amount,
        anon_flags.start,
        anon_flags.ingest,
        anon_flags.cout,
        anon_flags.manual,
        anon_flags.fields,
        output_file_path,
    )

    close_folder(output_file_path)


@anon_app.callback(invoke_without_command=True)  # allows default to anon_auto
def anon_default(
    ctx: typer.Context,  # contains ctx.config
    ingest: str = None,
    method: str = None,
    output: str = None,
    amount: int = None,
    cout: Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
):
    if (
        ctx.invoked_subcommand is not None
    ):  # this allows a default to run code and, not run when a sub command is invoked
        return
    anon_auto(ctx, *ctx.params)


@anon_app.command(name="auto")  # call with "auto" sub command
def anon_auto_command(
    ctx: typer.Context,  # contains ctx.config
    ingest: str = None,
    method: str = None,
    output: str = None,
    amount: int = None,
    cout: Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
):
    anon_auto(ctx, *ctx.params)


@anon_app.command(name="manual")  # call with "manual" sub command
def anon_manual_command(
    ctx: typer.Context,  # contains ctx.config and ctx.params  (params are the below field
    ingest: str = None,
    method: str = None,
    amount: int = None,
    start: int = None,
    output: str = None,
    cout: Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
    fields: str = typer.Option(None, help="Fields as JSON string"),
):
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
    anon_flags = flags.anon
    output_file_path = load_file_path(anon_flags.output)

    if anon_flags.ingest is None:
        raise Exception("Config 'ingest' required")

    anon_func(
        schema_model,
        anon_flags.method,
        anon_flags.amount,
        anon_flags.start,
        anon_flags.ingest,
        anon_flags.cout,
        True,  # manual
        anon_flags.fields,
        output_file_path,
    )

    close_folder(output_file_path)
