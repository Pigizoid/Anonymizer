import typer
from smoke_mirrors.app.extras.subcommands.json_converter import json_subcommand

json_app = typer.Typer()

json_app.add_typer(json_subcommand)
