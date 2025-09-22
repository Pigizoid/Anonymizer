import typer
from pathlib import Path
from sm.tools.schema_format_converter import convert_folder_to_JSON

json_subcommand = typer.Typer()

@json_subcommand.command(name="pyd_to_json")
def anon_auto_command(
    ctx: typer.Context,
    ingest: str = None
):
    convert_folder_to_JSON(Path(ingest))
    
    
