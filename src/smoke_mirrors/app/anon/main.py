import typer
from smoke_mirrors.app.anon.subcommands.auto import anon_auto_subcommand
from smoke_mirrors.app.anon.subcommands.manual import anon_manual_subcommand

anon_app = typer.Typer()

anon_app.add_typer(anon_manual_subcommand)
anon_app.add_typer(anon_auto_subcommand)
