import typer

from .test_subcommands.test_auto import anon_auto_subcommand
from .test_subcommands.test_manual import anon_manual_subcommand

anon_app = typer.Typer()

anon_app.add_typer(anon_manual_subcommand)
anon_app.add_typer(anon_auto_subcommand)
