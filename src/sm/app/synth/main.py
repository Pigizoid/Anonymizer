import typer

from .subcommands.batch import synth_batch_subcommand
from .subcommands.single import synth_single_subcommand

synth_app = typer.Typer()

synth_app.add_typer(synth_batch_subcommand)
synth_app.add_typer(synth_single_subcommand)
