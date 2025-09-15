import typer

from .test_subcommands.test_batch import synth_batch_subcommand
from .test_subcommands.test_single import synth_single_subcommand

synth_app = typer.Typer()

synth_app.add_typer(synth_batch_subcommand)
synth_app.add_typer(synth_single_subcommand)
