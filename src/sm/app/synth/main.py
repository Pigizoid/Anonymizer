import typer
from sm.app.synth.subcommands.batch import synth_batch_subcommand
from sm.app.synth.subcommands.single import synth_single_subcommand

synth_app = typer.Typer()

synth_app.add_typer(synth_batch_subcommand)
synth_app.add_typer(synth_single_subcommand)
