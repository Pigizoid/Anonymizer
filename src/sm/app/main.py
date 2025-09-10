from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
from pathlib import Path
from .synth.main import synth_app
from .anon.main import anon_app
from .models import SynthesiserConfig, AnonymiserConfig, Settings
import typer
import yaml


app = typer.Typer()


app.add_typer(synth_app, name="synth")
app.add_typer(anon_app, name="anon")



def make_settings_class(config_path: Optional[str]) -> type[BaseSettings]:
    '''
    Inputs:
        an optional config path string to the config.yaml file
    Ouputs:
        a BaseSettings class type, unprocessed with source functionality modified
    '''
    def yaml_settings_source() -> Dict[str, Any]:
        '''loads a config.yaml file and gets the data and outputs it as a dict'''
        if not Path(config_path).exists():
            return {}  # returning {} as empty to allow defaults to parse

        try:
            raw = yaml.safe_load(Path(config_path).read_text()) or {}
        except Exception as e:
            print(f"Yaml doesnt exist: {e}")
            return {}

        mapping: Dict[str, Any] = {}
        if "schema" in raw:
            mapping["schema_path"] = raw["schema"]
        elif "schema_path" in raw:
            mapping["schema_path"] = raw["schema_path"]

        if "synthesiser" in raw:
            mapping["synth"] = raw["synthesiser"]
        elif "synth" in raw:
            mapping["synth"] = raw["synth"]

        if "anonymiser" in raw:
            mapping["anon"] = raw["anonymiser"]
        elif "anonymizer" in raw:
            mapping["anon"] = raw["anonymizer"]
        elif "anon" in raw:
            mapping["anon"] = raw["anon"]

        return mapping

    def schema_defaults_source() -> Dict[str, Any]:
        '''outputs the defaults for the anon and synth schemas'''
        synth_defaults = {
            name: field.default
            for name, field in SynthesiserConfig.model_fields.items()
        }
        anon_defaults = {
            name: field.default for name, field in AnonymiserConfig.model_fields.items()
        }
        defaults = {
            "schema_path": "schema.py",
            "synth": synth_defaults,
            "anon": anon_defaults,
        }
        return defaults

    from collections.abc import Mapping

    def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """Standard deep dict merge: returns a new dict with b merged into a."""
        out = dict(a)
        for k, v in (b or {}).items():
            if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    def composite_source(sources):
        """
        this exists solely to merge input sources because pydantic merges dicts from source without allowing an option to overwirte the dict
        """
        order_low_to_high = reversed(sources)

        result: Dict[str, Any] = {}
        last_anon_fields = None

        for src in order_low_to_high:
            data = src() if callable(src) else (src or {})
            anon = data.get("anon")
            if isinstance(anon, Mapping) and "fields" in anon:
                last_anon_fields = anon["fields"]
            result = deep_merge(result, data)

        if last_anon_fields is not None:
            if "anon" not in result or not isinstance(result["anon"], Mapping):
                result["anon"] = {}
            result["anon"]["fields"] = last_anon_fields

        return result

    def _settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        '''the builtin settings customiser for source input organising for pydantic'''
        def build_source():
            return composite_source(
                [
                    init_settings,  # highest priority: values passed into the constructor
                    yaml_settings_source,  # next priority: values from YAML file
                    schema_defaults_source,  # then schema defaults loaded from schema.py
                    env_settings,
                    dotenv_settings,
                    file_secret_settings,
                ]
            )

        return (build_source,)

    Settings.settings_customise_sources = classmethod(_settings_customise_sources)
    return Settings


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Optional[str] = typer.Option(
        "config.yaml",  # default
        exists=False,  # dont check if path exists before allowing it as option
        file_okay=True,
        dir_okay=False,  # these 3 check its readable and a file
        readable=True,
        help="Path to YAML config file (must be called at top level)",
    ),
    schema_path: Optional[str] = typer.Option(
        "schema.py",
        exists=False,
    ),
    seed: Optional[str] = typer.Option(
        False,
        exists=False,
    ),
):
    '''
    the main command run at top level (used for allowing callback methods) -> loading a config arg at top level

    '''
    Settings = make_settings_class(config)
    ctx.obj = {
        "settings": Settings,
        "schema_path": schema_path,
        "seed": seed
    }

