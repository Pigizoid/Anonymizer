from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
from pathlib import Path
from smoke_mirrors.app.synth.main import synth_app
from smoke_mirrors.app.anon.main import anon_app
from smoke_mirrors.app.extras.main import json_app
from smoke_mirrors.app.models import SynthesiserConfig, AnonymiserConfig, Settings
from smoke_mirrors.app.helper_funcs import windows_path_to_pathlib
from collections.abc import Mapping
import typer
import yaml

app = typer.Typer()

app.add_typer(synth_app, name="synth")
app.add_typer(anon_app, name="anon")
app.add_typer(json_app, name="extras")


def make_settings_class(config_path: Optional[Path]) -> BaseSettings:
    """
    Inputs:\n
        an optional config path string to the config.yaml file
    Ouputs:\n
        a BaseSettings class type, unprocessed with source functionality modified
    """

    def yaml_settings_source() -> Dict[str, Any]:
        if config_path is None:
            return {}

        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text()) or {}
        else:
            raise Exception(f"Yaml doesnt exist: {config_path}")

        mapping: Dict[str, Any] = {}
        if "schema" in raw:
            mapping["schema_path"] = raw["schema"]
        elif "schema_path" in raw:
            mapping["schema_path"] = raw["schema_path"]
        
        if "schema_type" in raw:
            mapping["schema_type"] = raw["schema_type"]

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
        synth_defaults = {
            name: field.default
            for name, field in SynthesiserConfig.model_fields.items()
        }
        anon_defaults = {
            name: field.default for name, field in AnonymiserConfig.model_fields.items()
        }
        defaults = {
            "schema_path": None,
            "schema_type": None,
            "synth": synth_defaults,
            "anon": anon_defaults,
        }
        return defaults

    def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """Standard deep dict merge: returns a new dict with b merged into a."""
        out = dict(a)
        for k, v in (b or {}).items():
            if v is None:
                # Skip None in b, keep original value in a if it exists
                if k not in out:
                    out[k] = None
            elif k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
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
            if data == {}:
                continue
            anon = data.get("anon")
            if isinstance(anon, Mapping) and "fields" in anon:
                last_anon_fields = anon["fields"]  #keep track of latest field dict to prevent dict update extending
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
        """the builtin settings customiser for source input organising for pydantic"""

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
    config: Optional[Path] = typer.Option(
        None
    ),
    schema_path: Optional[Path] = typer.Option(
        None
    ),
    schema_type: Optional[str] = typer.Option(
        None
    ),
    seed: Optional[str] = typer.Option(
        False
    ),
):
    """
    the main command run at top level (used for allowing callback methods) -> loading a config arg at top level
    examples:
        sm --config config.yaml anon manual
        sm --config config.yaml anon auto
        sm --config config.yaml synth single
        sm --config config.yaml synth batch
    """
    if config is not None:
        config_path = windows_path_to_pathlib(config)
        if not config_path.exists():
            raise FileExistsError(f"File config '{config_path}' does not exist")
    else:
        config_path = None
    if schema_path is not None:
        schema_path = windows_path_to_pathlib(schema_path)
        if not schema_path.exists():
            raise FileExistsError(f"File schema '{schema_path}' does not exist")
    else:
        schema_path = None
    print("config file:",config_path)
    Settings = make_settings_class(config_path)
    ctx.obj = {"settings": Settings, "schema_path": schema_path, "schema_type": schema_type, "seed": seed}
    #fix settings to allow for schema type of either py or json and then make a new json_sytnehsiser
    #additionally check if loading json breaks anything before passing to the synth_func
