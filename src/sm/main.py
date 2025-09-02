#UTILS_main.py




from pydantic_settings import BaseSettings
import typing
from typing import Optional, Dict, Any
from pathlib import Path

import typer


import yaml


app = typer.Typer()

from .tools.app_synth import synth_app
from .tools.app_anon import anon_app
from .tools.helper_funcs import *
from .tools.app_models import SynthesiserConfig, AnonymiserConfig

app.add_typer(synth_app, name="synth")
app.add_typer(anon_app, name="anon")






def make_settings_class(config_path: Optional[str]) -> type[BaseSettings]:
    
    def yaml_settings_source() -> Dict[str, Any]:
        
        if not Path(config_path).exists():
            return {}  #returning {} as empty to allow defaults to parse

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
        
        synth_defaults = {name: field.default for name, field in SynthesiserConfig.model_fields.items()}
        anon_defaults = {name: field.default for name, field in AnonymiserConfig.model_fields.items()}
        defaults = {
            "schema_path": "schema.py",
            "synth": synth_defaults,
            "anon": anon_defaults
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
    
    class Settings(BaseSettings):
        schema_path: str
        synth: typing.Optional[SynthesiserConfig]
        anon: typing.Optional[AnonymiserConfig]
    
    def _settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        def build_source():
            return composite_source([
                init_settings,          # highest priority: values passed into the constructor
                yaml_settings_source,   # next priority: values from YAML file
                schema_defaults_source, # then schema defaults loaded from schema.py
                env_settings,
                dotenv_settings,
                file_secret_settings,
            ])
        return (build_source,)
    
    Settings.settings_customise_sources = classmethod(_settings_customise_sources)
    return Settings

# -----
# main app
# -----
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Optional[str] = typer.Option(
        "config.yaml",  #default
        exists=False,         #dont check if path exists before allowing it as option
        file_okay=True,
        dir_okay=False,       #these 3 check its readable and a file
        readable=True,
        help="Path to YAML config file",
    ),
    schema_path: Optional[str] = typer.Option(
        "schema.py",
        exists=False,
    )
):
    Settings = make_settings_class(config)
    ctx.obj = {
        "settings": Settings,
        "schema_path": schema_path,
    }





'''
config = "config.yaml"
Settings = make_settings_class(config)
print(Settings)

flags = {
    "method": None,
    "amount": 10000,
    "batch": 50,
    "output": "outputs.json",
    "cout": False
    }
flags = {key:param for key,param in flags.items() if param is not None}
synth = SynthesiserConfig(**flags)

print(Settings(synth=synth))
'''

'''





sm             

->  synthesise -> single
    -> batch
    
->  anonymise -> auto
    -> manual


sm synthesise *args (default is to use the "single" flag)
sm synthesise batch *args
sm anonymise *args (default is to use the "auto" flag)
sm anonymise manual *args



'''














