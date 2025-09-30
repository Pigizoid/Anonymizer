# app_models.py
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import typing
from pathlib import Path


class SynthesiserConfig(BaseModel):
    method: str = "mixed"
    amount: int = 1
    batch: int = 0
    output: str = ""
    cout: bool = False


class AnonymiserConfig(BaseModel):
    ingest: str = ""
    method: str = "mixed"
    amount: int = 1
    start: int = 0
    output: str = ""
    cout: bool = False
    manual: bool = False
    default: typing.Literal["mask", "synth", "perturb"] = "mask"
    fields: typing.Dict[str, str] = {}
    key_anon: bool = False


class Settings(BaseSettings):
    schema_path: typing.Optional[Path]
    schema_type: str
    seed: typing.Union[int, str, bool, None] = False
    synth: typing.Optional[SynthesiserConfig]
    anon: typing.Optional[AnonymiserConfig]
