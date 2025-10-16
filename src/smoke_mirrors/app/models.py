# app_models.py
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import typing
from pathlib import Path


class SynthesiserConfig(BaseModel):
    method: str = "mixed"
    amount: int = 1
    batch: int = 0
    output: Path = ""
    flat_output: bool = False
    stdcout: bool = False
    performance: bool = False
    realistic: bool = True


class AnonymiserConfig(BaseModel):
    ingest: Path = ""
    method: str = "mixed"
    amount: int = 1
    output: Path = ""
    stdcout: bool = False
    manual: bool = False
    default: typing.Literal["mask", "synth", "perturb"] = "mask"
    fields: typing.Dict[str, str] = {}
    key_anon: bool = False
    performance: bool = False
    realistic: bool = True


class Settings(BaseSettings):
    schema_path: typing.Optional[Path]
    schema_type: typing.Optional[str]
    seed: typing.Union[int, str, bool, None] = False
    synth: typing.Optional[SynthesiserConfig]
    anon: typing.Optional[AnonymiserConfig]
