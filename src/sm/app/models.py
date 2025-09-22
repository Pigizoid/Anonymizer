# app_models.py
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import typing


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


class Settings(BaseSettings):
    schema_path: typing.Optional[str]
    seed: typing.Union[int, str, bool, None] = False
    synth: typing.Optional[SynthesiserConfig]
    anon: typing.Optional[AnonymiserConfig]
