# app_models.py
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import typing


class SynthesiserConfig(BaseModel):
    method: str = Field(default="mixed")
    amount: int = Field(default=1)
    batch: int = Field(default=0)
    output: str = Field(default="")
    cout: bool = Field(default=False)


class AnonymiserConfig(BaseModel):
    ingest: str = Field(default="")
    method: str = Field(default="mixed")
    amount: int = Field(default=1)
    start: int = Field(default=0)
    output: str = Field(default="")
    cout: bool = Field(default=False)
    manual: bool = Field(default=False)
    default: typing.Literal["mask","synth","perturb"] = Field(default="mask")
    fields: typing.Dict[str, str] = Field(default={})



class Settings(BaseSettings):
    schema_path: str
    seed: typing.Union[int,str,bool,None] = Field(default=False)
    synth: typing.Optional[SynthesiserConfig]
    anon: typing.Optional[AnonymiserConfig]
