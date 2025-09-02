#UTILS_models.py
from pydantic import BaseModel, Field
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
    fields: typing.Dict[str, str] = Field(default={})