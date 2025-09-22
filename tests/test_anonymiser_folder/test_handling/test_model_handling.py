import pytest
from pydantic import BaseModel

from src.sm.anonymiser.handling.model_handling import (
    subset_model,
    new_model,
)
from src.sm.tools.model_funcs import get_model_fields


class schema_model(BaseModel):
    foo: str
    bar: int
    zar: bool


def test_subset_model():
    field_names = ["bar", "zar"]
    return_model = subset_model(schema_model, field_names)
    names = [name for name in get_model_fields(return_model).keys()]
    assert names == field_names


data = {"foo": "hello", "bar": 10, "zar": False}
def test_new_model():
    field_names = ["bar", "zar"]
    return_model = new_model(data, field_names)
    names = [name for name in get_model_fields(return_model).keys()]
    assert names == field_names
