import pytest
from pydantic import BaseModel

from src.sm.anonymiser.handling.model_handling import subset_model,guess_type,new_model
from src.sm.tools.model_funcs import get_model_fields

class schema_model(BaseModel):
    foo : str
    bar : int
    zar : bool
def test_subset_model():
    field_names = ["bar","zar"]
    return_model = subset_model(schema_model,field_names)
    names = [name for name in get_model_fields(return_model).keys()]
    assert names == field_names

values = [
    None,
    True,
    10,
    1.5,
    5+5j,
    b'hello',
    ("a","b"),
    ["a","b"],
    set(["a","b"]),
    {"a":"b"},
    "hello"
    ]
@pytest.mark.parametrize("value,expected",[(str(value),type(value)) for value in values ])
def test_guess_type(value,expected):
    assert guess_type(value) == expected

data = {
    "foo":"hello",
    "bar":10,
    "zar":False
}
def test_new_model():
    field_names = ["bar","zar"]
    return_model = new_model(data,field_names)
    names = [name for name in get_model_fields(return_model).keys()]
    assert names == field_names
