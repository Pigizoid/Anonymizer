import pytest
from pydantic import BaseModel
from src.sm.synthesiser.synthesiser import Synthesiser
from src.sm.anonymiser.anonymiser import anonymise

class schema_model(BaseModel):
    foo: str
    bar: int
    zar: bool


vals = [True, 1.5, 15, b"hello", "hello", []]
methods = ["faker", "mimesis", "mixed"]
synth = Synthesiser()
schema_input_data = {"foo": "hello", "bar": 10, "zar": True}
input_data = {"0":schema_input_data}
manuals = [True, False]
defaults = ["mask","perturb","synth"]
field_sets = [
    {"foo":"default"},
    {"bar":"default"},
    {"zar":"default"},
    {"foo":"default", "bar":"default"},
    {"bar":"default", "zar":"default"},
    {"foo":"default", "bar":"default", "zar":"default"},
]
amounts = [1, 5, 10]

input_sets = []
for method in methods:
    for manual in manuals:
        for default in defaults:
            for fields in field_sets:
                for amount in amounts:
                    input_sets.append((schema_model,input_data,method,manual,default,fields,amount,0))
@pytest.mark.parametrize("schema_model, data, method, manual, default, fields, amount, seed",input_sets)
def test_anonymise(schema_model, data, method, manual, default, fields, amount, seed):
    print(fields)
    return_data = anonymise(schema_model, data, method, manual, default, fields, amount, seed=seed)
    assert isinstance(return_data,dict)
    
