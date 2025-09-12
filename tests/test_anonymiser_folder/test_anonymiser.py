import pytest
from pydantic import BaseModel
from src.sm.synthesiser.synthesiser import Synthesiser
from src.sm.anonymiser.anonymiser import Anonymiser
anon = Anonymiser()


class schema_model(BaseModel):
    foo : str
    bar : int
    zar : bool
vals = [True,1.5,15,b'hello',"hello",[]]
methods = ["mask","synth","perturb"]
synth = Synthesiser()
input_data = { f"field_{val}":val for val in vals }
recursive_input_data = {
    "list": [x for x in range(10)],
    "tuple": (x for x in range(10)),
    "set": set(x for x in range(10)),
    "dict": {f"field_{x}":x for x in range(10)},
}
input_data.update(recursive_input_data)
schema_input_data = {"foo":"hello","bar":10,"zar":True}
input_data.update(recursive_input_data)
manuals = [True,False]
defaults = methods.copy()
field_sets = [["foo"],["bar"],["zar"],["foo","bar"],["bar","zar"],["foo","bar","zar"]]
amounts = [1,5,10]
'''
input_sets = []
for method in methods:
    for manual in manuals:
        for default in defaults:
            for fields in field_sets:
                for amount in amounts:
                    input_sets.append((schema_model,input_data,method,manual,0,default,fields,amount))
#should be 3*2*3*6*3 = 18*18 = 324 tests
@pytest.mark.parametrize("schema_model, data, method, manual, seed, default, fields, amount",input_sets)
def test_anonymise(schema_model, data, method, manual, seed, default, fields, amount):
    return_data = anon.anonymise(schema_model, data, method, manual, seed, default, fields, amount)
    assert isinstance(return_data,dict)
    assert all([isinstance(x,list) for x in return_data.values()])
    assert all([isinstance(x,BaseModel) for y in return_data.values() for x in y])
    assert all([len(x)==amount for x in return_data.values()])
    
'''