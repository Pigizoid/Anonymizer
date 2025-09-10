import pytest

from src.sm.synthesiser.synthesiser import Synthesiser
from src.sm.anonymiser.handling.data_handling import anonymise_value,anonymise_data

def check_types_match(input_val, return_val):
    if type(input_val) != type(return_val):
        return False
    if isinstance(input_val, dict):
        if input_val.keys() != return_val.keys():
            return False
        return all(check_types_match(input_val[k], return_val[k]) for k in input_val)
    elif isinstance(input_val, (list, tuple)):
        if len(input_val) != len(return_val):
            return False
        return all(check_types_match(i, j) for i, j in zip(input_val, return_val))
    elif isinstance(input_val, set):
        input_types = {type(i) for i in input_val}
        return_types = {type(i) for i in return_val}
        return input_types == return_types
    else:
        return type(input_val) == type(return_val)




vals = [True,1.5,15,b'hello',"hello",[]]
methods = ["mask","synth","perturb"]
synth = Synthesiser()
anonymise_value_test_data = []
for method in methods:  
    anonymise_value_test_data.extend([(val,["name",method]) for val in vals])
@pytest.mark.parametrize("seed, field_value, anon_methods, synth",[(0,val,anon_methods,synth) for val,anon_methods in anonymise_value_test_data])
def test_anonymise_value(seed, field_value, anon_methods, synth):
    assert isinstance(anonymise_value(seed, field_value, anon_methods, synth), type(field_value))


input_data = { f"field_{val}":val for val in vals }
recursive_input_data = {
    "list": [x for x in range(10)],
    "tuple": (x for x in range(10)),
    "set": set(x for x in range(10)),
    "dict": {f"field_{x}":x for x in range(10)},
}
input_data.update(recursive_input_data)
anon_methods = ["mask" for _ in range(len(input_data))]
def test_anonymise_data_mask():
    return_data = anonymise_data(0,input_data,anon_methods)
    assert return_data != input_data
    assert check_types_match(input_data,return_data)

anon_methods = ["synth" for _ in range(len(input_data))]
def test_anonymise_data_synth():
    return_data = anonymise_data(0,input_data,anon_methods,synth)
    assert return_data != input_data
    assert check_types_match(input_data,return_data)

anon_methods = ["perturb" for _ in range(len(input_data))]
def test_anonymise_data_perturb():
    return_data = anonymise_data(0,input_data,anon_methods)
    assert check_types_match(input_data,return_data)
    
