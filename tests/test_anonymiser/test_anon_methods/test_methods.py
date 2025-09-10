import pytest

from src.sm.synthesiser.synthesiser import Synthesiser
from src.sm.anonymiser.anon_methods.methods import mask_value,perturb_value,synth_value


@pytest.mark.parametrize("field_value,field_type,expected", [
    (True,bool,False),
    (1.5,float,0.0),
    (15,int,0),
    (b'hello',bytes,b'0'),
    ("hello",str,"****"),
    ([],list,[])
])
def test_mask_value(field_value, field_type, expected):
    assert mask_value(field_value, field_type) == expected


@pytest.mark.parametrize("field_value,field_type", [
    (True,bool),
    (1.5,float),
    (15,int),
    (b'hello',bytes),
    ("hello",str),
    ([],list)
])
def test_perturb_value(field_value, field_type):
    assert isinstance(perturb_value(field_value, field_type), field_type)


synth = Synthesiser()
@pytest.mark.parametrize("field_name,field_type", [
    ("bool",bool),
    ("float",float),
    ("int",int),
    ("bytes",bytes),
    ("string",str),
    ("list",list)
])
def test_synth_value(field_name, field_type):
    assert isinstance(synth_value(synth, field_name, field_type), field_type)
