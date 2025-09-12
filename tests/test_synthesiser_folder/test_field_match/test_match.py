import pytest
from pydantic import BaseModel
from src.sm.synthesiser.synthesiser import Synthesiser

class test_Address(BaseModel): #display example
    street: str
    city: str
field_names_1 = ["street","city"]

class test_Address_2(BaseModel): #display example
    street: str
    city: str
    social_security_number: str
    continent: str
field_names_2 = ["street","city","social_security_number","continent"]

def test_match_fields():
    synth = Synthesiser()
    return_value = synth.match_fields(field_names_1)
    assert isinstance(return_value, dict)
    assert all(
        [isinstance(x, str) and isinstance(y, str) for x, y in return_value.items()]
    )

@pytest.mark.parametrize("method,expected",[("faker",[True,True,True,False]),("mimesis",[True,True,False,True]),("mixed",[True,True,True,True])])
def test_match_fields_alternate_methods(method,expected):
    synth_match1 = Synthesiser(method=method)
    return_value = synth_match1.match_fields(field_names_2)
    assert all([(return_value[field_names_2[x]] != "")==expected[x] for x in range(len(field_names_2))])



class test_Address_3(BaseModel):
    street: str
    city: str
    social_security_number: str
    continent: str

class test_Address_4(BaseModel):
    name: str
    phone_number: str
    social_security_number: str
    extra: test_Address_3


def test_recursive_match_fields():
    synth = Synthesiser()
    return_data = synth.recursive_match_fields(test_Address_4)
    assert isinstance(return_data,dict)
    assert all([isinstance(x,dict) for x in return_data.values()])
    assert "test_Address_4" in return_data
    assert "test_Address_3" in return_data
    assert list(return_data["test_Address_4"].keys()) == ["name","phone_number","social_security_number","extra"]
    assert list(return_data["test_Address_3"].keys()) == ["street","city","social_security_number","continent"]
