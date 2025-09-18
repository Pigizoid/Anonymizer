
from src.sm.library.dataclass import AnonField,SynthField
from src.sm.tools.model_funcs import get_model_fields
from tests.schema import test_user
import pytest


fields = get_model_fields(test_user)


amounts = [1,5,10]
anon_methods = ["mask","perturb"]
anon_datas = ["hello",True,100,{"field1":"bob"}]
anon_inputs = []
for amount in amounts:
    for method in anon_methods:
        for data in anon_datas:
            anon_inputs.append(("foo",amount,method,data))
@pytest.mark.parametrize("field_name,amount,method,data",anon_inputs)
def test_AnonField(field_name,amount,method,data):
    return_value = AnonField(field_name,amount,method,data)
    assert len(return_value.vals) == amount

synth_inputs = []
synth_methods = ["faker","mimesis","mixed"]
for field_name,field_data in fields.items():
    for method in synth_methods:
        synth_inputs.append((field_name,amount,method,field_data))
@pytest.mark.parametrize("field_name,amount,method,field_data",synth_inputs)
def test_SynthField(field_name,amount,method,field_data):
    return_value = SynthField(field_name,amount,method,field_data)
    assert len(return_value.vals) == amount



test_inpts= []
for amount in amounts:
    test_inpts.append((amount))
@pytest.mark.parametrize("amount",test_inpts)
def test_MixedField(amount):
    data = dict(
        foo=AnonField("foo",amount,"mask","hello"),
        bar=AnonField("bar",amount,"perturb",10),
        zar=AnonField("zar",amount,"mask",True),
        lar=SynthField("name",amount,"faker",fields["name"]),
        nest=AnonField("nest",amount,"mask",{"field1":"bob"})
    )
    assert all([len(x.vals) == amount for x in data.values()])
    assert all([type(x) == content.type for content in data.values() for x in content.vals])
    assert data['lar'].method == "faker"

def test_Field_comprehension():
    data = {
        field_name : SynthField(field_name,amount,"faker",field_data) 
        for field_name,field_data in fields.items()
    }
    assert all([len(x.vals) == amount for x in data.values()])
    assert all([x.method == "faker" for x in data.values()])

