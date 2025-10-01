
from dataclasses import dataclass, field
from typing import Any, List, Dict, Type, Annotated, Union
if __name__ == "__main__":
    import os,sys,pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
from smoke_mirrors.anonymiser.anonymiser import anonymise_data
from smoke_mirrors.synthesiser.synthesiser import JsonSynthesiser
from smoke_mirrors.tools.model_funcs import get_json_model_fields
from pydantic import BaseModel, Field
from src.smoke_mirrors.synthesiser.helper_funcs.matching_fields import match_fields
from src.smoke_mirrors.synthesiser.helper_funcs.constraints import check_generation_constraints


'''
functionality:
    amount = 10
    data = dict(
        foo=AnonField("foo",amount,"mask","hello"),
        bar=AnonField("bar",amount,"perturb",10),
        zar=AnonField("zar",amount,"mask",True),
        lar=SynthField("name",amount,"faker",fields["name"]),
        nest=AnonField("nest",amount,"mask",{"field1":"bob"})
    )

    data['foo'].type -> str
    data['foo'].vals -> ["hrllo","ello","ghidi","*****",...]
    data['foo'].method -> "mask"

    data['lar'].type -> str
    data['lar'].vals -> ["john","larry","penny",...]
    data['lar'].method -> "faker"
'''



@dataclass
class SMField:
    field_name: str
    amount: int
    method: str
    type: Type = field(init=False)
    vals: List[Any] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.type = type(self.value)



@dataclass
class AnonField(SMField):
    value: Any

    def __post_init__(self):
        self.type = type(self.value)
        if self.method == "synth":
            raise ValueError("Method 'synth' is not available for AnonField, use SynthField instead")
        self.vals = [anonymise_data(self.value,(self.field_name,self.method)) for _ in range(self.amount)]
        

@dataclass
class SynthField(SMField):
    field_info : Dict

    def __post_init__(self):
        self.synth = JsonSynthesiser(method=self.method)
        self.match_name = match_fields([self.field_name],method=self.method)[self.field_name]
        self.applied_constraints = check_generation_constraints(self.field_name,self.field_info)
        self.type = self.applied_constraints["annotation"]

        self.vals = [self.synth.generate_synth_data(self.field_name, self.match_name, self.applied_constraints, f"{self.field_name}(?)[{self.amount}]") for _ in range(self.amount)]
        #self.vals = [self.synth.generate_single_value(self.field_name,self.type) for _ in range(self.amount)]


class User(BaseModel):
    name: str = Field(pattern=r'[A-Z]{1,1}[a-z]{1,10}')
    email: str
    age: int = Field(ge=10,le=100)


if __name__ == "__main__":
    fields = get_json_model_fields(User.model_json_schema)
    amount = 10
    #usage 1
    data = dict(
        foo=AnonField("foo",amount,"mask","hello"),
        bar=AnonField("bar",amount,"perturb",10),
        zar=AnonField("zar",amount,"mask",True),
        lar=SynthField("name",amount,"faker",fields["name"]),
        nest=AnonField("nest",amount,"mask",{"field1":"bob"})
    )
    for z in range(amount):
        print(dict({x:data[x].vals[z] for x in data.keys()}))

    #usage 2
    data = {
        field_name : SynthField(field_name,amount,"faker",field_data) 
        for field_name,field_data in fields.items()
        }

    for z in range(amount):
        print(dict({x:data[x].vals[z] for x in data.keys()}))



