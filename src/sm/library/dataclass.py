
from dataclasses import dataclass, field
from typing import Any, List, Dict, Type, Annotated
import os,sys,pathlib
if __name__ == "__main__":
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
from sm.anonymiser.handling.data_handling import anonymise_data
from sm.synthesiser.synthesiser import Synthesiser
    

from pydantic import Field

'''
functionality:
    amount = 10
    data = dict(
        foo=AnonField("hello",amount,"masked"),
        bar=AnonField(10,amount,"perturb"),
        zar=AnonField(True,amount,"mask"),
        lar=SynthField("name",amount,"faker",Field(pattern=r'[a-z]*')),
        nest=AnonField({"field1":"bob",...},amount,"masked")
    )

    data.foo.type -> str
    data.foo.vals -> ["hrllo","ello","ghidi","*****",...]
    data.foo.method -> "anon \\ masked"

    data.lar.type -> str
    data.lar.vals -> ["john","larry","penny",...]
    data.foo.method -> "synth \\ faker"
'''



@dataclass
class SMField:
    field_name: str
    value: Any
    amount: int
    method: str
    type: Type = field(init=False)
    vals: List[Any] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.type = type(self.value)

    def generate(self):
        raise Exception("SMField generation not to be called directly, use AnonField or SynthField")



@dataclass
class AnonField(SMField):
    def __post_init__(self):
        self.type = type(self.value)
        self.generate()
    
    def generate(self):
        self.vals = [anonymise_data(self.value,(self.field_name,self.method)) for _ in range(self.amount)]
        

@dataclass
class SynthField(SMField):
    def __post_init__(self):
        self.synth = Synthesiser(method=self.method)
        self.type = self.value
        self.generate()

    def generate(self):
        self.vals = [self.synth.generate_single_value(self.field_name,self.value) for _ in range(self.amount)]
        



amount = 10
data = dict(
    foo=AnonField("foo","hello",amount,"mask"),
    bar=AnonField("bar",10,amount,"perturb"),
    zar=AnonField("zar",True,amount,"mask"),
    lar=SynthField("name",str,amount,"faker"),
    nest=AnonField("nest",{"field1":"bob"},amount,"mask")
)
print(data.items())
for z in range(amount):
    print(dict({x:data[x].vals[z] for x in data.keys()}))






