import pydantic

from pydantic import BaseModel
from typing import Dict, List, Tuple, Set

from .synthesiser import Synthesiser
from .pre_made_data import recursive_types
from ..tools.model_funcs import get_model_fields

import random
import string

class Anonymiser:
    def subset_model(self,schema_model, field_names) -> BaseModel:
        fields = {
            name: (field.annotation, field.default)
            for name, field in get_model_fields(schema_model).items()
            if name in field_names
        }
        return pydantic.create_model("new_schema_model", **fields)

    def guess_type(self,value) -> type:
        if value is None:
            return type(None)
        try:
            if value.lower() in {"true", "false"}:
                return bool
        except:
            pass
        try:
            int(value)
            return int
        except:
            pass
        try:
            float(value)
            return float
        except:
            pass
        try:
            complex(value)
            return complex
        except:
            pass
        if isinstance(value, bytes):
            return bytes
        elif isinstance(value, tuple):
            return tuple
        elif isinstance(value, list):
            return list
        elif isinstance(value, set):
            return set
        elif isinstance(value, frozenset):
            return frozenset
        elif isinstance(value, dict):
            return dict
        else:
            return str

    def new_model(self,data, field_names) -> BaseModel:
        fields = {
            name: (self.guess_type(content))
            for name, content in data.items()
            if name in field_names
        }
        return pydantic.create_model("new_schema_model", **fields)

    def mask_value(self,field_value,field_type):
        if field_type == bool:
            return False
        elif field_type == float:
            return 0.0
        elif field_type == int:
            return 0
        elif field_type == bytes:
            return b'0'
        elif field_type == str:
            return "****" 
        else:
            return field_value

    def perturb_value(self,field_value,field_type):
        if field_type == bool:
            return bool(random.randint(0,1))
        elif field_type == float:
            amount = 0.1 #10%
            noise = field_value * amount * (random.random() * 2 - 1) # (0 to 1)*2 -1   ->   -1 to 1
            return field_value + noise
        elif field_type == int:
            amount = 0.2 #10%
            noise = field_value * amount * (random.random() * 2 - 1) # (0 to 1)*2 -1   ->   -1 to 1
            return round(field_value + noise) + random.randint(0,2)-1
        elif field_type == bytes:
            new_field_value = bytearray(field_value)
            for x in range(len(new_field_value)):
                new_field_value[x] ^= random.randint(1, 255)  # XOR with random bitmask
            return bytes(new_field_value)
        elif field_type == str:
            new_field_value = ""
            field_val_len = len(field_value)
            length_change = random.randint(1,field_val_len*2)
            real_letters = min(field_val_len,length_change)  #min:1, max:len
            additional_letters = max(0,length_change-field_val_len)
            for x in range(real_letters):
                new_field_value+= chr(max(0,(ord(field_value[x])+(random.randint(0,10)-5))))
            for x in range(additional_letters):
                new_field_value+= random.choice(string.ascii_letters + string.digits + '_')
            return new_field_value
        else:
            return field_value

    def anonymise_value(self, seed, field_value, anon_methods):
        field_name = anon_methods[0]
        anon_method = anon_methods[1]
        field_type = self.guess_type(field_value)
        field_value = field_type(field_value)
        if anon_method == "mask":
            return self.mask_value(field_value,field_type)
        elif anon_method == "synth":
            return self.synth.generate_single_value(field_name,field_type)
        elif anon_method == "perturb":
            return self.perturb_value(field_value,field_type)
    
    def anonymise_data(self, seed, input_data, anon_methods):
        input_data_type = type(input_data)
        if input_data_type in recursive_types:
            if input_data_type in [List,list, Tuple,tuple, Set,set]:
                patch_data = []
                for value in input_data:
                    patch_data.append(self.anonymise_data(seed,value,anon_methods))
                if input_data_type in [Tuple,tuple]:
                    patch_data = tuple(patch_data)
                elif input_data_type in [Set,set]:
                    patch_data = set(patch_data)
                return_data = patch_data
            elif input_data_type in [Dict, dict]:
                patch_data = input_data.copy()
                if isinstance(anon_methods,dict):
                    for key,value in input_data.items():
                        if key in anon_methods:  #only filter the specified fields
                            anon_method = [key,anon_methods[key]]
                            patch_data[key] = self.anonymise_data(seed,value,anon_method)
                        else:
                            patch_data[key] = input_data[key]
                else:
                    anon_method = anon_methods
                    for key,value in input_data.items():
                        patch_data[key] = self.anonymise_data(seed,value,anon_method)
                return_data = patch_data
            else:
                raise Exception(
                    f"Recersive data type| {input_data_type} |not handled"
                )
        else:
            return_data = self.anonymise_value(seed, input_data, anon_methods)
        return return_data
        






    def anonymise(
        self, schema_model, data, method, manual, seed, default, fields, amount
    ) -> Dict[str, List[BaseModel]]:
        # data comes in as a dict of dicts
        self.synth = Synthesiser(method=method)
        anonymised_data = {}

        print(fields.values())
        print(f"Seed: {seed}")
        print(default)
        # default : val
        # value types
        # - default
        # - mask
        # - synth
        # - perturb

        for index, data_entry in data.items():
            schema_match = True
            try:
                schema_model(**data_entry)
            except:
                schema_match = False
            if manual:
                field_names = fields.keys()
                anon_methods = fields
            else:  # auto
                if schema_match:
                    field_names = [x[0] for x in self.synth.get_model_data(schema_model)]
                    
                else:
                    field_names = data_entry.keys()
                    print(
                        f"Schema '{schema_model.__name__}' does not match data, defaulting to data keys"
                    )
                anon_methods = {field_name:default for field_name in field_names}
            # name_match_pairs = synth.match_fields(field_names)
            # field_names = [ field for field,match in name_match_pairs.items() if match != ""]
            result_schema = self.new_model(data_entry, data_entry.keys())
            
            if manual or default != "synth":
                return_data = [ self.anonymise_data(seed=seed, input_data=data_entry, anon_methods=anon_methods) for _ in range(amount) ]
            else:
                if schema_match:
                    new_schema_model = self.subset_model(schema_model, field_names)
                else:
                    new_schema_model = self.new_model(data_entry, field_names)
                
                return_data = self.synth.synthesise(
                    new_schema_model, method=method, amount=amount, seed=seed
                )
            anonymised_data_set = []
            for return_entry in return_data:
                new_fields = data_entry.copy()
                for field in field_names:
                    if isinstance(return_entry,dict):
                        new_fields[field] = return_entry[field]
                    else:
                        new_fields[field] = getattr(return_entry, field)
                anonymised_data_set.append(result_schema(**new_fields))
            anonymised_data[index] = anonymised_data_set
        # data returns as a dict of lists of models
        return anonymised_data
