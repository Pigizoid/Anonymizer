from pydantic import BaseModel
from typing import Dict, List
from .handling.model_handling import new_model, subset_model
from .handling.data_handling import anonymise_data

from sm.tools.model_funcs import get_model_data
from sm.synthesiser.synthesiser import Synthesiser



class Anonymiser:

    def anonymise(
        self, schema_model, data, method, manual, seed, default, fields, amount
    ) -> Dict[str, List[BaseModel]]:
        '''
        Inputs:
            a schema model
            data as a dict of dicts
                {index:  #index as integer
                    {json data}
                }
            a method of the methods"mixed","mimesis","faker"
            a "manual" boolean, to flag automatic or manual anonymisation
            an anonymisation seed
            the default anonymisation method of the methods "mask","synth","perturb"
            a dict of fields = [field_name,method] of the methods "default","mask","synth","perturb"
            an amount as an int
        Outputs:
            a dict of lists of schema BaseModel
                {index:  #index as integer
                    [{BaseModel}] * amount
                }
        '''
        anonymised_data = {}

        print(fields.values())
        print(f"Seed: {seed}")
        print(default)

        for index, data_entry in data.items():
            schema_match = True
            try:
                schema_model(**data_entry)
            except:
                schema_match = False
            if manual:
                field_names = fields.keys()
                anon_methods = {}
                for key,value in fields.items():
                    if value == "default":
                        value = default
                    anon_methods[key] = value

            else:  # auto
                if schema_match:
                    field_names = [x[0] for x in get_model_data(schema_model)]
                    
                else:
                    field_names = data_entry.keys()
                    print(
                        f"Schema '{schema_model.__name__}' does not match data, defaulting to data keys"
                    )
                anon_methods = {field_name:default for field_name in field_names}
            # name_match_pairs = synth.match_fields(field_names)
            # field_names = [ field for field,match in name_match_pairs.items() if match != ""]
            result_schema = new_model(data_entry, data_entry.keys())
            
            synth = Synthesiser(method=method)

            if manual or default != "synth":
                return_data = [ anonymise_data(seed=seed, input_data=data_entry, anon_methods=anon_methods, synth=synth) for _ in range(amount) ]
            else:
                if schema_match:
                    new_schema_model = subset_model(schema_model, field_names)
                else:
                    new_schema_model = new_model(data_entry, field_names)
                
                return_data = synth.synthesise(
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
        return anonymised_data
