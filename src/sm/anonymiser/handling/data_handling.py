from typing import Dict, List, Tuple, Set

from ..anon_methods.methods import mask_value,perturb_value,synth_value
from .model_handling import guess_type

from ...pre_made_data import recursive_types




def anonymise_value(seed, field_value, anon_methods, synth=None):
    '''Inputs a value and anonymisation method and returns an anonymised value
    anon_method = [field_name,method] of the methods "mask","synth","perturb"'''
    field_name = anon_methods[0]
    anon_method = anon_methods[1]
    field_type = guess_type(field_value)
    field_value = field_type(field_value)
    if anon_method == "mask":
        return mask_value(field_value,field_type)
    elif anon_method == "synth":
        return synth_value(synth,field_name,field_type)
    elif anon_method == "perturb":
        return perturb_value(field_value,field_type)
    else:
        raise ValueError(f"Invalid anonymisation method '{anon_method}' for field '{field_name}'")

def anonymise_data(seed, input_data, anon_methods, synth=None):
    '''A recursive data nonymiser, inputs json data as python structure and, a dict of anon_methods
    Outputs the same structure anonymised
    anon_methods = {field_name:method} of the methods "mask","synth","perturb"'''
    input_data_type = type(input_data)
    if input_data_type in recursive_types:
        if input_data_type in [List,list, Tuple,tuple, Set,set]:
            patch_data = []
            for value in input_data:
                patch_data.append(anonymise_data(seed,value,anon_methods, synth))
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
                        patch_data[key] = anonymise_data(seed,value,anon_method, synth)
                    else:
                        patch_data[key] = input_data[key]
            else:
                anon_method = anon_methods
                for key,value in input_data.items():
                    patch_data[key] = anonymise_data(seed,value,anon_method, synth)
            return_data = patch_data
        else:
            raise Exception(
                f"Recersive data type| {input_data_type} |not handled"
            )
    else:
        return_data = anonymise_value(seed, input_data, anon_methods, synth)
    return return_data
        