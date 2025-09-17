from typing import Dict, List, Tuple, Set, Any, Union
from sm.pre_made_data import recursive_types
import random
import string
from sm.synthesiser.synthesiser import Synthesiser


def mask_value(field_value: Any) -> Any:
    """
    Inputs:
        value of Any type
    Outputs:
        if value type in [bool,float,int,bytes,str]:
            returns masked value of the same type as value
        else:
            returns original value unmasked
    """
    field_type = type(field_value)
    if field_type is bool:
        return False
    elif field_type is float:
        return 0.0
    elif field_type is int:
        return 0
    elif field_type is bytes:
        return b"0"
    elif field_type is str:
        return "****"
    else:
        return field_value


def perturb_value(field_value: Any) -> Any:
    """
    Inputs:\n
        value of Any type
    Outputs:\n
        if value type in [bool,float,int,bytes,str]:
            returns perturbed value of the same type as value
        else:\n
            returns original value unchanged
    """
    field_type = type(field_value)
    if field_type is bool:
        return bool(random.randint(0, 1))
    elif field_type is float:
        amount = 0.1  # 10%
        noise = (
            field_value * amount * (random.random() * 2 - 1)
        )  # (0 to 1)*2 -1   ->   -1 to 1
        return float(field_value + noise)
    elif field_type is int:
        amount = 0.2  # 10%
        noise = (
            field_value * amount * (random.random() * 2 - 1)
        )  # (0 to 1)*2 -1   ->   -1 to 1
        return int(round(field_value + noise) + random.randint(0, 2) - 1)
    elif field_type is bytes:
        new_field_value = bytearray(field_value)
        for x in range(len(new_field_value)):
            new_field_value[x] ^= random.randint(1, 255)  # XOR with random bitmask
        return bytes(new_field_value)
    elif field_type is str:
        new_field_value = ""
        field_val_len = len(field_value)
        length_change = random.randint(1, field_val_len * 2)
        real_letters = min(field_val_len, length_change)  # min:1, max:len
        additional_letters = max(0, length_change - field_val_len)
        for x in range(real_letters):
            new_field_value += chr(
                max(0, (ord(field_value[x]) + (random.randint(0, 10) - 5)))
            )
        for x in range(additional_letters):
            new_field_value += random.choice(string.ascii_letters + string.digits + "_")
        return str(new_field_value)
    else:
        return field_value


def anonymise_value(field_value: Any, anon_method: Tuple[str,str], seed: Union[int,str,None]="random", synth:Synthesiser=None) -> Any:
    """
    Inputs:\n
        value to anonymise
        anon_method = (field_name,method) of the methods "mask","synth","perturb"
        optional seed=Any for seeding random
        optional synth=Synthesiser() object, for using "synth" method
    Outputs:\n
        anonymised value with the same type as the input value
    """
    field_name = anon_method[0]
    anon_method = anon_method[1]
    if anon_method == "mask":
        return mask_value(field_value)
    elif anon_method == "synth":
        return synth.generate_single_value(field_name, type(field_value))
    elif anon_method == "perturb":
        return perturb_value(field_value)
    else:
        raise ValueError(
            f"Invalid anonymisation method '{anon_method}' for field '{field_name}'"
        )


def anonymise_data(input_data:Any, anon_methods: Union[Dict[str,str],Tuple[str,str]], seed: Union[int,str,None]="random", synth:Synthesiser=None) -> Any:
    """
    Recursive method\n
    Inputs:\n
        input data of any type
        anon methods:\n
            1.   {field_names:methods} or 2.   (field_name:method)
            of the methods "mask","synth","perturb"
        optional seed=Any for seeding random
        optional synth=Synthesiser() object, for using "synth" method
    Outputs:\n
        anonymised data with the same type and structure as input data
    """
    input_data_type = type(input_data)
    if input_data_type in recursive_types:
        if input_data_type in [List, list, Tuple, tuple, Set, set]:
            patch_data = []
            for value in input_data:
                patch_data.append(anonymise_data(seed, value, anon_methods, synth))
            if input_data_type in [Tuple, tuple]:
                patch_data = tuple(patch_data)
            elif input_data_type in [Set, set]:
                patch_data = set(patch_data)
            return_data = patch_data
        elif input_data_type in [Dict, dict]:
            patch_data = input_data.copy()
            if isinstance(anon_methods, dict):
                for key, value in input_data.items():
                    if key in anon_methods:  # only filter the specified fields
                        anon_method = (key, anon_methods[key])
                        patch_data[key] = anonymise_data(
                            seed, value, anon_method, synth
                        )
                    else:
                        patch_data[key] = input_data[key]
            else:
                anon_method = anon_methods
                for key, value in input_data.items():
                    patch_data[key] = anonymise_data(seed, value, anon_method, synth)
            return_data = patch_data
        else:
            raise Exception(f"Recersive data type| {input_data_type} |not handled")
    else:
        return_data = anonymise_value(seed, input_data, anon_methods, synth)
    return return_data
