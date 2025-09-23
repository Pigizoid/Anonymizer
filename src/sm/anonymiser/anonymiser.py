from pydantic import BaseModel, create_model
from typing import Dict, List, Tuple, Set, Any, Union
from sm.tools.model_funcs import get_model_data, get_model_fields
from sm.synthesiser.synthesiser import Synthesiser
from sm.pre_made_data import recursive_types
import random
import string


# ----- Model handling -----
def subset_model(schema_model:BaseModel, field_names:List[str]) -> BaseModel:
    """
    Inputs:\n
        schema model
        list of field names
    Outputs:\n
        schema model with name "new_schema_model" that contanins only field names in the list of field names
        inferences output data types from the input schema
    """
    fields = {
        name: (field.annotation, field.default)
        for name, field in get_model_fields(schema_model).items()
        if name in field_names
    }
    return create_model("new_schema_model", **fields)


def new_model(data:Dict[str,Any], field_names:List[str]) -> BaseModel:
    """
    Inputs:\n
        dict of data = {field_name:content}
        list of field names
    Outputs:\n
        schema model with name "new_schema_model" that contanins only field names in the list of field names
        inferences output data types from the input data
    """
    fields = {
        name: (type(content))
        for name, content in data.items()
        if name in field_names
    }
    return create_model("new_schema_model", **fields)
# ----- Model handling -----


# ----- Value handling -----
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
                patch_data.append(anonymise_data(value, anon_methods, seed, synth))
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
                            value, anon_method, seed, synth
                        )
                    else:
                        patch_data[key] = input_data[key]
            else:
                anon_method = anon_methods
                for key, value in input_data.items():
                    patch_data[key] = anonymise_data(value, anon_method, seed, synth)
            return_data = patch_data
        else:
            raise Exception(f"Recersive data type| {input_data_type} |not handled")
    else:
        return_data = anonymise_value(input_data, anon_methods, seed, synth)
    return return_data
# ----- Value handling -----

# ----- Central function -----
def anonymise(
    schema_model:Union[BaseModel,None], data:Dict[str,Any], method:str, manual:bool, default:str, fields:Dict[str,str], amount:int, seed:Union[int,str,None]="random"
) -> Dict[str, List[BaseModel]]:
    """
    Inputs:\n
        schema model
        data as a dict of dicts
            {index:  #index as integer
                {json data}
            }
        method of the methods"mixed","mimesis","faker"
        "manual" boolean, to flag automatic or manual anonymisation
        default anonymisation method of the methods "mask","synth","perturb"
        dict of fields = [field_name,method] of the methods "default","mask","synth","perturb"
        amount as an int
        optional anonymisation seed
    Outputs:\n
        dict of lists of schema BaseModel
            {index:  #index as integer
                [{BaseModel}] * amount
            }
    """
    anonymised_data = {}
    
    print(f"fields: {fields.values()} |Seed: {seed} |Default: {default}")
    print(data)
    for index, data_entry in data.items():
        schema_match = True
        if schema_model is not None:
            try:
                schema_model(**data_entry)
            except:
                schema_match = False
        else:
            schema_match = False
        if manual:
            field_names = fields.keys()
            anon_methods = {}
            for key, value in fields.items():
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
            anon_methods = {field_name: default for field_name in field_names}
        # name_match_pairs = synth.match_fields(field_names)
        # field_names = [ field for field,match in name_match_pairs.items() if match != ""]
        result_schema = new_model(data_entry, data_entry.keys())

        synth = Synthesiser(method=method)

        if manual or default != "synth":
            return_data = [
                anonymise_data(
                    input_data=data_entry,
                    anon_methods=anon_methods,
                    seed=seed,
                    synth=synth,
                )
                for _ in range(amount)
            ]
        else:
            if schema_match:
                new_schema_model = subset_model(schema_model, field_names)
            else:
                new_schema_model = new_model(data_entry, field_names)

            return_data = synth.synthesise(
                new_schema_model, method=method, amount=amount, seed=seed
            )
        for return_entry in return_data:
            anonymised_data_set = []
            new_fields = data_entry.copy()
            for field in field_names:
                if isinstance(return_entry, dict):
                    new_fields[field] = return_entry[field]
                else:
                    new_fields[field] = getattr(return_entry, field)
            anonymised_data_set.append(result_schema(**new_fields))
        anonymised_data[index] = anonymised_data_set
    return anonymised_data
# ----- Central function -----