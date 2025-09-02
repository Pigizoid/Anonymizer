from pydantic import BaseModel

import typing
from typing import get_origin, get_args
from typing import List, Dict, Tuple, Set, Union, Literal, Annotated, Optional

from faker import Faker
import mimesis
from mimesis import Generic


import random
import inspect

import rstr
import exrex
import re
import string
from decimal import Decimal, ROUND_HALF_UP

import time

fake = Faker()
generic = Generic(mimesis.locales.Locale.EN)

from .pre_made_data import provider_return_types,python_builtin_types,recursive_types,all_constr_attribs,default_constr_dict


class Synthesiser:
    
    def __init__(self, method="faker"):
        self.outputpooling = {}
        self.word_list, methods_map = self.list_match_methods(method)
        self.word_tokens = {word: word.split("_") for word in self.word_list }
        self.word_tokens_set = {word: set(word.split("_")) for word in self.word_list }
        self.resolved_methods = self.make_resolved_methods(self.word_list, methods_map)
    
    def print_path(self,generate_path, elapsed_time):
        pooling_depth = len(list(map(int, re.findall(r"\[(\d+)\]", generate_path))))
        print(f"Time taken: {elapsed_time:.2f} seconds","    "*pooling_depth,generate_path)
    
    def list_faker_methods(self):
        methods = []
        methods_map = {}
        fake = Faker()
        for attr in dir(fake):
            try:  # this is used to ensure the providers dont error when called
                if not attr.startswith("_") and callable(getattr(fake, attr)):
                    methods.append(attr)
                    methods_map[attr] = fake
            except:
                pass
        return (methods, methods_map)

    
    def list_mimesis_methods(self):
        methods = []
        methods_map = {}
        for provider_name in sorted(generic.__dict__.keys()):
            provider_cls = getattr(generic, provider_name)
            if inspect.isclass(provider_cls):
                try:
                    sig = inspect.signature(provider_cls)
                    if all(
                        p.default != inspect.Parameter.empty
                        or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                        for p in sig.parameters.values()
                    ):
                        instance = provider_cls()
                    else:
                        continue
                except (TypeError, ValueError):
                    continue
                for attr in dir(instance):
                    if not attr.startswith("_") and callable(getattr(instance, attr)):
                        methods.append(attr)
                        methods_map[attr] = instance
        return (methods, methods_map)

    
    def list_match_methods(self, method):
        methods = []
        methods_map = {}
        if method == "faker":
            methods, methods_map = self.list_faker_methods()
        elif method == "mimesis":
            methods, methods_map = self.list_mimesis_methods()
        elif method == "mixed":
            methodsF, methods_mapF = self.list_faker_methods()

            methodsM, methods_mapM = self.list_mimesis_methods()

            methods = methodsF
            methods.extend(methodsM)

            methods_map = methods_mapF
            methods_map.update(methods_mapM)
        else:
            raise Exception(f"Unexpected method: {method}")
        methods = list(set(methods))
        # for method in methods:
        # print(f"Method: {method}, Map: {methods_map[method]}")
        return (methods, methods_map)

    
    def get_model_data(self, model):
        model_data = []
        for field_name, model_field in model.model_fields.items():
            model_data.append([field_name, model_field])
        # print(type(model_field))
        return model_data

    
    def levenshtein_distance(self, word1, word2, modifiers=[0, 0, 0]):
        len_word1, len_word2 = len(word1), len(word2)
        distance_point = [
            [0 for _ in range(len_word2 + 1)] for _ in range(len_word1 + 1)
        ]
        for i in range(len_word1 + 1):
            distance_point[i][0] = i
        for j in range(len_word2 + 1):
            distance_point[0][j] = j
        for i in range(1, len_word1 + 1):
            for j in range(1, len_word2 + 1):
                cost = 0 if word1[i - 1] == word2[j - 1] else 1
                distance_point[i][j] = min(
                    distance_point[i - 1][j] + 1 + modifiers[0],
                    distance_point[i][j - 1] + 1 + modifiers[1],
                    distance_point[i - 1][j - 1] + cost + modifiers[2],
                )
        return distance_point[len_word1][len_word2]

    
    def calc_difference(self,target_word, word, word_tokens, word_tokens_set, target_tokens, target_tokens_set):
        if target_tokens_set == word_tokens_set:  # name_first -> first_name
            return 0
        
        if (
            ("".join([letter[0] for letter in word.split("_")])) == target_word.lower()
        ):  # abbreviation mapping	ssn -> social_security_number   #ssn is target word
            return 0
        if (
            "".join([letter[0] for letter in target_word.lower().split("_")])
        ) == word:  # abbreviation mapping	social_security_number -> ssn
            return 0

        main_distance = self.levenshtein_distance(
            target_word, word.lower()
        )  # close early exit
        if main_distance <= 1 or main_distance >= len(target_word):
            return main_distance

        main_distance_joined = main_distance - (
            (
                len(target_word) - len(target_word.replace("_", ""))
            ) 
            +(
                len(word) - len(word.replace("_", ""))
            )
        )
        if main_distance_joined <= 1:
            return main_distance_joined
        

        token_distance = 0
        cross_points = len(word_tokens) * len(target_tokens)
        if len(target_tokens) != len(word_tokens):
            if len(target_tokens) < len(word_tokens):
                if target_tokens_set.issubset(word_tokens_set):
                    token_distance = word_tokens.index(target_tokens[0]) * cross_points
                    distance = (main_distance / 2 + token_distance) / 2
                else:
                    token_distance = (
                        self.levenshtein_distance(target_word, word.lower())
                        * cross_points
                    )
                    distance = (main_distance + token_distance) / 2
            else:
                if word_tokens_set.issubset(target_tokens_set):
                    token_distance = target_tokens.index(word_tokens[0]) * cross_points
                    distance = (main_distance / 2 + token_distance) / 2
                else:
                    token_distance = (
                        self.levenshtein_distance(target_word, word.lower())
                        * cross_points
                    )
                    distance = (main_distance + token_distance) / 2
        else:
            distance = (
                self.levenshtein_distance(target_word, word.lower())
                * cross_points
            )
        return distance

    
    def match_fields(self, field_names):

        field_matches = []
        for t_word in field_names:
            closest_matches = []
            distances = []
            target_word = t_word.lower()
            target_tokens = target_word.split("_")
            target_tokens_set = set(target_tokens)
            for word in self.word_list:
                distance = self.calc_difference(
                    target_word, word, self.word_tokens[word], self.word_tokens_set[word], target_tokens, target_tokens_set
                )
                distances.append([word, distance])
            sorted_by_distance = sorted(distances, key=lambda dist: dist[1])
            if sorted_by_distance != [] and sorted_by_distance[0] != []:
                min_value = sorted_by_distance[0][1]
            else:
                min_value = (len(target_word) // 2 - 0.5)+1
            if min_value > (len(target_word) // 2 - 0.5):
                potential_matches = []
                for word in self.word_list:
                    if target_word in word:
                        distance = self.levenshtein_distance(
                            target_word, word, [-0.5, 0.5, -0.5]
                        )
                        potential_matches.append([word, distance])
                sorted_match_by_distance = sorted(potential_matches, key=lambda dist: dist[1])
                if sorted_match_by_distance != [] and sorted_match_by_distance[0] != []:
                    min_value = sorted_match_by_distance[0][1]
                else:
                    min_value = (len(target_word) // 2)+1
                if min_value > len(target_word) // 2:
                    closest_matches = [
                        item
                        for item in sorted_match_by_distance
                        if item[1] == min_value
                    ]
            else:
                closest_matches = [
                    item for item in sorted_by_distance if item[1] == min_value
                ]
            if len(closest_matches) == 0:
                field_matches.append("")
            else:
                field_matches.append(closest_matches[0][0])
        
        field_match_pairs = {}
        for name,match in zip([x for x in field_names], field_matches):
            field_match_pairs[name] = match
        
        return field_match_pairs
    
    def recursive_match_schema_fields(self, schema_model, field_match_pairs={}): #needs test
        model_data = self.get_model_data(schema_model)
        field_names =  [ x[0] for x in model_data]
        if schema_model.__name__ not in field_match_pairs.keys():
            field_match_pairs[schema_model.__name__] = self.match_fields(field_names)
        
        
        for x in model_data:
            data_type = x[1].annotation
            if (
                inspect.isclass(data_type)
                and issubclass(data_type, BaseModel)
                and data_type.__name__ not in field_match_pairs.keys()
                ):
                #print(data_type.__name__, field_match_pairs.keys())
                field_match_pairs.update(self.recursive_match_schema_fields(data_type))
                
        
        return field_match_pairs
        
    
    def check_generation_constraints(self, name, field):
        cout = False
        if cout:
            print("__")
            print(f"	Name:{name}")
            print(f"	Field:{field}")
        field_info = field.metadata
        constraints = {}
        is_req = getattr(field, "is_required", None)
        constraints["required"] = is_req() if callable(is_req) else bool(is_req)
        for attr in all_constr_attribs:
            return_val = getattr(field, attr, None)
            if return_val is None:
                for field_info_element in field_info:
                    return_val = getattr(field_info_element, attr, None)
                    if return_val is not None:
                        break
            constraints[attr] = return_val

        if cout:
            print(f"Constraints:{constraints}")
        annotation = constraints["annotation"]
        constraints["origin"] = get_origin(annotation)
        constraints["args"] = get_args(annotation)

        return constraints

    
    def generate_from_constraints(self, field_name, constraints, generate_path):
        # print(f"Name:{field_name}\nFields:{constraints}")
        # print("__")

        if (generate_path not in self.outputpooling or 
            (
                generate_path in self.outputpooling
                and self.outputpooling[generate_path] == []
            )
        ):
            pooling_numbers = list(map(int, re.findall(r"\[(\d+)\]", generate_path)))
            amount = pooling_numbers[0]
            pooling_count = 1
            for x in pooling_numbers:
                pooling_count *= x
            
            # print(f"Path:{generate_path}")
            # print("count", pooling_count)
            start_time = time.time()
            data_pool = []

            if constraints["pattern"] is not None:
                # print(f"Path:{generate_path}")
                # HERE HERE
                # check if a map exists for this constraint sequence (pattern and constraints)
                # if there is not, generate a map of possible results and then run regex forward pass over each (could be slow for initial pass through)
                # save the forward pass map to global

                # print(constraints["pattern"])

                min_length = constraints["min_length"]
                max_length = constraints["max_length"]
                pattern = constraints["pattern"]
                if min_length is None and max_length is None:
                    pattern = pattern
                elif min_length is not None and max_length is None:
                    # pattern = f"^(?=.{{{min_length},}}$)(?:{pattern})$"
                    if pattern[-1] == "$":
                        pattern = pattern[:-1]
                    while pattern[-1] in ["+", "*", "?"]:
                        pattern = pattern[:-1]
                    pattern = pattern + f"{{{min_length},}}$"

                elif min_length is None and max_length is not None:
                    # pattern = f"^(?=.{{0,{max_length}}}$)(?:{pattern})$"
                    if pattern[-1] == "$":
                        pattern = pattern[:-1]
                    while pattern[-1] in ["+", "*", "?"]:
                        pattern = pattern[:-1]
                    pattern = pattern + f"{{0,{max_length}}}$"
                else:
                    # pattern = f"^(?=.{{{min_length},{max_length}}}$)(?:{pattern})$"
                    if pattern[-1] == "$":
                        pattern = pattern[:-1]
                    while pattern[-1] in ["+", "*", "?"]:
                        pattern = pattern[:-1]
                    pattern = pattern + f"{{{min_length},{max_length}}}$"

                # print(pattern)
                # print(exrex.getone(pattern))
                try:
                    data_pool = [
                        exrex.getone(pattern) for x in range(max(1, pooling_count))
                    ]
                except:
                    data_pool = [
                        rstr.xeger(pattern) for x in range(max(1, pooling_count))
                    ]

                if constraints["to_upper"] is not None:
                    data_pool = [item.upper() for item in data_pool]
                elif constraints["to_lower"] is not None:
                    data_pool = [item.lower() for item in data_pool]

                # print(data_pool)
                # regex_generation =
                # if constraints["annotation"] != str:
                # regex_generation = str(regex_generation)
                # return_value = regex_generation
                # print(data_pool)
                # input("wait...")
                if data_pool == []:
                    raise Exception(f"No data pool for {constraints['pattern']}")
            else:
                # print("Generate from annotation",str(constraints["annotation"]))
                data_type = constraints["annotation"]
                if data_type is str:
                    string_list = string.ascii_letters + string.digits
                    if constraints["to_upper"] is not None:
                        string_list = [item.upper() for item in string_list]
                    elif constraints["to_lower"] is not None:
                        string_list = [item.lower() for item in string_list]
                    if constraints["min_length"] is not None:
                        min_length = constraints["min_length"]
                    else:
                        min_length = 1
                    if constraints["max_length"] is not None:
                        max_length = constraints["max_length"]
                    else:
                        max_length = min_length + max(1, (pooling_count // amount)) + 3
                    data_pool = [
                        "".join(
                            random.choices(
                                string_list, k=(random.randint(min_length, max_length))
                            )
                        )
                        for x in range(max(1, pooling_count))
                    ]

                    if data_pool == []:
                        raise Exception(f"No data pool for str:{generate_path}")

                elif data_type is int or data_type is float or data_type is Decimal:
                    if constraints["lt"] is not None:
                        lt = constraints["lt"] - 1
                    else:
                        lt = 10 * max(1, (pooling_count // amount) + 3)
                    if constraints["le"] is not None:
                        lt = min(lt, constraints["le"])

                    if constraints["gt"] is not None:
                        gt = constraints["gt"] + 1
                    else:
                        gt = lt * -1
                    if constraints["ge"] is not None:
                        gt = max(gt, constraints["ge"])

                    if gt > lt:
                        raise Exception(
                            f"Value for gt:{gt} is greater than value for lt:{lt}"
                        )

                    if constraints["multiple_of"]:
                        multiple_of = constraints["multiple_of"]
                        first = ((gt + multiple_of) // multiple_of) * multiple_of

                        if data_type is int or (
                            (data_type is float)
                            and constraints["decimal_places"] is None
                        ):
                            count = (lt - gt) // multiple_of
                            if count <= 0:
                                raise Exception(
                                    f"No multiples of {multiple_of} fit in the range [{gt}, {lt})."
                                )
                            try:
                                data_pool = [
                                    first + (idx + 1) * multiple_of
                                    for idx in range(count - 1)
                                ]  # count is capped at poling_count
                            except:
                                data_pool == []
                        else:  # data_type is float and constraints["decimal_places"] is not None
                            decimal_places = constraints["decimal_places"]
                            scale = Decimal(10) ** Decimal(decimal_places)
                            decimal_precision = Decimal(10) ** Decimal(
                                decimal_places * -1
                            )
                            # print(str(multiple_of))
                            # print(Decimal(repr(multiple_of)).as_tuple())
                            precision = Decimal(10) ** (
                                Decimal(repr(multiple_of)).as_tuple().exponent
                            )
                            # print(precision)

                            first = (
                                (Decimal(gt) + Decimal(multiple_of))
                                // Decimal(multiple_of)
                            ) * Decimal(multiple_of)

                            first = Decimal(first).quantize(
                                precision, rounding=ROUND_HALF_UP
                            )
                            multiple_of = Decimal(multiple_of).quantize(
                                precision, rounding=ROUND_HALF_UP
                            )

                            min_scaled = Decimal(first * scale)
                            scaled_mult = Decimal(multiple_of * scale)

                            # print(first,multiple_of,min_scaled,scaled_mult,scale)

                            data_pool = []
                            try:
                                for x in range(
                                    int((lt * scale - min_scaled) // scaled_mult) - 1
                                ):
                                    potential_val = (
                                        min_scaled + (x + 1) * scaled_mult
                                    ) / scale
                                    if potential_val == potential_val.quantize(
                                        decimal_precision, rounding=ROUND_HALF_UP
                                    ):
                                        data_pool.append(potential_val)
                                # print(data_pool)
                            except:
                                data_pool = []
                            if data_pool == []:
                                raise Exception(
                                    f"No multiples of {multiple_of} fit in the range [{gt}, {lt}) with max decimal digits{decimal_places}."
                                )
                    else:
                        if constraints["decimal_places"] is not None:
                            data_pool = [
                                round(
                                    random.uniform(gt, lt),
                                    constraints["decimal_places"],
                                )
                                for _ in range(max(1, pooling_count))
                            ]
                        else:
                            data_pool = [
                                random.randint(gt, lt)
                                for _ in range(max(1, pooling_count))
                            ]

                    if constraints["allow_inf_nan"] is not None:
                        allowed_inf_types = ["inf", "-inf", "nan"]
                        if (
                            constraints["gt"] is not None
                            or constraints["ge"] is not None
                        ):
                            allowed_inf_types.remove("-inf")
                            allowed_inf_types.remove("nan")
                        if (
                            constraints["lt"] is not None
                            or constraints["le"] is not None
                        ):
                            allowed_inf_types.remove("inf")
                            if "nan" in allowed_inf_types:
                                allowed_inf_types.remove("nan")
                        if constraints["multiple_of"]:
                            allowed_inf_types = []
                        if constraints["max_digits"]:
                            allowed_inf_types = []

                        if data_type is float and allowed_inf_types != []:
                            data_pool.extend(
                                [
                                    random.choice(allowed_inf_types)
                                    for x in range(len(data_pool))
                                ]
                            )
                    random.shuffle(data_pool)
                    if data_pool == []:
                        raise Exception(f"No data pool for num:{generate_path}")
                    
                    #if the data pool doesnt have enough elements, extend it with copies of existing elements
                    original_len = len(data_pool)
                    while len(data_pool) < pooling_count:
                        # Append elements from the original list in order
                        for item in data_pool[:original_len]:
                            if len(data_pool) < pooling_count:
                                data_pool.append(item)
                            else:
                                break

                elif data_type is bool:
                    data_pool = [
                        random.randint(1, 2) == 1 for x in range(max(1, pooling_count))
                    ]
                    if data_pool == []:
                        raise Exception(f"No data pool for bool:{generate_path}")
                elif data_type is complex:
                    # return_value= '0+0j'
                    max_num = 10 ** max(1, (pooling_count // amount) + 3)
                    min_num = max_num * -1

                    data_pool = [
                        complex(
                            random.randint(0, min_num + max_num) - min_num,
                            random.randint(0, min_num + max_num) - min_num,
                        )
                        for x in range(max(1, pooling_count))
                    ]
                    if data_pool == []:
                        raise Exception(f"No data pool for complex:{generate_path}")
                elif data_type is bytes:
                    if constraints["min_length"] is not None:
                        min_length = constraints["min_length"]
                    else:
                        min_length = 1
                    if constraints["max_length"] is not None:
                        max_length = constraints["max_length"]
                    else:
                        max_length = min_length + max(1, (pooling_count // amount)) + 3
                    data_pool = [
                        random.randbytes(random.randint(min_length, max_length))
                        for x in range(max(1, pooling_count))
                    ]
                    if data_pool == []:
                        raise Exception(f"No data pool for bytes:{generate_path}")
                else:
                    data_pool = ["error" for x in range(max(1, pooling_count))]
                    if data_pool == []:
                        raise Exception(f"No data pool for default:{generate_path}")
            
            elapsed_time = time.time() - start_time  # end timer
            self.print_path(generate_path,elapsed_time)
            
            self.outputpooling[generate_path] = data_pool
            #print(f"Pool | Time taken: {elapsed_time:.2f} seconds\n")
        return_value = self.outputpooling[generate_path].pop()
        if self.outputpooling[generate_path] == []:
            del self.outputpooling[generate_path]

        return_value = constraints["annotation"](return_value)
        # input("wait...")
        # exit()
        return return_value

    
    def apply_constraints(self, return_value, constraints, match_name, generate_path): #needs test
        ("strip_whitespace",)
        ("to_upper",)
        ("to_lower",)
        ("strict",)  # first 4 are from pydantic StringConstraint aka constr()
        ("default",)
        ("annotation",)
        ("min_length",)
        ("max_length",)
        ("pattern",)
        ("gt",)
        ("lt",)
        ("ge",)
        ("le",)
        ("multiple_of",)
        ("allow_inf_nan",)
        ("max_digits",)
        ("decimal_places",)

        pooling_numbers = list(map(int, re.findall(r"\[(\d+)\]", generate_path)))
        amount = pooling_numbers[0]
        pooling_count = 1
        for x in pooling_numbers:
            pooling_count *= x

        data_type = type(return_value)
        if data_type is str:
            string_list = string.ascii_letters + string.digits
            if constraints["to_upper"] is not None:
                string_list = [item.upper() for item in string_list]
            elif constraints["to_lower"] is not None:
                string_list = [item.lower() for item in string_list]

            if constraints["to_upper"] is not None:
                return_value = return_value.upper()
            elif constraints["to_lower"]:
                return_value = return_value.lower()
            if constraints["min_length"] is not None:
                min_length = constraints["min_length"]
            else:
                min_length = 1
            if constraints["max_length"] is not None:
                max_length = constraints["max_length"]

            if len(return_value) < min_length:
                pad_length = min_length - len(return_value)
                return_value += "".join(random.choices(string_list, k=pad_length))
            if constraints["max_length"] is not None:
                if len(return_value) > max_length:
                    return_value = return_value[:max_length]

        elif data_type is int or data_type is float:
            if constraints["lt"] is not None:
                lt = constraints["lt"] - 1
            else:
                lt = 10 * max(1, (pooling_count // amount) + 3)
            if constraints["le"] is not None:
                lt = min(lt, constraints["le"])

            if constraints["gt"] is not None:
                if data_type is int:
                    gt = constraints["gt"] + 1
                else:
                    if constraints["multiple_of"] is not None:
                        gt = constraints["gt"] + constraints["multiple_of"]
                    else:
                        gt = constraints["gt"] + 0.0001

            else:
                gt = lt * -1
            if constraints["ge"] is not None:
                gt = max(gt, constraints["ge"])

            if gt > lt:
                raise Exception(f"Value for gt:{gt} is greater than value for lt:{lt}")

            if constraints["multiple_of"]:
                multiple_of = constraints["multiple_of"]
                first = ((gt + multiple_of) // multiple_of) * multiple_of

                if data_type is int or (
                    (data_type is float) and constraints["decimal_places"] is None
                ):
                    count = (lt - gt) // multiple_of
                    if count <= 0:
                        raise Exception(
                            f"No multiples of {multiple_of} fit in the range [{gt}, {lt})."
                        )
                    idx = (return_value // multiple_of) % count

                    return_value = first + idx * multiple_of
                else:  # data_type is float and constraints["decimal_places"] is not None
                    return_value = self.generate_from_constraints(
                        match_name, constraints, generate_path
                    )
            else:
                if constraints["decimal_places"] is not None:
                    return_value = round(return_value, constraints["decimal_places"])
                else:
                    try:
                        return_value = int(return_value)
                    except:
                        return_value = self.generate_from_constraints(
                            match_name, constraints, generate_path
                        )

        elif data_type is bool:
            try:
                return_value = bool(return_value)
            except:
                return_value = random.choice([True, False])

        elif data_type is complex:
            try:
                return_value = complex(return_value)
            except:
                return_value = self.generate_from_constraints(
                    match_name, constraints, generate_path
                )

        elif data_type is bytes:
            try:
                return_value = bytes(return_value)
            except:
                return_value = self.generate_from_constraints(
                    match_name, constraints, generate_path
                )
        return_value = constraints["annotation"](return_value)
        return return_value

    
    def generate_synth_data(self, field_name, match_name, applied_constraints, generate_path):
        # print("__")
        #print(f"	Generating for: {field_name}")
        # print(applied_constraints)
        # data_type = get_origin(applied_constraints["annotation"])
        data_type = applied_constraints["origin"]
        output_data = ""
        if data_type == Annotated:
            # print(applied_constraints)
            # print("~~")
            new_info = typing.get_args(applied_constraints["annotation"])
            new_data_type = new_info[0]  # [0] is the data type
            # print(new_data_type)
            constr_constraints = {}

            for attr in all_constr_attribs:
                return_val = getattr(
                    new_info[1], attr, None
                )  # [1] is the data constraints
                constr_constraints[attr] = return_val
            # print(constr_constraints)
            new_constraints = constr_constraints
            new_constraints["required"] = True
            new_constraints["annotation"] = new_data_type
            new_constraints["origin"] = get_origin(new_data_type)
            new_constraints["args"] = get_args(new_data_type)
            # print("___")
            # print(new_constraints)
            #generate_path += ".Annotated"
            output_data = self.generate_synth_data(
                field_name, match_name, new_constraints, generate_path
            )
            # print("returned:",output_data)
            # input("wait...")

        else:
            if not data_type:  # if data_type is None:
                data_type = applied_constraints["annotation"]
            data_args = applied_constraints["args"]
            # data_args = get_args(applied_constraints["annotation"])
            # print(f"	{data_type}\n	{data_args}")
            if data_type is type(None):
                output_data = None
            elif data_type in recursive_types:
                if applied_constraints["min_length"] is not None:
                    min_amount = applied_constraints["min_length"]
                else:
                    min_amount = 1
                if applied_constraints["max_length"] is not None:
                    max_amount = applied_constraints["max_length"]
                else:
                    max_amount = min_amount + 4
                if data_type in [List, list]:
                    output_data = []
                    new_applied_constraints = default_constr_dict.copy()
                    new_applied_constraints.update(applied_constraints)
                    if len(data_args) == 0:
                        data_args = [str]
                    for x in range(random.randint(min_amount, max_amount)):
                        chosen_index = random.randint(0, len(data_args) - 1)
                        chosen_type = data_args[chosen_index]
                        new_applied_constraints["annotation"] = chosen_type
                        new_applied_constraints["origin"] = get_origin(chosen_type)
                        new_applied_constraints["args"] = get_args(chosen_type)
                        list_generate_path = (
                            generate_path + f".List({chosen_index})[{max_amount}]"
                        )
                        output_data.append(
                            self.generate_synth_data(
                                field_name,
                                match_name,
                                new_applied_constraints,
                                list_generate_path
                            )
                        )

                elif data_type in [Dict, dict]:
                    new_left_applied_constraints = default_constr_dict.copy()
                    new_left_applied_constraints.update(applied_constraints)
                    new_right_applied_constraints = default_constr_dict.copy()
                    new_right_applied_constraints.update(applied_constraints)
                    if len(data_args) == 0:
                        data_args = [str, str]
                    chosen_type_left = data_args[0]
                    new_left_applied_constraints["annotation"] = chosen_type_left
                    new_left_applied_constraints["origin"] = get_origin(
                        chosen_type_left
                    )
                    new_left_applied_constraints["args"] = get_args(chosen_type_left)
                    chosen_type_right = data_args[1]
                    new_right_applied_constraints["annotation"] = chosen_type_right
                    new_right_applied_constraints["origin"] = get_origin(
                        chosen_type_right
                    )
                    new_right_applied_constraints["args"] = get_args(chosen_type_right)
                    output_data = {}
                    current_amount = 0
                    target_amount = random.randint(min_amount, max_amount)
                    current_tries = 0
                    target_amountx2 = target_amount * 2
                    # for x in range(random.randint(min_amount,max_amount)):
                    dict_keys = set()
                    while current_amount < target_amount:
                        generate_path_v1 = generate_path + f".Dict(Left)[{max_amount}]"
                        v1 = self.generate_synth_data(
                            field_name,
                            match_name,
                            new_left_applied_constraints,
                            generate_path_v1
                        )

                        dict_keys.add(v1)
                        current_amount = len(dict_keys)
                        current_tries += 1
                        if current_tries > target_amountx2:
                            if min_amount != 1:
                                raise Exception(
                                    f"Not enough provider keys for {field_name}, \nkeys: {dict_keys}"
                                )
                            else:
                                break
                    for key in dict_keys:
                        generate_path_v2 = generate_path + f".Dict(Right)[{max_amount}]"
                        v2 = self.generate_synth_data(
                            field_name,
                            match_name,
                            new_right_applied_constraints,
                            generate_path_v2
                        )
                        output_data[key] = v2

                elif data_type in [Tuple, tuple]:
                    output_data = []
                    new_applied_constraints = default_constr_dict.copy()
                    new_applied_constraints.update(applied_constraints)
                    if len(data_args) == 0:
                        data_args = [str for x in range(random.randint(min_amount, max_amount))]
                    for x in range(len(data_args)):
                        chosen_type = data_args[x]
                        new_applied_constraints["annotation"] = chosen_type
                        new_applied_constraints["origin"] = get_origin(chosen_type)
                        new_applied_constraints["args"] = get_args(chosen_type)
                        tuple_generate_path = (
                            generate_path + f".Tuple({x})[{max_amount}]"
                        )
                        output_data.append(
                            self.generate_synth_data(
                                field_name,
                                match_name,
                                new_applied_constraints,
                                tuple_generate_path
                            )
                        )

                elif data_type in [Set, set, frozenset]:
                    output_data = []
                    new_applied_constraints = default_constr_dict.copy()
                    new_applied_constraints.update(applied_constraints)
                    current_amount = 0
                    target_amount = random.randint(min_amount, max_amount)
                    if len(data_args) == 0:
                        data_args = [str]
                    chosen_type = data_args[0]
                    new_applied_constraints["annotation"] = chosen_type
                    new_applied_constraints["origin"] = get_origin(chosen_type)
                    new_applied_constraints["args"] = get_args(chosen_type)
                    current_tries = 0
                    target_amountx2 = target_amount * 2
                    if data_type is frozenset:
                        pathstr = "FrozenSet"
                    else:
                        pathstr = "Set"
                    while current_amount < target_amount:
                        set_generate_path = (
                            generate_path + f".{pathstr}({0})[{max_amount}]"
                        )
                        output_data.append(
                            self.generate_synth_data(
                                field_name,
                                match_name,
                                new_applied_constraints,
                                set_generate_path
                            )
                        )
                        current_amount = len(output_data)
                        current_tries += 1
                        if current_tries > target_amountx2:
                            if min_amount != 1:
                                raise Exception(
                                    f"Not enough provider keys for {field_name}"
                                )
                            else:
                                break
                    if data_type is frozenset:
                        output_data = frozenset(output_data)

                elif data_type == Union:
                    new_applied_constraints = default_constr_dict.copy()
                    new_applied_constraints.update(applied_constraints)
                    chosen_index = random.randint(0, len(data_args) - 1)
                    chosen_type = data_args[chosen_index]
                    new_applied_constraints["annotation"] = chosen_type
                    new_applied_constraints["origin"] = get_origin(chosen_type)
                    new_applied_constraints["args"] = get_args(chosen_type)
                    generate_path += f".Union({chosen_index})"
                    output_data = self.generate_synth_data(
                        field_name, match_name, new_applied_constraints, generate_path
                    )

                elif data_type == Optional:
                    # should use Union, but still here for fallback
                    new_applied_constraints = default_constr_dict.copy()
                    new_applied_constraints.update(applied_constraints)
                    data_args.extend(None)
                    chosen_index = random.randint(0, len(data_args) - 1)
                    chosen_type = data_args[chosen_index]
                    if chosen_type is None:
                        output_data = None
                    else:
                        new_applied_constraints["annotation"] = chosen_type
                        new_applied_constraints["origin"] = get_origin(chosen_type)
                        new_applied_constraints["args"] = get_args(chosen_type)
                        generate_path += f".Optional({chosen_index})"
                        output_data = self.generate_synth_data(
                            field_name,
                            match_name,
                            new_applied_constraints,
                            generate_path
                        )

                elif data_type == Literal:
                    # print("Literal Here")
                    # print(data_args)
                    output_data = random.choice(data_args)
                    # print(output_data)

                else:
                    raise Exception(
                        f"Recersive data type| {data_type} : {data_args} |not handled"
                    )
                # print("	recursive")
            elif data_type in python_builtin_types:
                # print("	generate data")
                func = self.resolved_methods.get(match_name)
                # print(applied_constraints["pattern"])
                # print(func)
                # print(match_name)
                if applied_constraints["pattern"] or not func or match_name == "":
                    # print("generating,",field_name)
                    # print(inspect.getsource(self.generate_from_constraints))
                    output_data = self.generate_from_constraints(
                        field_name, applied_constraints, generate_path
                    )
                else:
                    if (generate_path not in self.outputpooling or 
                        (
                            generate_path in self.outputpooling
                            and self.outputpooling[generate_path] == []
                        )
                    ):
                        pooling_numbers = list(map(int, re.findall(r"\[(\d+)\]", generate_path)))
                        pooling_count = 1
                        for x in pooling_numbers:
                            pooling_count *= x
                        
                        start_time = time.time()
                        
                        data_pool = [ self.apply_constraints(func(), applied_constraints, match_name, generate_path) for _ in range(pooling_count) ]
                        #can be used better for uniqueness and bulking later
                        
                        elapsed_time = time.time() - start_time
                        
                        self.print_path(generate_path,elapsed_time)
                        
                        self.outputpooling[generate_path] = data_pool
                    output_data = self.outputpooling[generate_path].pop()

            else:
                if (
                    data_type.__class__.__module__
                    == "pydantic._internal._model_construction"  # checking if its a model
                    and inspect.isclass(data_type)
                    and issubclass(data_type, BaseModel)  
                    # extra checks to make sure its a BaseModel
                ):
                    output_data = self.synthesise_recursive(
                        data_type, self.method, amount=1, path=generate_path+"."
                    )
                    #print("big nested")

                else:
                    raise Exception(
                        f"Unkown data type ({data_type}) for field {field_name}"
                    )

        # print(f"Data: {output_data}")
        # print("__")

        # apply constraints of output after data is provided

        return output_data

    
    def make_applied_constraints(self, schema_model):
        applied_constraints = {}

        for name, field in schema_model.model_fields.items():
            applied_constraints[name] = self.check_generation_constraints(
                name, field
            )

        return applied_constraints
    
    def recursive_make_schema_applied_constraints(self, schema_model): #needs test
        applied_constraints = {}
        model_data = self.get_model_data(schema_model)
        
        applied_constraints[schema_model.__name__] = self.make_applied_constraints(schema_model)
        
        for x in model_data:
            data_type = x[1].annotation
            if (
                inspect.isclass(data_type)
                and issubclass(data_type, BaseModel)
                and data_type.__name__ not in applied_constraints.keys()
                ):
                applied_constraints.update(self.recursive_make_schema_applied_constraints(data_type))
            
        
        return applied_constraints
        
    
    def make_resolved_methods(self, name_match_pairs, methods_map):
        resolved_methods = {}
        for match_name in name_match_pairs:
            if match_name != "":
                provider_instance = methods_map[match_name]
                resolved_methods[match_name] = getattr(provider_instance, match_name)
            else:
                resolved_methods[match_name] = None
        #print("r",resolved_methods)
        return resolved_methods

    
    def synthesise_recursive(self, schema_model, method="faker", amount=1, path=""):
        
        schema_name = schema_model.__name__
        
        synthesised_data = {}
        # print("__")
        for name in schema_model.model_fields.keys():
            # print(f"Field:{name}")
            if not self.applied_constraints[schema_name][name]["required"]:
                if random.randint(1, 2) == 1:
                    continue
            
            generate_path = path+f"{schema_model.__name__}({name})[{amount}]"
            
            synthesised_data[name] = self.generate_synth_data(
                name,
                self.field_match_pairs[schema_name][name],
                self.applied_constraints[schema_name][name],
                generate_path
            )
            # print(f"	Data:{synthesised_data[name]}")
        # print("__")
        return synthesised_data
    
    
    def synthesise(self, schema_model, method="faker", amount=1):
        
        if amount == 0:
            return []
        
        self.method = method
        
        self.field_match_pairs = self.recursive_match_schema_fields(schema_model)
        
        self.applied_constraints = self.recursive_make_schema_applied_constraints(schema_model)
        #print(self.applied_constraints)
        dataset = []
        for x in range(amount):
            
            synthesised_data = self.synthesise_recursive(schema_model, method, amount)
            
            dataset.append(schema_model(**synthesised_data))
            if (x+1)%max(1,min(100,amount//10))==0:
                print(f"Completed: {x+1}/{amount}{' '*30}",end="\r")
        print(f"Completed: {amount}/{amount}{' '*30}")
        return dataset
#'''
