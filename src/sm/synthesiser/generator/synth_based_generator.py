from pydantic import BaseModel
from typing import (
    List,
    Dict,
    Tuple,
    Set,
    Union,
    Optional,
    Literal,
    Any,
    Annotated,
    get_args,
    get_origin,
)
import random
import re
import time
import inspect
from ..misc import print_path

from sm.pre_made_data import (
    all_constr_attribs,
    default_constr_dict,
    recursive_types,
    python_builtin_types,
)


"""
Generation path:
    format of  'Schema_name(field_name)[amount/length]'
    format of  'n(f)[a].Type(index)[amount/length]'
    e.g.
    1.
        class User():
            address: (_,_,_)
        User(address)[1].Tuple(0)[1]
        -> would be generating a data pool for
        class User():
            address: (X,_,_)
    2.
        class Address():
            street : str
        class User():
            nested : Address
        User(nested)[100].Address(street)[1]
    3.
        class User():
            address: List[str,int]
        User(address)[1].List(0)[10]
        -> means
            [str]*10
        
        User(address)[1].List(1)[20]
        -> means
            [int]*20
"""


class synth_based_generator_class:
    def make_new_contraints(self, applied_constraints):
        """
        Inputs a set of applied constraints and updates with defaults
        fast method for code reusage
        """
        new_applied_constraints = default_constr_dict.copy()
        new_applied_constraints.update(applied_constraints)
        return new_applied_constraints

    def generate_synth_data(
        self, field_name, match_name, applied_constraints, generate_path
    ) -> Any:  # value or collection
        """
        Inputs:
            field name
            match name
            constraints
            generate path

        Recursive function that crawls through the constraints structure
        Handles collection types and model recursion
        Returns a python structure of the constraints format with fully generated data
        """
        # print("__")
        # print(f"	Generating for: {field_name}")
        # print(applied_constraints)
        # data_type = get_origin(applied_constraints["annotation"])
        data_type = applied_constraints["origin"]
        output_data = ""
        if data_type == Annotated:
            # print(applied_constraints)
            # print("~~")
            new_info = get_args(applied_constraints["annotation"])
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
            generate_path += ".Annotated"
            output_data = self.generate_synth_data(
                field_name, match_name, new_constraints, generate_path
            )
            # print("returned:",output_data)

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

                new_applied_constraints = self.make_new_contraints(applied_constraints)

                if data_type in [List, list]:
                    output_data = []
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
                                list_generate_path,
                            )
                        )

                elif data_type in [Dict, dict]:
                    new_left_applied_constraints = new_applied_constraints.copy()
                    new_right_applied_constraints = new_applied_constraints.copy()
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
                            generate_path_v1,
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
                            generate_path_v2,
                        )
                        output_data[key] = v2

                elif data_type in [Tuple, tuple]:
                    output_data = []
                    if len(data_args) == 0:
                        data_args = [
                            str for x in range(random.randint(min_amount, max_amount))
                        ]
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
                                tuple_generate_path,
                            )
                        )

                elif data_type in [Set, set, frozenset]:
                    output_data = []
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
                                set_generate_path,
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
                            generate_path,
                        )

                elif data_type == Literal:
                    output_data = random.choice(data_args)

                else:
                    raise Exception(
                        f"Recersive data type| {data_type} : {data_args} |not handled"
                    )
            elif data_type in python_builtin_types:
                if match_name == "":
                    func = None
                else:
                    func = self.resolved_methods[match_name]
                if applied_constraints["pattern"] or not func or match_name == "":
                    output_data = self.generate_from_constraints(
                        field_name, applied_constraints, generate_path
                    )
                else:
                    if generate_path not in self.outputpooling or (
                        generate_path in self.outputpooling
                        and self.outputpooling[generate_path] == []
                    ):
                        pooling_numbers = list(
                            map(int, re.findall(r"\[(\d+)\]", generate_path))
                        )
                        pooling_count = 1
                        for x in pooling_numbers:
                            pooling_count *= x

                        start_time = time.time()
                        data_temp_pool = [func() for _ in range(pooling_count)]

                        data_pool = [
                            self.apply_constraints(
                                func_val,
                                applied_constraints,
                                match_name,
                                generate_path,
                                pooling_count,
                                pooling_numbers[0],
                            )
                            for func_val in data_temp_pool
                        ]

                        elapsed_time = time.time() - start_time

                        print_path(generate_path, elapsed_time)

                        self.outputpooling[generate_path] = data_pool
                    output_data = self.outputpooling[generate_path].pop()

            else:
                if inspect.isclass(data_type) and (
                    issubclass(data_type, BaseModel) or isinstance(data_type, BaseModel)
                ):
                    output_data = self.synthesise_recursive(
                        data_type, self.method, amount=1, path=generate_path + "."
                    )
                    # print("big nested")

                else:
                    raise Exception(
                        f"Unkown data type ({data_type}) for field {field_name}"
                    )
        # print(f"Data: {output_data}")
        # print("__")
        # apply constraints of output after data is provided
        return output_data

    def generate_single_value(self, field_name, field_type):
        """
        Inputs field_name and field_type and outputs a single generated value
        by calling internal functions
        """
        matched_field = self.match_fields([field_name])
        if matched_field[field_name] != "":
            func = self.resolved_methods[matched_field[field_name]]
            value = func()
        else:
            applied_constraints = default_constr_dict.copy()
            applied_constraints["annotation"] = field_type
            value = self.generate_from_constraints(
                field_name, applied_constraints, "self()[1].generate"
            )
        return value
