
from typing import Dict, Any, get_origin, get_args
from decimal import Decimal, ROUND_HALF_UP
from multiprocessing import Pool
from functools import partial
import exrex
import string
import re
import time
import random
import rstr
from ..misc import print_path

from sm.pre_made_data import all_constr_attribs
from sm.tools.regex_generator import regex_builder


def make_one_string(pattern):
    return exrex.getone(pattern)


def make_one_decimal(x, decimal_precision, min_scaled, scaled_mult, scale):
    potential_val = (min_scaled + (x + 1) * scaled_mult) / scale
    if potential_val == potential_val.quantize(
        decimal_precision, rounding=ROUND_HALF_UP
    ):
        return potential_val
    else:
        return None


string_list = string.ascii_letters + string.digits


class constraint_based_generator_class():
    def check_generation_constraints(self, name, field) -> Dict[str, Any]:
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

    def generate_from_constraints(
        self, field_name, constraints, generate_path
    ) -> Any:  # value
        # print(f"Name:{field_name}\nFields:{constraints}")
        # print("__")

        if generate_path not in self.outputpooling or (
            generate_path in self.outputpooling
            and self.outputpooling[generate_path] == []
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
                # print(self.make_one_string(pattern))
                PERFORMANCE = True  # risky but faster
                try:
                    if max(1, pooling_count) > 10000 and PERFORMANCE:
                        src = regex_builder.compile_regex_to_function_source(pattern)
                        safe_builtins = {
                            "len": len, "range": range, "min": min, "max": max,
                            "list": list, "chr": chr, "ord": ord, "set": set, "map": map,
                        }
                        env = {
                            "__builtins__": safe_builtins, #dissallow anything except random and string
                            "random": random,
                            "string": string,
                        }
                        exec(src, env)
                        gen = env["gen"]
                        """
                        with Pool() as p:
                            data_pool = p.map(
                                _worker_compile_and_gen, 
                                [(pattern, i) for i in range(pooling_count)]
                            )
                        #"""
                        data_pool = [gen() for _ in range(max(1, pooling_count))]
                    elif max(1, pooling_count) > 100:
                        with Pool() as p:
                            data_pool = p.map(
                                make_one_string, [pattern] * max(1, pooling_count)
                            )
                    else:
                        data_pool = [
                            exrex.getone(pattern) for x in range(max(1, pooling_count))
                        ]
                except Exception as e:
                    print(f"failed reg pool and used fallback: {e}")
                    data_pool = [
                        rstr.xeger(pattern) for x in range(max(1, pooling_count))
                    ]

                if constraints["to_upper"] is not None:
                    data_pool = [item.upper() for item in data_pool]
                elif constraints["to_lower"] is not None:
                    data_pool = [item.lower() for item in data_pool]
                if data_pool == []:
                    raise Exception(f"No data pool for {constraints['pattern']}")
            else:
                # print("Generate from annotation",str(constraints["annotation"]))
                data_type = constraints["annotation"]
                if data_type is str:
                    temp_string_list = string_list
                    if constraints["to_upper"] is not None:
                        temp_string_list = [item.upper() for item in temp_string_list]
                    elif constraints["to_lower"] is not None:
                        temp_string_list = [item.lower() for item in temp_string_list]
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
                                temp_string_list,
                                k=(random.randint(min_length, max_length)),
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
                                if max(1, pooling_count) > 100:
                                    with Pool() as p:
                                        func = partial(
                                            make_one_decimal,
                                            decimal_precision=decimal_precision,
                                            min_scaled=min_scaled,
                                            scaled_mult=scaled_mult,
                                            scale=scale,
                                        )
                                        data_pool = p.map(
                                            func,
                                            range(
                                                int(
                                                    (lt * scale - min_scaled)
                                                    // scaled_mult
                                                )
                                                - 1
                                            ),
                                        )
                                    data_pool = [d for d in data_pool if d is not None]
                                else:
                                    for x in range(
                                        int((lt * scale - min_scaled) // scaled_mult)
                                        - 1
                                    ):
                                        potential_val = (
                                            min_scaled + (x + 1) * scaled_mult
                                        ) / scale
                                        if potential_val == potential_val.quantize(
                                            decimal_precision, rounding=ROUND_HALF_UP
                                        ):
                                            data_pool.append(potential_val)
                            except Exception:
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

                    # if the data pool doesnt have enough elements, extend it with copies of existing elements
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
            print_path(generate_path, elapsed_time)

            self.outputpooling[generate_path] = data_pool
            # print(f"Pool | Time taken: {elapsed_time:.2f} seconds\n")
        return_value = self.outputpooling[generate_path].pop()
        if self.outputpooling[generate_path] == []:
            del self.outputpooling[generate_path]

        return_value = constraints["annotation"](return_value)
        # input("wait...")
        # exit()
        return return_value

    def apply_constraints(
        self,
        return_value,
        constraints,
        match_name,
        generate_path,
        pooling_count,
        amount,
    ) -> Any:  # value   # needs test
        data_type = type(return_value)
        if data_type is str:
            # print("\tconstraining str")
            temp_string_list = string_list
            if constraints["to_upper"] is not None:
                temp_string_list = [item.upper() for item in string_list]
            elif constraints["to_lower"] is not None:
                temp_string_list = [item.lower() for item in string_list]

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
                return_value += "".join(random.choices(temp_string_list, k=pad_length))
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
        return constraints["annotation"](return_value)

