from typing import Tuple


from faker import Faker
import mimesis
from mimesis import Generic
import inspect
from smoke_mirrors.synthesiser.custom_provider_methods import CustomProviders


def generate_provider_return_types(provider_names, provider_instances):
    return_types = {}

    for name in sorted(provider_names):
        try:
            values = [
                getattr(provider_instances[name], name, None)() for _ in range(31)
            ]
        except Exception:
            continue
        types_list = []
        for value in values:
            if value is None:
                types_list.append(type(None))
                continue
            try:
                int(value)
                types_list.append(int)
                continue
            except (TypeError, ValueError):
                pass

            try:
                float(value)
                types_list.append(float)
                continue
            except (TypeError, ValueError):
                pass

            if isinstance(value, str) and value.lower() in {"true", "false"}:
                return_type = bool
            elif isinstance(value, bytes):
                return_type = bytes
            elif isinstance(value, tuple):
                return_type = tuple
            elif isinstance(value, list):
                return_type = list
            elif isinstance(value, set):
                return_type = set
            elif isinstance(value, frozenset):
                return_type = frozenset
            elif isinstance(value, dict):
                return_type = dict
            else:
                return_type = str

            types_list.append(return_type)

        available_types = [
            type(None),
            int,
            float,
            bool,
            bytes,
            tuple,
            list,
            set,
            frozenset,
            dict,
            str,
        ]
        for selected_type in reversed(available_types):
            if selected_type in types_list:
                maximal_type = selected_type
                break
        return_types[name] = maximal_type

    return dict(sorted(return_types.items()))


fake = Faker()
fake.add_provider(CustomProviders)
generic = Generic(mimesis.locales.Locale.EN)


def list_faker_methods() -> Tuple[list, dict]:
    methods = []
    methods_map = {}
    for attr in dir(fake):
        try:  # this is used to ensure the providers dont error when called
            if (
                not attr.startswith("_")
                and callable(getattr(fake, attr))
                and attr.lower() == attr
            ):
                methods.append(attr)
                methods_map[attr] = fake
        except Exception:
            pass
    return (methods, methods_map)


def list_mimesis_methods() -> Tuple[list, dict]:
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
                if (
                    not attr.startswith("_")
                    and callable(getattr(instance, attr))
                    and attr.lower() == attr
                ):
                    methods.append(attr)
                    methods_map[attr] = instance
    return (methods, methods_map)


def list_match_methods(method) -> Tuple[list, dict]:
    methods = []
    methods_map = {}
    if method == "faker":
        methods, methods_map = list_faker_methods()
    elif method == "mimesis":
        methods, methods_map = list_mimesis_methods()
    elif method == "mixed":
        methodsF, methods_mapF = list_faker_methods()

        methodsM, methods_mapM = list_mimesis_methods()

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


def setup_func():
    with open("pre_made_data.py", "w+") as f:
        f.write("""from typing import List, Dict, Tuple, Set, Union, Literal, Optional, Any

from decimal import Decimal

from faker import Faker
import mimesis
from mimesis import Generic
import inspect
                
                \n""")
        f.write("provider_return_types = {\n")
        names, instances = list_match_methods("mixed")
        p_types = generate_provider_return_types(names, instances)
        for name, v in p_types.items():
            if v.__name__ == "NoneType":
                f.write(f"    '{name}' : None,\n")
            else:
                f.write(f"    '{name}' : {v.__name__},\n")
        f.write("}\n")

        methods = ["faker", "mimesis", "mixed"]
        f.write("provider_methods = {\n")
        for method in methods:
            f.write(f"    '{method}' : {{\n")
            word_list, _ = list_match_methods(method)
            word_list = [word for word in word_list if word in p_types.keys()]
            f.write(f"        'word_list': {word_list},\n")
            word_tokens = {
                word: word.split("_") for word in word_list if word in p_types.keys()
            }
            f.write(f"        'word_tokens': {word_tokens},\n")
            word_tokens_set = {
                word: set(word.split("_"))
                for word in word_list
                if word in p_types.keys()
            }
            f.write(f"        'word_tokens_set': {word_tokens_set}\n")
            f.write("    },\n")
        f.write("}\n")
        f.write("""
python_builtin_types = {
    str,
    int,
    float,
    Decimal,
    bool,
    complex,
    bytes,
    tuple,
    list,
    set,
    frozenset,
    dict,
}
typing_origins = {List, Dict, Tuple, Set, Union, Literal, Optional}
recursive_types = {
    List,
    Dict,
    Tuple,
    Set,
    Union,
    Literal,
    Optional,
    list,
    dict,
    tuple,
    set,
    frozenset,
}
all_constr_attribs = {
    "default",
    "annotation",
    "min_length",
    "max_length",
    "pattern",
    "gt",
    "lt",
    "ge",
    "le",
    "multiple_of",
}

default_constr_dict = {
    "required": True,
    "default": None,
    "annotation": None,
    "min_length": None,
    "max_length": None,
    "pattern": None,
    "gt": None,
    "lt": None,
    "ge": None,
    "le": None,
    "multiple_of": None,
    "origin": None,
    "args": None,
}
        """)
        print("created data")


if __name__ == "__main__":
    import os
    import pathlib

    test_dir = pathlib.Path(__file__).resolve().parent
    os.chdir(test_dir)
    print("Working dir:", os.getcwd())
    # os.chrdir(os.getcwd()+"\\")
    setup_func()
