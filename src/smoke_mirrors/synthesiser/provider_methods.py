from typing import Tuple, Dict, List, Any
import inspect
from faker import Faker
import mimesis
from smoke_mirrors.pre_made_data import provider_methods


fake = Faker()
generic = mimesis.Generic(mimesis.locales.Locale.EN)

def list_faker_methods(method_list:Dict[str,Any]=None) -> Tuple[list, dict]:
    methods = []
    methods_map = {}
    for attr in dir(fake):
        try:
            getattr(fake, attr)
        except:
            continue
        if method_list == None:
            if not attr.startswith("_") and attr.lower() == attr:
                methods.append(attr)
                methods_map[attr] = fake
        elif attr in method_list:
            methods.append(attr)
            methods_map[attr] = fake
    return (methods, methods_map)

def list_mimesis_methods(method_list:Dict[str,Any]=None) -> Tuple[list, dict]:
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
                try:
                    getattr(instance, attr)
                except:
                    continue
                if method_list == None:
                    if not attr.startswith("_") and attr.lower() == attr:
                        methods.append(attr)
                        methods_map[attr] = instance
                elif attr in method_list:
                    methods.append(attr)
                    methods_map[attr] = instance
    return (methods, methods_map)

def list_match_methods(method:str) -> Tuple[list, dict]:
    """
    Inputs:\n
        method of methods "faker","mimesis","mixed"
    method for listing all generation providers in method\n
    Outputs:\n
        tuple of (methods,methods_map):
            methods_map = {provider_name, parent callable}
    """
    methods = []
    methods_map = {}
    methods_dict = {
        "faker":["faker"],
        "mimesis":["mimesis"],
        "mixed":["faker","mimesis"]
    }
    method_funcs_dict = {
        "faker":list_faker_methods,
        "mimesis":list_mimesis_methods
    }
    if method not in methods_dict:
        raise Exception(f"Unexpected method: {method}")
    methods_list = methods_dict[method]
    methods = []
    methods_map = {}

    for m in methods_list:
        method_func = method_funcs_dict[m]
        returned_methods, returned_methods_map = method_func(provider_methods[m]["word_list"])

        methods.extend(returned_methods)
        methods_map.update(returned_methods_map)

    methods = list(set(methods))
    return (methods, methods_map)

def make_resolved_methods(name_matches:List[str], methods_map:Dict[str,Any]) -> Dict[str, Any]:
    """
    Inputs:\n
        list of provider names
        dict of {name:provider parent callables}
    instanciates each parent callable\n
    Outputs:\n
        dict of {name:instance}
    """
    resolved_methods = {}
    for match_name in name_matches:
        if match_name != "":
            provider_instance = methods_map[match_name]
            resolved_methods[match_name] = getattr(provider_instance, match_name)
        else:
            resolved_methods[match_name] = None
    # print("r",resolved_methods)
    return resolved_methods
