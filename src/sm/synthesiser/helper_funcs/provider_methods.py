from typing import Tuple, Dict, List, Any
import inspect
from faker import Faker
import mimesis


fake = Faker()
generic = mimesis.Generic(mimesis.locales.Locale.EN)

def list_faker_methods() -> Tuple[list, dict]:
    """
    method for listing all generation providers in Faker\n
    Outputs:\n
        tuple of (methods,methods_map):
            methods_map = {provider_name, parent callable}
    """
    methods = []
    methods_map = {}
    for attr in dir(fake):
        try:  # this is used to ensure the providers dont error when called
            if not attr.startswith("_") and callable(getattr(fake, attr)):
                methods.append(attr)
                methods_map[attr] = fake
        except:
            pass
    return (methods, methods_map)

def list_mimesis_methods() -> Tuple[list, dict]:
    """
    method for listing all generation providers in mimesis\n
    Outputs:\n
        tuple of (methods,methods_map):
            methods_map = {provider_name, parent callable}
    """
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
