from pydantic import BaseModel
from typing import Dict, Any, List
import random
from .initialiser.class_init import init_class
from .initialiser.get_constraints import get_constraints_class
from .initialiser.match_methods import match_methods_class
from .field_match.match import match_class
from .generator.constraint_based_generator import constraint_based_generator_class
from .generator.synth_based_generator import synth_based_generator_class

from sm.tools.model_funcs import get_model_fields

"""
The main synthesiser class
pulls from sub classes to buidl full functionality
"""


class Synthesiser(
    init_class,
    get_constraints_class,
    match_methods_class,
    match_class,
    constraint_based_generator_class,
    synth_based_generator_class,
):
    def synthesise_recursive(
        self, schema_model, method="faker", amount=1, path=""
    ) -> Dict[str, Any]:
        """
        The main recursive call of the synthesiser class
        Used internally by generate_synth_data for nested schema models
        Inputs:
            schema model
            method
            amount
            generation path
        """
        schema_name = schema_model.__name__
        synthesised_data = {}
        # print("__")
        for name in get_model_fields(schema_model).keys():
            # print(f"Field:{name}")
            if not self.applied_constraints[schema_name][name]["required"]:
                if random.randint(1, 2) == 1:
                    continue

            generate_path = path + f"{schema_model.__name__}({name})[{amount}]"
            # self.apply_constraints(func(), applied_constraints, match_name, generate_path)
            synthesised_data[name] = self.generate_synth_data(
                name,
                self.field_match_pairs[schema_name][name],
                self.applied_constraints[schema_name][name],
                generate_path,
            )
            # print(f"	Data:{synthesised_data[name]}")
        # print("__")
        return synthesised_data

    def synthesise(
        self, schema_model, method="faker", amount=1, seed="random"
    ) -> List[BaseModel]:
        """
        The main call function of the synthesiser class
        Inputs:
            schema model
            method of methods "faker","mimesis","mixed"
            amount of returned data to generate as int
            data seed as either int or as "random","relational"
        calls recursive synthesis on schema model after initial setup
        Ouputs:
            list of pydantic BaseModel with synthesised data
            [BaseModel]*amount
        """
        if amount == 0:
            return []

        if seed == "random":
            self.input_seed = "random"
            self.seed = random.randint(0, 1_000_000_000_000)
            random.seed(seed)
        elif seed == "relational":  # not done
            self.input_seed = seed
            self.seed = random.randint(0, 1_000_000_000_000)
            random.seed(seed)
        else:
            try:
                seed = int(seed)
                self.input_seed = seed
                random.seed(seed)
            except:
                self.input_seed = seed
                self.seed = random.randint(0, 1_000_000_000_000)
                random.seed(seed)

        self.method = method

        self.field_match_pairs = self.recursive_match_fields(schema_model)
        self.applied_constraints = self.recursive_get_applied_constraints(schema_model)
        # print(self.applied_constraints)
        dataset = []
        for x in range(amount):
            synthesised_data = self.synthesise_recursive(
                schema_model, method=method, amount=amount
            )

            dataset.append(schema_model(**synthesised_data))
            if (x + 1) % max(1, amount // 100) == 0:  # 1% at a time
                print(
                    f"Completed: {x + 1}/{amount}:{round(((x + 1) / amount) * 100, 2)}%{' ' * 30}",
                    end="\r",
                )
        print(f"Completed: {amount}/{amount}{' ' * 30}")
        return dataset
