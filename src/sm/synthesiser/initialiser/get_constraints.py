from pydantic import BaseModel
from typing import Dict, Any
import inspect
from sm.tools.model_funcs import get_model_data, get_model_fields


class get_constraints_class:
    def get_applied_constraints(self, schema_model) -> Dict[str, Dict[str, Any]]:
        """
        Gets constraints for all fields in a schema model and returns and returns a dict of {name:constraints}
        """
        applied_constraints = {}

        for name, field in get_model_fields(schema_model).items():
            applied_constraints[name] = self.check_generation_constraints(name, field)

        return applied_constraints

    def recursive_get_applied_constraints(
        self, schema_model
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:  # needs test
        """
        Recursive function to get applied constraint of input schema and all nested schemas
        """
        applied_constraints = {}
        model_data = get_model_data(schema_model)

        applied_constraints[schema_model.__name__] = self.get_applied_constraints(
            schema_model
        )

        for x in model_data:
            data_type = x[1].annotation
            if (
                inspect.isclass(data_type)
                and issubclass(data_type, BaseModel)
                and data_type.__name__ not in applied_constraints.keys()
            ):
                applied_constraints.update(
                    self.recursive_get_applied_constraints(data_type)
                )

        return applied_constraints
