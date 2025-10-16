from dataclasses import dataclass, field
from typing import Any, Dict, List
import copy
from decimal import Decimal


def floating_point_sanitize_schema(schema_contents):
    if isinstance(schema_contents, float):
        return Decimal(str(schema_contents))
    elif isinstance(schema_contents, dict):
        return {
            k: floating_point_sanitize_schema(v) for k, v in schema_contents.items()
        }
    elif isinstance(schema_contents, list):
        return [floating_point_sanitize_schema(item) for item in schema_contents]
    elif isinstance(schema_contents, tuple):
        return tuple(floating_point_sanitize_schema(item) for item in schema_contents)
    else:
        return schema_contents


@dataclass
class JsonSchemaClass:
    contents: Dict[str, Any]
    name: str = None
    defs: dict = field(init=False)

    def __post_init__(self):
        if "title" in self.contents:
            self.__name__ = self.contents["title"]
        else:
            self.__name__ = self.name
        self.sanitised_contents = floating_point_sanitize_schema(self.contents)
        if self.contents["type"] != "object":
            self.required = []
            self.properties = self.contents
        else:
            self.required = self.contents["required"]
            self.properties = self.contents["properties"]
        self.fields = copy.deepcopy(self.properties)
        if "$defs" in self.contents:
            defs = {}
            for name, content in self.contents["$defs"].items():
                defs[name] = JsonSchemaClass(content, name=name)
            defs[self.__name__] = self
            self.defs = defs
        else:
            self.defs = {}
        if self.contents["type"] != "object":
            self.fields["required"] = True
        else:
            for name, field in self.fields.items():
                if name in self.required:
                    self.fields[name]["required"] = True
                else:
                    self.fields[name]["required"] = False

    def new_subset_model(
        self,
        fields: List[str],
        name: str,
    ):
        new_model = copy.deepcopy(self)
        new_model.__name__ = name
        properties = {
            field_name: field
            for field_name, field in new_model.contents["properties"].items()
            if field_name in fields
        }
        required = {
            field_name for field_name in new_model.required if field_name in fields
        }
        new_fields = {
            field_name: field
            for field_name, field in new_model.fields.items()
            if field_name in fields
        }
        new_model.contents["properties"] = properties
        new_model.required = required
        new_model.fields = new_fields

        return new_model
