from dataclasses import dataclass, field
from typing import Any
import copy
from decimal import Decimal



def floating_point_sanitize_schema(schema_contents):
    if isinstance(schema_contents, float):
        return Decimal(str(schema_contents))
    elif isinstance(schema_contents, dict):
        return {k: floating_point_sanitize_schema(v) for k, v in schema_contents.items()}
    elif isinstance(schema_contents, list):
        return [floating_point_sanitize_schema(item) for item in schema_contents]
    elif isinstance(schema_contents, tuple):
        return tuple(floating_point_sanitize_schema(item) for item in schema_contents)
    else:
        return schema_contents

@dataclass
class JsonSchemaClass:
    name: str
    contents: Any
    defs: dict = field(init=False)

    def __post_init__(self):
        self.__name__ = self.name
        self.contents = floating_point_sanitize_schema(self.contents)
        self.required = self.contents["required"]
        self.properties = self.contents["properties"]
        self.fields = copy.deepcopy(self.properties)
        if "$defs" in self.contents:
            defs = {}
            for name,content in self.contents["$defs"].items():
                defs[name] = JsonSchemaClass(name,content)
            self.defs = defs
        else:
            self.defs = {}
        for name,field in self.fields.items():
            if name in self.required:
                self.fields[name]["required"] = True
            else:
                self.fields[name]["required"] = False
