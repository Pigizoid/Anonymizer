from dataclasses import dataclass, field
from typing import Any


@dataclass
class JsonSchemaClass:
    name: str
    contents: Any
    defs: dict = field(init=False)

    def __post_init__(self):
        self.__name__ = self.name
        self.required = self.contents["required"]
        self.properties = self.contents["properties"]
        self.fields = self.properties
        for name,field in self.fields.items():
            if name in self.required:
                self.fields[name]["required"] = True
            else:
                self.fields[name]["required"] = False
        if "$defs" in self.contents:
            defs = {}
            for name,content in self.contents["$defs"].items():
                defs[name] = JsonSchemaClass(name,content)
            self.defs = defs
        else:
            self.defs = {}