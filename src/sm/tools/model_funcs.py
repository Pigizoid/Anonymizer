from pydantic import BaseModel
from pydantic.fields import FieldInfo

from typing import Union, Type, Dict, Any

ModelLike = Union[Type[BaseModel], BaseModel]

def get_model_fields(schema_model: ModelLike) -> Dict[str, Any]:
    
    if isinstance(schema_model, type) and issubclass(schema_model, BaseModel):
        model_cls = schema_model
        instance = None
    elif isinstance(schema_model, BaseModel):
        model_cls = type(schema_model)
        instance = schema_model
    else:
        raise TypeError("Schema must be a pydantic BaseModel class or instance")
    
    model_fields: Dict[str, FieldInfo] = getattr(model_cls, "model_fields", {})
    return model_fields