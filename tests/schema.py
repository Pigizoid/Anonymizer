from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Set, Literal, Annotated
import decimal

from pydantic import constr


class USER(BaseModel):
    name: str


#'''
class New_Address4(BaseModel):
    user: USER
    street: str


#'''


#'''
class New_Address3(BaseModel):
    username: str = Field(min_length=3, max_length=20, pattern="^[a-zA-Z0-9_]+$")
    email: Optional[str]
    age: int = Field(gt=12, lt=100)
    bio: Optional[str] = Field(default=None, max_length=250)
    interests: List[str] = Field(default_factory=list)


#'''


#'''
class New_Address2(BaseModel):
    field_decimal: decimal.Decimal = Field(
        gt=3, lt=15, multiple_of=0.000005, decimal_places=2
    )
    field_float: float = Field(gt=15, allow_inf_nan=True)


#'''


#'''
class New_Address1(BaseModel):
    street: str
    city: str
    zip_code: List[
        Dict[
            constr(pattern=r"^\d{3}(-\d{6})?$"),
            List[constr(pattern=r"^\d{5}(-\d{4})?$")],
        ]
    ]  # either ddddd or ddddd-dddd
    country: str = Field(default="USA")
    social_security_number: str
    continent: Optional[str]
    phone_number: List[Dict[str, List[str]]]
    title: Set[str]
    direction: Literal["north", "south", "east", "west"]
    bananas: str
    user: USER
    name: Annotated[
        Dict[constr(max_length=1000), constr(max_length=2000)], Field(min_length=30)
    ]


#'''


#'''
class Address(BaseModel):
    street: str
    city: str
    zip_code: List[
        Dict[
            constr(pattern=r"^\d{3}(-\d{6})?$"),
            List[constr(pattern=r"^\d{5}(-\d{4})?$")],
        ]
    ]  # either ddddd or ddddd-dddd
    country: str = Field(default="USA")
    social_security_number: str
    continent: Optional[str]
    phone_number: List[Dict[str, List[str]]]
    title: Set[str]
    direction: Literal["north", "south", "east", "west"]
    bananas: str
    user: USER
    name: Annotated[
        Dict[constr(max_length=1000), constr(max_length=2000)], Field(min_length=30)
    ]
    field_decimal_constr: decimal.Decimal = Field(
        gt=3, lt=15, multiple_of=0.000005, decimal_places=2
    )
    field_float_constr: float = Field(gt=15, allow_inf_nan=True)
    field_str: str
    field_int: int
    field_float: float
    field_bool: bool
    field_complex: complex
    field_bytes: bytes
    field_tuple: tuple
    field_list: list  # List[constr(min_length = 20)] = Field(min_length = 50)
    field_set: set
    field_frozenset: frozenset
    field_dict: dict
    username: str = Field(min_length=3, max_length=20, pattern="^[a-zA-Z0-9_]+$")
    email: Optional[str]
    age: int = Field(gt=12, lt=100)
    bio: Optional[str] = Field(default=None, max_length=250)
    interests: List[str] = Field(default_factory=list)
    n1: New_Address1
    n2: New_Address2
    n3: New_Address3
    n4: New_Address4


#'''
