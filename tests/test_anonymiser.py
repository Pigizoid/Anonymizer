# test_anonymiser.py


from src.functions.anonymiser import Anonymiser


from pydantic import *

from typing import *

import json


from faker import Faker
import mimesis
from mimesis import Generic


fake = Faker()
generic = Generic(mimesis.locales.Locale.EN)


class test_Address(BaseModel):
    street: str
    city: str
    social_security_number: str
    continent: str


class test_Address_subset(BaseModel):
    street: str
    city: str


def test_subset_model_expect_pass():
    schema_model = test_Address
    field_names = ["street", "city"]
    return_value = Anonymiser.subset_model(schema_model, field_names)
    assert isinstance(return_value, type)
    assert issubclass(return_value, BaseModel)
    assert return_value.__name__ == "new_schema_model"
    assert set(return_value.model_fields.keys()) == {"street", "city"}


def test_subset_model_expect_alternate():
    pass


def test_subset_model_expect_fail():
    pass


schema_model = test_Address


def test_anonymise_expect_pass():
    data = '{"street":"street","city":"city","social_security_number":"social_security_number","continent":"continent"}'
    data = json.loads(data)
    method = "mixed"
    manual = True
    fields = {"street": "default", "city": "default"}
    return_data = Anonymiser.anonymise(schema_model, data, method, manual, fields)
    assert isinstance(return_data, dict)
    assert return_data["street"] != "street"
    assert return_data["city"] != "city"
    assert return_data["social_security_number"] == "social_security_number"
    assert return_data["continent"] == "continent"


def test_anonymise_expect_alternate():
    pass


def test_anonymise_expect_fail():
    pass
