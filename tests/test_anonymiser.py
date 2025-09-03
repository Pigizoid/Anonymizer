# test_anonymiser.py


from src.sm.functions.anonymiser import Anonymiser


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
    data = '{"0":{"street":"street","city":"city","social_security_number":"social_security_number","continent":"continent"}}'
    data = json.loads(data)
    method = "mixed"
    manual = True
    fields = {"street": "default", "city": "default"}
    amount = 5
    return_data = Anonymiser.anonymise(schema_model, data, method, manual, fields, amount)
    #print(return_data)
    assert isinstance(return_data, dict)
    assert all([isinstance(y, list) for x,y in return_data.items()])
    assert all([len(y)==amount for x,y in return_data.items()])
    assert all([isinstance(z, BaseModel) for x,y in return_data.items() for z in y])
    assert all([getattr(z, "street") != "street" for x,y in return_data.items() for z in y])
    assert all([getattr(z, "city") != "city" for x,y in return_data.items() for z in y])
    assert all([getattr(z, "social_security_number") == "social_security_number" for x,y in return_data.items() for z in y])
    assert all([getattr(z, "continent") == "continent" for x,y in return_data.items() for z in y])


def test_anonymise_expect_alternate_manual_false():
    data = '{"0":{"street":"street","city":"city","social_security_number":"social_security_number","continent":"continent"}}'
    data = json.loads(data)
    method = "mixed"
    manual = False
    fields = {"street": "default", "city": "default"}
    amount = 5
    return_data = Anonymiser.anonymise(schema_model, data, method, manual, fields, amount)
    #print(return_data)
    assert isinstance(return_data, dict)
    assert all([isinstance(y, list) for x,y in return_data.items()])
    assert all([len(y)==amount for x,y in return_data.items()])
    assert all([isinstance(z, BaseModel) for x,y in return_data.items() for z in y])
    assert all([getattr(z, "street") != "street" for x,y in return_data.items() for z in y])
    assert all([getattr(z, "city") != "city" for x,y in return_data.items() for z in y])
    assert all([getattr(z, "social_security_number") != "social_security_number" for x,y in return_data.items() for z in y])
    assert all([getattr(z, "continent") != "continent" for x,y in return_data.items() for z in y])

def test_anonymise_expect_alternate_more_data():
    data = '{"0":{"street":"street","city":"city","social_security_number":"social_security_number","continent":"continent"},"1":{"street":"street","city":"city","social_security_number":"social_security_number","continent":"continent"}}'
    data = json.loads(data)
    method = "mixed"
    manual = True
    fields = {"street": "default", "city": "default"}
    amount = 5
    return_data = Anonymiser.anonymise(schema_model, data, method, manual, fields, amount)
    #print(return_data)
    assert isinstance(return_data, dict)
    assert all([isinstance(y, list) for x,y in return_data.items()])
    assert all([len(y)==amount for x,y in return_data.items()])
    assert all([isinstance(z, BaseModel) for x,y in return_data.items() for z in y])
    assert all([getattr(z, "street") != "street" for x,y in return_data.items() for z in y])
    assert all([getattr(z, "city") != "city" for x,y in return_data.items() for z in y])
    assert all([getattr(z, "social_security_number") == "social_security_number" for x,y in return_data.items() for z in y])
    assert all([getattr(z, "continent") == "continent" for x,y in return_data.items() for z in y])

def test_anonymise_expect_fail():
    pass
