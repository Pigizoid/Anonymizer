import pytest
from pydantic import BaseModel
from src.smoke_mirrors.synthesiser.synthesiser import Synthesiser
from src.smoke_mirrors.anonymiser.anonymiser import anonymise, subset_model, new_model, mask_value, perturb_value, anonymise_value, anonymise_data
from src.smoke_mirrors.tools.model_funcs import get_model_fields


class schema_model(BaseModel):
    foo: str
    bar: int
    zar: bool


vals = [True, 1.5, 15, b"hello", "hello", []]
methods = ["faker", "mimesis", "mixed"]
synth = Synthesiser()
schema_input_data = {"foo": "hello", "bar": 10, "zar": True}
input_data = {"0":schema_input_data}
manuals = [True, False]
defaults = ["mask","perturb","synth"]
field_sets = [
    {"foo":"default"},
    {"bar":"default"},
    {"zar":"default"},
    {"foo":"default", "bar":"default"},
    {"bar":"default", "zar":"default"},
    {"foo":"default", "bar":"default", "zar":"default"},
]
amounts = [1, 5, 10]
key_anons = [True,False]

input_sets = []
for method in methods:
    for manual in manuals:
        for default in defaults:
            for fields in field_sets:
                for amount in amounts:
                    for key_anon in key_anons:
                        input_sets.append((schema_model,input_data,method,manual,default,fields,amount,0,key_anon))
@pytest.mark.parametrize("schema_model, data, method, manual, default, fields, amount, seed, key_anon",input_sets)
def test_anonymise(schema_model, data, method, manual, default, fields, amount, seed, key_anon):
    print(fields)
    return_data = anonymise(schema_model, data, method, manual, default, fields, amount, seed=seed, key_anon=key_anon)
    assert isinstance(return_data,dict)


@pytest.mark.parametrize(
    "field_value,expected",
    [
        (True, False),
        (1.5, 0.0),
        (15, 0),
        (b"hello", b"0"),
        ("hello", "****"),
        ([], []),
    ],
)
def test_mask_value(field_value, expected):
    assert mask_value(field_value) == expected


@pytest.mark.parametrize(
    "field_value",
    [
        (True),
        (1.5),
        (15),
        (b"hello"),
        ("hello"),
        ([]),
    ],
)
def test_perturb_value(field_value):
    assert isinstance(perturb_value(field_value), type(field_value))


vals = [True, 1.5, 15, b"hello", "hello", []]
methods = ["mask", "synth", "perturb"]
synth = Synthesiser()
anonymise_value_test_data = []
for method in methods:
    anonymise_value_test_data.extend([(val, ["name", method]) for val in vals])


@pytest.mark.parametrize(
    "field_value, anon_methods, seed, synth",
    [(val, anon_methods, 0, synth) for val, anon_methods in anonymise_value_test_data],
)
def test_anonymise_value(field_value, anon_methods, seed, synth):
    assert isinstance(
        anonymise_value(field_value, anon_methods, seed, synth), type(field_value)
    )


input_data = {f"field_{x}": val for x, val in enumerate(vals)}
recursive_input_data = {
    "list": [x for x in range(10)],
    "tuple": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    "set": set(x for x in range(10)),
    "dict": {f"field_{x}": x for x in range(10)},
}
input_data.update(recursive_input_data)
anon_methods = ["mask" for _ in range(len(input_data))]


def test_anonymise_data_mask():
    return_data = anonymise_data(input_data, anon_methods, 0)
    assert return_data != input_data


anon_methods = ["synth" for _ in range(len(input_data))]


def test_anonymise_data_synth():
    return_data = anonymise_data(input_data, anon_methods, 0, synth)
    assert return_data != input_data


anon_methods = ["perturb" for _ in range(len(input_data))]


def test_anonymise_data_perturb():
    return_data = anonymise_data(input_data, anon_methods, 0)
    print(input_data)
    print(return_data)
    assert type(return_data["field_0"]) is type(input_data["field_0"])
    assert type(return_data["field_1"]) is type(input_data["field_1"])
    assert type(return_data["field_2"]) is type(input_data["field_2"])
    assert type(return_data["field_3"]) is type(input_data["field_3"])
    assert type(return_data["field_4"]) is type(input_data["field_4"])
    assert type(return_data["field_5"]) is type(input_data["field_5"])
    assert all(
        [
            type(return_data["list"][x]) is type(input_data["list"][x])
            for x in range(len(input_data["list"]))
        ]
    )
    assert all(
        [
            type(x1) is type(x2)
            for x1, x2 in zip(return_data["tuple"], input_data["tuple"])
        ]
    )
    assert set([type(x) for x in return_data["set"]]) == set(
        [type(x) for x in input_data["set"]]
    )
    assert all(
        [
            type(return_data["dict"][f"field_{x}"])
            is type(input_data["dict"][f"field_{x}"])
            for x in range(len(input_data["dict"]))
        ]
    )



class schema_model(BaseModel):
    foo: str
    bar: int
    zar: bool


def test_subset_model():
    field_names = ["bar", "zar"]
    return_model = subset_model(schema_model, field_names)
    names = [name for name in get_model_fields(return_model).keys()]
    assert names == field_names


data = {"foo": "hello", "bar": 10, "zar": False}
def test_new_model():
    field_names = ["bar", "zar"]
    return_model = new_model(data, field_names)
    names = [name for name in get_model_fields(return_model).keys()]
    assert names == field_names

