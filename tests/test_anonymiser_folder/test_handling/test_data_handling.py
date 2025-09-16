import pytest

from src.sm.synthesiser.synthesiser import Synthesiser
from src.sm.anonymiser.handling.data_handling import anonymise_value, anonymise_data
from src.sm.anonymiser.handling.data_handling import (
    mask_value,
    perturb_value
)


@pytest.mark.parametrize(
    "field_value,field_type,expected",
    [
        (True, bool, False),
        (1.5, float, 0.0),
        (15, int, 0),
        (b"hello", bytes, b"0"),
        ("hello", str, "****"),
        ([], list, []),
    ],
)
def test_mask_value(field_value, field_type, expected):
    assert mask_value(field_value, field_type) == expected


@pytest.mark.parametrize(
    "field_value,field_type",
    [
        (True, bool),
        (1.5, float),
        (15, int),
        (b"hello", bytes),
        ("hello", str),
        ([], list),
    ],
)
def test_perturb_value(field_value, field_type):
    assert isinstance(perturb_value(field_value, field_type), field_type)


vals = [True, 1.5, 15, b"hello", "hello", []]
methods = ["mask", "synth", "perturb"]
synth = Synthesiser()
anonymise_value_test_data = []
for method in methods:
    anonymise_value_test_data.extend([(val, ["name", method]) for val in vals])


@pytest.mark.parametrize(
    "seed, field_value, anon_methods, synth",
    [(0, val, anon_methods, synth) for val, anon_methods in anonymise_value_test_data],
)
def test_anonymise_value(seed, field_value, anon_methods, synth):
    assert isinstance(
        anonymise_value(seed, field_value, anon_methods, synth), type(field_value)
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
    return_data = anonymise_data(0, input_data, anon_methods)
    assert return_data != input_data


anon_methods = ["synth" for _ in range(len(input_data))]


def test_anonymise_data_synth():
    return_data = anonymise_data(0, input_data, anon_methods, synth)
    assert return_data != input_data


anon_methods = ["perturb" for _ in range(len(input_data))]


def test_anonymise_data_perturb():
    return_data = anonymise_data(0, input_data, anon_methods)
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
