import pytest

from src.sm.synthesiser.synthesiser import Synthesiser
from src.sm.anonymiser.handling.data_handling import anonymise_value, anonymise_data
from src.sm.anonymiser.handling.data_handling import (
    mask_value,
    perturb_value
)


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
