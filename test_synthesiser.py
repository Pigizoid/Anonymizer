# test_synthesiser.py


import pytest


from functions.synthesiser import Synthesiser


import pydantic
from pydantic import *

from typing import *


from faker import Faker
import mimesis
from mimesis import Generic

import re

fake = Faker()
generic = Generic(mimesis.locales.Locale.EN)


class test_Address(BaseModel):
    street: str
    city: str


class test_Address_2(BaseModel):
    street: str
    city: str
    social_security_number: str
    continent: str


test_schemas = [
    test_Address,
]


test_word_num = 10
test_words = [fake.name() for x in range(test_word_num)]

test_words_1 = []
for x in range(test_word_num):
    test_words_1.extend(test_words)

test_words_2 = []
for x in range(test_word_num):
    test_words_2.extend([test_words[x] for y in range(test_word_num)])


def test_list_faker_methods_pass():
    return_Data = Synthesiser.list_faker_methods()
    assert isinstance(return_Data, tuple)
    assert isinstance(return_Data[0], list)
    assert isinstance(return_Data[1], dict)


def test_list_faker_methods_alternate():
    pass


def test_list_faker_methods_fail():
    pass


def test_list_mimesis_methods_pass():
    return_Data = Synthesiser.list_mimesis_methods()
    assert isinstance(return_Data, tuple)
    assert isinstance(return_Data[0], list)
    assert isinstance(return_Data[1], dict)


def test_list_mimesis_methods_alternate():
    pass


def test_list_mimesis_methods_fail():
    pass


@pytest.mark.parametrize("method", [("faker"), ("mimesis"), ("mixed")])
def test_list_match_methods_pass(method):
    return_Data = Synthesiser.list_match_methods(method)
    assert isinstance(return_Data, tuple)
    assert isinstance(return_Data[0], list)
    assert isinstance(return_Data[1], dict)


def test_list_match_methods_alternate():
    pass


def test_list_match_methods_fail():
    with pytest.raises(Exception):
        return_Data = Synthesiser.list_match_methods("bad_method")


test_get_model_data_pass_schemas = [test_Address]


@pytest.mark.parametrize(
    "schema_model", [(schema_item) for schema_item in test_get_model_data_pass_schemas]
)
def test_get_model_data_pass(schema_model):
    model_data = Synthesiser.get_model_data(schema_model)
    assert isinstance(model_data, list)
    assert all([isinstance(item, list) for item in model_data])
    assert all([isinstance(item[0], str) for item in model_data])
    assert all([isinstance(item[1], pydantic.fields.FieldInfo) for item in model_data])


def test_get_model_data_alternate():
    pass


def test_get_model_data_fail():
    pass
    # with pytest.raises(Exception):
    # model_data = Synthesiser.get_model_data("bad_method")


@pytest.mark.parametrize(
    "word1,word2,modifiers",
    [(word1, word2, [0, 0, 0]) for word1, word2 in zip(test_words_1, test_words_2)],
)
def test_levenshtein_distance_pass(word1, word2, modifiers):
    value = Synthesiser.levenshtein_distance(word1, word2, modifiers)
    assert isinstance(value, int) or isinstance(value, float)


@pytest.mark.parametrize(
    "word1,word2,modifiers",
    [(word1, word2, [0, 0, 0]) for word1, word2 in zip(test_words_1, test_words_2)],
)
def test_levenshtein_distance_alternate_modifiers(word1, word2, modifiers):
    value = Synthesiser.levenshtein_distance(word1, word2, modifiers)
    value_nm = Synthesiser.levenshtein_distance(word1, word2)
    value_m1 = Synthesiser.levenshtein_distance(word1, word2, [1, 0, 0])
    value_m2 = Synthesiser.levenshtein_distance(word1, word2, [0, 1, 0])
    value_m3 = Synthesiser.levenshtein_distance(word1, word2, [0, 0, 1])
    assert value_nm == value
    assert value_m1 >= value
    assert value_m2 >= value
    assert value_m3 >= value


def test_levenshtein_distance_alternate_value_check():
    value = Synthesiser.levenshtein_distance("test", "tester")
    assert value == 2


def test_levenshtein_distance_fail():
    pass


@pytest.mark.parametrize(
    "target_word,word,target_tokens,target_tokens_set",
    [
        (word1, word2, word1.split("_"), set(word1.split("_")))
        for word1, word2 in zip(test_words_1, test_words_2)
    ],
)
def test_calc_difference_pass(target_word, word, target_tokens, target_tokens_set):
    value = Synthesiser.calc_difference(
        target_word, word, target_tokens, target_tokens_set
    )
    assert isinstance(value, int) or isinstance(value, float)


def test_calc_difference_alternate_abbreviation():
    value = Synthesiser.calc_difference(
        "ssn", "social_security_number", "ssn".split("_"), set("ssn".split("_"))
    )
    assert value == 0
    value = Synthesiser.calc_difference(
        "social_security_number",
        "ssn",
        "social_security_number".split("_"),
        set("social_security_number".split("_")),
    )
    assert value == 0


def test_calc_difference_fail():
    pass


def test_match_fields_expect_pass():
    return_value = Synthesiser.match_fields(test_Address, method="mixed")
    assert isinstance(return_value, tuple)
    assert isinstance(return_value[0], tuple)
    assert isinstance(return_value[0][0], list)
    assert all([isinstance(x, str) for x in return_value[0][0]])
    assert isinstance(return_value[0][1], list)
    assert isinstance(return_value[1], dict)


def test_match_fields_expect_alternate_methods():
    return_value_faker = Synthesiser.match_fields(test_Address_2, method="faker")
    assert return_value_faker[0][1][0] != ""
    assert return_value_faker[0][1][1] != ""
    assert return_value_faker[0][1][2] != ""
    assert return_value_faker[0][1][3] == ""  # doesnt contain this


def test_match_fields_expect_alternate_methods1():
    return_value_mimesis = Synthesiser.match_fields(test_Address_2, method="mimesis")
    assert return_value_mimesis[0][1][0] != ""
    assert return_value_mimesis[0][1][1] != ""
    assert return_value_mimesis[0][1][2] == ""  # doesnt contain this
    assert return_value_mimesis[0][1][3] != ""


def test_match_fields_expect_alternate_methods2():
    return_value_mixed = Synthesiser.match_fields(test_Address_2, method="mixed")
    assert return_value_mixed[0][1][0] != ""
    assert return_value_mixed[0][1][1] != ""
    assert return_value_mixed[0][1][2] != ""
    assert return_value_mixed[0][1][3] != ""


def test_match_fields_expect_alternate_methods3():
    return_value_default = Synthesiser.match_fields(test_Address_2)
    assert return_value_default[0][1][0] != ""
    assert return_value_default[0][1][1] != ""
    assert return_value_default[0][1][2] != ""
    assert return_value_default[0][1][3] == ""  # same as faker


def test_match_fields_expect_fail():
    pass


class Constraints1(BaseModel):
    name: str


class Constraints2(BaseModel):
    name: str = Field(default="hello")


class Constraints3(BaseModel):
    street: str
    city: str
    zip_code: str = Field(pattern=r"^\d{5}(-\d{4})?$")
    country: str = Field(default="USA")


class Constraints4(BaseModel):
    social_security_number: str
    continent: Optional[str]
    phone_number: List[Dict[str, List[str]]]
    title: Set[str]
    direction: Literal["north", "south", "east", "west"]
    name: Annotated[
        Dict[constr(max_length=1000), constr(max_length=2000)], Field(min_length=30)
    ]


class Constraints5(BaseModel):
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
    name: Annotated[
        Dict[constr(max_length=1000), constr(max_length=2000)], Field(min_length=30)
    ]


test_check_generation_constraints_pass_schemas = [test_Address]


@pytest.mark.parametrize(
    "schema_model",
    [(schema_item) for schema_item in test_check_generation_constraints_pass_schemas],
)
def test_check_generation_constraints_expect_pass(schema_model):
    method = "mixed"

    for name, field in schema_model.model_fields.items():
        return_value = Synthesiser.check_generation_constraints(name, field)
        assert isinstance(return_value, dict)
        assert set(return_value.keys()) == set(
            [
                "strip_whitespace",
                "to_upper",
                "to_lower",
                "strict",
                "default",
                "annotation",
                "min_length",
                "max_length",
                "pattern",
                "gt",
                "lt",
                "ge",
                "le",
                "multiple_of",
                "allow_inf_nan",
                "max_digits",
                "decimal_places",
                "origin",
                "args",
                "required",
            ]
        )


import decimal


class Constraints6(BaseModel):
    constr_strip_whitespace: constr(strip_whitespace=True)
    constr_to_upper: constr(to_upper=True)
    constr_to_lower: constr(to_lower=True)
    constr_strict: constr(strict=True)
    constr_default: str = Field(default="AAA")
    constr_annotation: str
    constr_min_length: constr(min_length=5)
    constr_max_length: constr(max_length=5)
    constr_pattern: constr(pattern=r"^\d{5}(-\d{4})?$")
    constr_gt: int = Field(gt=5)
    constr_lt: int = Field(lt=5)
    constr_ge: int = Field(ge=5)
    constr_le: int = Field(le=5)
    constr_multiple_of: int = Field(multiple_of=5)
    constr_allow_inf_nan: int = Field(allow_inf_nan=True)
    constr_max_digits: decimal.Decimal = Field(max_digits=5)
    constr_decimal_places: decimal.Decimal = Field(decimal_places=5)
    constr_origin: List[str]
    constr_args: Dict[str, str]
    constr_required: str


def test_check_generation_constraints_expect_alternate():
    schema_model = Constraints6
    method = "mixed"

    return_value = {}
    for name, field in schema_model.model_fields.items():
        return_value[name] = Synthesiser.check_generation_constraints(name, field)
    assert return_value["constr_strip_whitespace"]["strip_whitespace"] == True
    assert return_value["constr_to_upper"]["to_upper"] == True
    assert return_value["constr_to_lower"]["to_lower"] == True
    assert return_value["constr_strict"]["strict"] == True
    assert return_value["constr_default"]["default"] == "AAA"
    assert return_value["constr_annotation"]["annotation"] == str
    assert return_value["constr_min_length"]["min_length"] == 5
    assert return_value["constr_max_length"]["max_length"] == 5
    assert return_value["constr_pattern"]["pattern"] == r"^\d{5}(-\d{4})?$"
    assert return_value["constr_gt"]["gt"] == 5
    assert return_value["constr_lt"]["lt"] == 5
    assert return_value["constr_ge"]["ge"] == 5
    assert return_value["constr_le"]["le"] == 5
    assert return_value["constr_multiple_of"]["multiple_of"] == 5
    assert return_value["constr_allow_inf_nan"]["allow_inf_nan"] == True
    assert return_value["constr_max_digits"]["max_digits"] == 5
    assert return_value["constr_decimal_places"]["decimal_places"] == 5
    assert return_value["constr_origin"]["origin"] in [List, list]
    assert return_value["constr_args"]["args"]
    assert return_value["constr_required"]["required"] == True


def test_check_generation_constraints_expect_fail():
    pass


class generate_constraints1(BaseModel):
    test: list[Dict[constr(pattern=r"^a$"), constr(pattern=r"^a$")]]


class generate_constraints2(BaseModel):
    test: list[Dict[str, str]]


def test_generate_from_constraints_expect_pass():
    generate_path = "test[100].List(0)[10].Dict(Right)[10].Annotated"
    schema_model = generate_constraints1
    constraints = {}
    constraints["annotation"] = str
    constraints["pattern"] = r"^a$"
    return_value = Synthesiser.generate_from_constraints(
        "test", constraints, generate_path
    )
    assert return_value == "a"
    data_pool = Synthesiser.outputpooling
    assert (
        len(data_pool[generate_path]) == 10000 - 1
    )  # -1 because the pop method was run
    assert all(
        [re.search(r"a", text).group() == text for text in data_pool[generate_path]]
    )


def test_generate_from_constraints_expect_alternate():
    generate_path = "test[100].List(0)[10].Dict(Right)[10]"
    schema_model = generate_constraints2
    constraints = {}
    constraints["annotation"] = str
    constraints["pattern"] = None
    return_value = Synthesiser.generate_from_constraints(
        "test", constraints, generate_path
    )
    assert isinstance(return_value, str)


def test_generate_from_constraints_expect_fail():
    pass


class generate_test1(BaseModel):
    test_none: None
    test_basic: str
    test_pattern: str = Field(constr(pattern=r"^a$"))
    test_list: List[str]
    test_dict: Dict[constr(pattern=r"^\d{50}$"), str]
    test_tuple: Tuple[str, str, str]
    test_set: Set[str]
    test_union: Union[str, None]
    test_literal: Literal["1", "2", "3", "4"]
    test_recursive: List[Dict[constr(pattern=r"^\d{50}$"), List[Tuple[str, str]]]]
    test_list_length: List[constr(pattern=r"^\d{50}$")] = Field(constr(min_length=20))
    test_list_length: Dict[constr(pattern=r"^\d{50}$"), constr(pattern=r"^\d{50}$")] = (
        Field(constr(min_length=20))
    )
    test_list_length: Set[constr(pattern=r"^\d{50}$")] = Field(constr(min_length=20))


class generate_test2(BaseModel):
    test_dict_fail: Dict[constr(pattern=r"^a$"), constr(pattern=r"^a$")] = Field(
        min_length=20
    )  # returns "a" which means there isnt enough keys


def test_generate_synth_data_expect_pass():
    schema_model = generate_test1
    method = "mixed"
    name_match_pairs, methods_map = Synthesiser.match_fields(schema_model, method)
    field_match_pairs = {
        name: match for name, match in zip(name_match_pairs[0], name_match_pairs[1])
    }
    applied_constraints = Synthesiser.make_applied_constraints(schema_model)
    return_value = {}
    for name, field in schema_model.model_fields.items():
        generate_path = f"{name}[1]"
        return_value[name] = Synthesiser.generate_synth_data(
            name, field_match_pairs[name], applied_constraints[name], generate_path
        )
    assert schema_model(**return_value)


def test_generate_synth_data_expect_alternate():
    pass


def test_generate_synth_data_expect_fail():
    schema_model = generate_test2
    method = "mixed"
    name_match_pairs, methods_map = Synthesiser.match_fields(schema_model, method)
    field_match_pairs = {
        name: match for name, match in zip(name_match_pairs[0], name_match_pairs[1])
    }
    applied_constraints = Synthesiser.make_applied_constraints(schema_model)
    return_value = {}
    with pytest.raises(Exception):
        for name, field in schema_model.model_fields.items():
            generate_path = f"{name}[1]"
            return_value[name] = Synthesiser.generate_synth_data(
                name, field_match_pairs[name], applied_constraints[name], generate_path
            )


def test_make_applied_constraints_expect_pass():
    return_value = Synthesiser.make_applied_constraints(generate_test1)
    assert isinstance(return_value, dict)
    assert all([isinstance(value, dict) for value in return_value.values()])
    keys = [
        "strip_whitespace",
        "to_upper",
        "to_lower",
        "strict",
        "default",
        "annotation",
        "min_length",
        "max_length",
        "pattern",
        "gt",
        "lt",
        "ge",
        "le",
        "multiple_of",
        "allow_inf_nan",
        "max_digits",
        "decimal_places",
        "origin",
        "args",
        "required",
    ]
    assert all([key in value for key in keys for value in return_value.values()])


def test_make_applied_constraints_expect_alternate():
    pass


def test_make_applied_constraints_expect_fail():
    pass


class resolved_test(BaseModel):
    street: str
    city: str
    social_security_number: str
    continent: str


class resolved_test2(BaseModel):
    street: str
    banana: str


def test_make_resolved_methods_expect_pass():
    method = "mixed"
    schema_model = resolved_test
    name_match_pairs, methods_map = Synthesiser.match_fields(schema_model, method)
    Synthesiser.make_resolved_methods(name_match_pairs[1], methods_map)
    return_value = Synthesiser.resolved_methods
    assert isinstance(return_value, dict)
    # print([type(item) for item in return_value.values()])
    assert all([callable(item) for item in return_value.values()])


def test_make_resolved_methods_expect_alternate():
    method = "mixed"
    schema_model = resolved_test2
    name_match_pairs, methods_map = Synthesiser.match_fields(schema_model, method)
    Synthesiser.make_resolved_methods(name_match_pairs[1], methods_map)
    return_value = Synthesiser.resolved_methods
    # print(return_value)
    assert isinstance(return_value, dict)
    assert "banana" not in return_value


def test_make_resolved_methods_expect_fail():
    pass


synthesise_schema_model = generate_test1


@pytest.mark.parametrize(
    "schema_model,method,amount",
    [
        (synthesise_schema_model, "faker", 1),
        (synthesise_schema_model, "mimesis", 1),
        (synthesise_schema_model, "mixed", 1),
        (synthesise_schema_model, "faker", 10),
        (synthesise_schema_model, "mimesis", 10),
        (synthesise_schema_model, "mixed", 10),
    ],
)
def test_synthesise_expect_pass(schema_model, method, amount):
    return_data = Synthesiser.synthesise(schema_model, method, amount)
    assert isinstance(return_data, list)
    assert len(return_data) == amount
    assert all([isinstance(item, schema_model) for item in return_data])


def test_synthesise_expect_alternate():
    pass


def test_synthesise_expect_fail():
    pass


"""
def test__expect_pass():
	
def test__expect_alternate():
	
def test__expect_fail():
	
"""
