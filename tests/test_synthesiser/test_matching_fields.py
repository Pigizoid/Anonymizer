import pytest

from faker import Faker
fake = Faker()
from src.smoke_mirrors.pre_made_data import provider_methods
from smoke_mirrors.synthesiser.matching_fields import levenshtein_distance, calc_difference, match_fields




test_word_num = 10
test_words = [fake.name() for x in range(test_word_num)]
test_words_1 = []
for x in range(test_word_num):
    test_words_1.extend(test_words)
test_words_2 = []
for x in range(test_word_num):
    test_words_2.extend([test_words[x] for y in range(test_word_num)])


@pytest.mark.parametrize(
    "word1,word2,modifiers",
    [(word1, word2, [0, 0, 0]) for word1, word2 in zip(test_words_1, test_words_2)],
)
def test_levenshtein_distance_pass(word1, word2, modifiers):
    value = levenshtein_distance(word1, word2, modifiers)
    assert isinstance(value, int) or isinstance(value, float)


@pytest.mark.parametrize(
    "word1,word2,modifiers",
    [(word1, word2, [0, 0, 0]) for word1, word2 in zip(test_words_1, test_words_2)],
)
def test_levenshtein_distance_alternate_modifiers(word1, word2, modifiers):
    value = levenshtein_distance(word1, word2, modifiers)
    value_nm = levenshtein_distance(word1, word2)
    value_m1 = levenshtein_distance(word1, word2, [1, 0, 0])
    value_m2 = levenshtein_distance(word1, word2, [0, 1, 0])
    value_m3 = levenshtein_distance(word1, word2, [0, 0, 1])
    assert value_nm == value
    assert value_m1 >= value
    assert value_m2 >= value
    assert value_m3 >= value


def test_levenshtein_distance_alternate_value_check():
    value = levenshtein_distance("test", "tester")
    assert value == 2


word_list = provider_methods["faker"]["word_list"]
word_tokens = provider_methods["faker"]["word_tokens"]
word_tokens_set = provider_methods["faker"]["word_tokens_set"]


@pytest.mark.parametrize(
    "target_word,word,target_tokens,target_tokens_set",
    [
        (word1, word2, word1.split("_"), set(word1.split("_")))
        for word1, word2 in zip(test_words_1, test_words_2)
    ],
)
def test_calc_difference_pass(target_word, word, target_tokens, target_tokens_set):
    value = calc_difference(
        target_word,
        word,
        word_tokens,
        word_tokens_set,
        target_tokens,
        target_tokens_set,
    )
    assert isinstance(value, int) or isinstance(value, float)


def test_calc_difference_alternate_abbreviation():
    value = calc_difference(
        "ssn",
        "social_security_number",
        word_tokens,
        word_tokens_set,
        "ssn".split("_"),
        set("ssn".split("_")),
    )
    assert value == 0
    value = calc_difference(
        "social_security_number",
        "ssn",
        word_tokens,
        word_tokens_set,
        "social_security_number".split("_"),
        set("social_security_number".split("_")),
    )
    assert value == 0


field_names_1 = ["street", "city"]


field_names_2 = ["street", "city", "social_security_number", "continent"]


def test_match_fields():
    return_value = match_fields(field_names_1,"mixed")
    assert isinstance(return_value, dict)
    assert all(
        [isinstance(x, str) and isinstance(y, str) for x, y in return_value.items()]
    )


@pytest.mark.parametrize(
    "method,expected",
    [
        ("faker", ["street_name", "city", "ssn", (False,"continent")]),
        ("mimesis", ["street_name", "city", (False,"ssn"), "continent"]),
        ("mixed", ["street_name", "city", "ssn", "continent"]),
    ],
)
def test_match_fields_alternate_methods(method, expected):
    return_value = match_fields(field_names_2, method)
    print(return_value)
    assert all(
        [
            (return_value[field_names_2[x]] == expected[x] or (return_value[field_names_2[x]] == expected[x][1]) == expected[x][0])
            for x in range(len(field_names_2))
        ]
    )

