import pytest

from src.sm.synthesiser.field_match.calc_difference import (
    levenshtein_distance,
    calc_difference,
)
from src.sm.synthesiser.synthesiser import Synthesiser
from faker import Faker

synth = Synthesiser()
fake = Faker()

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


word_list = synth.word_list
word_tokens = synth.word_tokens
word_tokens_set = synth.word_tokens_set


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
