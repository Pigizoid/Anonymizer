from typing import Dict, List, Set
from pydantic import BaseModel
import inspect
from sm.pre_made_data import provider_methods
from sm.tools.model_funcs import get_model_data

def levenshtein_distance(word1:str, word2:str, modifiers=None) -> float:
    """
    algorithm for calculating difference of two strings\n
    e.g:\n
        cat -> car = 1
        books -> bo = 3
        books -> cooks = 1
        cat -> cat = 0
    Inputs:\n
        word1 string
        word2 string
    Outputs:\n
        float
    """
    if modifiers is None:
        modifiers = [0, 0, 0]
    len_word1, len_word2 = len(word1), len(word2)
    distance_point = [[0 for _ in range(len_word2 + 1)] for _ in range(len_word1 + 1)]
    for i in range(len_word1 + 1):
        distance_point[i][0] = i
    for j in range(len_word2 + 1):
        distance_point[0][j] = j
    for i in range(1, len_word1 + 1):
        for j in range(1, len_word2 + 1):
            cost = 0 if word1[i - 1] == word2[j - 1] else 1
            distance_point[i][j] = min(
                distance_point[i - 1][j] + 1 + modifiers[0],
                distance_point[i][j - 1] + 1 + modifiers[1],
                distance_point[i - 1][j - 1] + cost + modifiers[2],
            )
    return distance_point[len_word1][len_word2]

def calc_difference(
    target_word:str,
    word:str,
    word_tokens:Dict[str,List[str]],
    word_tokens_set:Dict[str,Set[str]],
    target_tokens:Dict[str,List[str]],
    target_tokens_set:Dict[str,Set[str]],
) -> float:
    """
    algorithm for calculating difference of two strings\n
    while implementing robust fuzzy matching\n
    e.g:\n
        books -> cooks = 1
        books -> bo = 3
        name_first -> first_name = 0
        social_security_number -> ssn = 0
        ssn -> social_security_number = 0
    additionally matches by stripping underscores\n
    and matching based on word positioning\n
    with later positioned words adding a higher cost rating\n
    Inputs:\n
        target word
        input word
    Outputs:\n
        float
    """
    if target_tokens_set == word_tokens_set:  # name_first -> first_name
        return 0

    if (
        ("".join([letter[0] for letter in word.split("_")])) == target_word.lower()
    ):  # abbreviation mapping	ssn -> social_security_number   #ssn is target word
        return 0
    if (
        "".join([letter[0] for letter in target_word.lower().split("_")])
    ) == word:  # abbreviation mapping	social_security_number -> ssn
        return 0

    main_distance = levenshtein_distance(target_word, word.lower())  # close early exit
    if main_distance <= 1 or main_distance >= len(target_word):
        return main_distance

    main_distance_joined = main_distance - (
        (len(target_word) - len(target_word.replace("_", "")))
        + (len(word) - len(word.replace("_", "")))
    )
    if main_distance_joined <= 1:
        return main_distance_joined

    token_distance = 0
    cross_points = len(word_tokens) * len(target_tokens)
    if len(target_tokens) != len(word_tokens):
        if len(target_tokens) < len(word_tokens):
            if target_tokens_set.issubset(word_tokens_set):
                token_distance = word_tokens.index(target_tokens[0]) * cross_points
                distance = (main_distance / 2 + token_distance) / 2
            else:
                token_distance = (
                    levenshtein_distance(target_word, word.lower()) * cross_points
                )
                distance = (main_distance + token_distance) / 2
        else:
            if word_tokens_set.issubset(target_tokens_set):
                token_distance = target_tokens.index(word_tokens[0]) * cross_points
                distance = (main_distance / 2 + token_distance) / 2
            else:
                token_distance = (
                    levenshtein_distance(target_word, word.lower()) * cross_points
                )
                distance = (main_distance + token_distance) / 2
    else:
        distance = levenshtein_distance(target_word, word.lower()) * cross_points
    return distance

def match_fields(field_names: List[str], method: str) -> Dict[str, str]:
    """
    1. calculates the distance of each field_name in the list:\n
        to the closest matching generation provider
        e.g. street -> street_name
    2. if none is found, retry with modified values [-0.5, 0.5, -0.5]\n
    Inputs:\n
        list of field_names
    Outputs:\n
        matched fields = {field_name, match name}
    """
    data_set = provider_methods[method]
    word_list = data_set["word_list"]
    word_tokens = data_set["word_tokens"]
    word_tokens_set = data_set["word_tokens_set"]

    field_matches = []
    for t_word in field_names:
        closest_matches = []
        distances = []
        target_word = t_word.lower()
        target_tokens = target_word.split("_")
        target_tokens_set = set(target_tokens)
        for word in word_list:
            distance = calc_difference(
                target_word,
                word,
                word_tokens[word],
                word_tokens_set[word],
                target_tokens,
                target_tokens_set,
            )
            distances.append([word, distance])
        sorted_by_distance = sorted(distances, key=lambda dist: dist[1])
        if sorted_by_distance != [] and sorted_by_distance[0] != []:
            min_value = sorted_by_distance[0][1]
        else:
            min_value = (len(target_word) // 2 - 0.5) + 1
        if min_value > (len(target_word) // 2 - 0.5):
            potential_matches = []
            for word in word_list:
                if target_word in word:
                    distance = levenshtein_distance(
                        target_word, word, [-0.5, 0.5, -0.5]
                    )
                    potential_matches.append([word, distance])
            sorted_match_by_distance = sorted(
                potential_matches, key=lambda dist: dist[1]
            )
            if sorted_match_by_distance != [] and sorted_match_by_distance[0] != []:
                min_value = sorted_match_by_distance[0][1]
            else:
                min_value = (len(target_word) // 2) + 1
            if min_value > len(target_word) // 2:
                closest_matches = [
                    item
                    for item in sorted_match_by_distance
                    if item[1] == min_value
                ]
        else:
            closest_matches = [
                item for item in sorted_by_distance if item[1] == min_value
            ]
        if len(closest_matches) == 0:
            field_matches.append("")
        else:
            field_matches.append(closest_matches[0][0])

    field_match_pairs = {}
    for name, match in zip([x for x in field_names], field_matches):
        field_match_pairs[name] = match
    return field_match_pairs

def recursive_match_fields(schema_model:BaseModel, method, field_match_pairs:Dict[str,Dict[str,str]]=None) -> Dict[str, Dict[str, str]]:
    """
    Goes through an input schema model and matches all fields to a generation provider\n
    recurses through nested schemas, adding to list of providers for each field\n
    uses optional field_match_pairs input during recursion and remove duplicate nested schemas\n
    Inputs:\n
        schema model
        optional field match pairs (generated during output from recursion)
    Outputs:\n
        field match pairs = {schema name: {field name: match name}}
    """
    if field_match_pairs is None:
        field_match_pairs = {}
    # in body not in function, because default collections are stored in memory not by instance
    model_data = get_model_data(schema_model)
    field_names = [x[0] for x in model_data]

    if schema_model.__name__ not in field_match_pairs.keys():
        field_match_pairs[schema_model.__name__] = match_fields(field_names, method)
    for x in model_data:
        data_type = x[1].annotation
        if (
            inspect.isclass(data_type)
            and issubclass(data_type, BaseModel)
            and data_type.__name__ not in field_match_pairs.keys()
        ):
            # print(data_type.__name__, field_match_pairs.keys())
            field_match_pairs.update(recursive_match_fields(data_type,method))
    return field_match_pairs
