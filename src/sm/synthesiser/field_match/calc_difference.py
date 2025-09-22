from typing import Dict, List, Set
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
