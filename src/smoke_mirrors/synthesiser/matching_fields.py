from typing import Dict, List, Set, Tuple
from smoke_mirrors.pre_made_data import provider_methods, provider_return_types


def levenshtein_distance(word1: str, word2: str, modifiers=None) -> float:
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
        optional modifiers [ int, int, int ]  deletion, insertion, substitution
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


VOWELS = {ord(ch) for ch in set("aeiouAEIOU")}


def levenshtein_distance_fast(
    word1: str,
    word2: str,
    modifiers: Tuple[float, float, float] = None,
    max_dist: float = None,
    vowel_flag: bool = False,
    allow_transpositions: bool = True,
) -> float:
    while word1 and word2 and word1[0] == word2[0]:
        word1, word2 = word1[1:], word2[1:]
    while word1 and word2 and word1[-1] == word2[-1]:
        word1, word2 = word1[:-1], word2[:-1]

    if modifiers is None:
        modifiers = (0, 0, 0)
    del_cost, ins_cost, sub_mod = modifiers

    len1, len2 = len(word1), len(word2)
    if len2 == 0:
        return len1 * (min(del_cost, ins_cost) + 1)
    if len1 < len2 and del_cost == ins_cost:
        word1, word2 = word2, word1
        len1, len2 = len2, len1

    word1_ords = [ord(c) for c in word1]
    word2_ords = [ord(c) for c in word2]

    prev_prev = [0] * (len2 + 1)
    prev = list(range(len2 + 1))
    curr = prev_prev.copy()

    for i in range(1, len1 + 1):
        curr[0] = i
        row_min = i
        w1c = word1_ords[i - 1]

        for j in range(1, len2 + 1):
            w2c = word2_ords[j - 1]
            deletion = prev[j] + 1 + del_cost
            insertion = curr[j - 1] + 1 + ins_cost
            substitution = prev[j - 1] + (w1c != w2c) + sub_mod
            if vowel_flag:
                substitution += 0.1 * (w1c not in VOWELS) + 0.1 * (w2c not in VOWELS)

            v = min(deletion, insertion, substitution)

            if (
                allow_transpositions
                and i > 1
                and j > 1
                and w1c == word2_ords[j - 2]
                and word1_ords[i - 2] == w2c
            ):
                v = min(v, prev_prev[j - 2] + 1)

            curr[j] = v
            if v < row_min:
                row_min = v
        if max_dist is not None and row_min > max_dist:
            return float("inf")

        prev_prev, prev, curr = prev, curr, prev_prev

    return prev[len2]


def calc_difference(
    target_word: str,
    word: str,
    word_tokens: Dict[str, List[str]] = None,
    word_tokens_set: Dict[str, Set[str]] = None,
    target_tokens: Dict[str, List[str]] = None,
    target_tokens_set: Dict[str, Set[str]] = None,
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
    if word_tokens is None:
        word_tokens = word.split("_")

    if word_tokens_set is None:
        word_tokens_set = set(word_tokens)

    if target_tokens is None:
        target_tokens = target_word.split("_")

    if target_tokens_set is None:
        target_tokens_set = set(target_tokens)

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

    main_distance = levenshtein_distance_fast(
        target_word, word.lower()
    )  # close early exit
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
                    levenshtein_distance_fast(target_word, word.lower()) * cross_points
                )
                distance = (main_distance + token_distance) / 2
        else:
            if word_tokens_set.issubset(target_tokens_set):
                token_distance = target_tokens.index(word_tokens[0]) * cross_points
                distance = (main_distance / 2 + token_distance) / 2
            else:
                token_distance = (
                    levenshtein_distance_fast(target_word, word.lower()) * cross_points
                )
                distance = (main_distance + token_distance) / 2
    else:
        distance = levenshtein_distance_fast(target_word, word.lower()) * cross_points
    return distance


def match_fields(
    field_names: List[str], method: str, field_types=None
) -> Dict[str, str]:
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
        if target_word == "":
            break
        if target_word[0] == "_":
            target_word = target_word[1:]
        target_tokens = target_word.split("_")
        target_tokens_set = set(target_tokens)
        if t_word in word_list:
            distances.append([t_word, 0])
        else:
            target_letters = set("".join(target_tokens_set))
            filtered_word_list = [
                word for word in word_list if len(set(word) & target_letters) > 1
            ]  # filter match at least 2 letters
            if field_types is not None and field_types[t_word] is not None:
                filtered_word_list = [
                    word
                    for word in word_list
                    if provider_return_types[word] == field_types[t_word]
                ]  # filter by return type
            for word in filtered_word_list:
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
        min_value = (len(target_word) // 2 - 0.5) + 1
        if sorted_by_distance != [] and sorted_by_distance[0] != []:
            temp_min_value = sorted_by_distance[0][1]
            if temp_min_value <= min_value:
                min_value = temp_min_value
        closest_matches = [item for item in sorted_by_distance if item[1] <= min_value]
        if len(closest_matches) == 0:
            field_matches.append("")
        else:
            field_matches.append(closest_matches[0][0])

    field_match_pairs = {}
    for name, match in zip([x for x in field_names], field_matches):
        field_match_pairs[name] = match
    return field_match_pairs
