from typing import Dict
from pydantic import BaseModel
import inspect
from .calc_difference import calc_difference, levenshtein_distance

from sm.tools.model_funcs import get_model_data



class match_class():
    def match_fields(self, field_names) -> Dict[str, str]:
        field_matches = []
        for t_word in field_names:
            closest_matches = []
            distances = []
            target_word = t_word.lower()
            target_tokens = target_word.split("_")
            target_tokens_set = set(target_tokens)
            for word in self.word_list:
                distance = calc_difference(
                    target_word,
                    word,
                    self.word_tokens[word],
                    self.word_tokens_set[word],
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
                for word in self.word_list:
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

    def recursive_match_fields(
        self, schema_model, field_match_pairs=None
    ) -> Dict[str, Dict[str, str]]:  # needs test
        if field_match_pairs is None:
            field_match_pairs = {}  # because default dicts are stored in memory not by instance
        model_data = get_model_data(schema_model)
        field_names = [x[0] for x in model_data]

        if schema_model.__name__ not in field_match_pairs.keys():
            field_match_pairs[schema_model.__name__] = self.match_fields(field_names)
        for x in model_data:
            data_type = x[1].annotation
            if (
                inspect.isclass(data_type)
                and issubclass(data_type, BaseModel)
                and data_type.__name__ not in field_match_pairs.keys()
            ):
                # print(data_type.__name__, field_match_pairs.keys())
                field_match_pairs.update(self.recursive_match_fields(data_type))
        return field_match_pairs
