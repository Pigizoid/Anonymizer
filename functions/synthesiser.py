from pydantic import BaseModel
from faker import Faker
import mimesis
from mimesis import Generic
import inspect


fake = Faker()


generic = Generic(mimesis.locales.Locale.EN)





"""
		speed_data = [x for x in range(len(data_names))]
		for index,name,provider in zip(range(len(data_names)),data_names,method_providers):
			speed_data[index] = [provider() for x in range(amount)]
		data_set = output = [schema_model(**dict(zip(data_names, vals))) for vals in zip(*speed_data)]
"""


"""

class Synthesiser():
	
	@staticmethod
	def synthesise(schema_model,method="faker",amount=1):
		method_providers = []
		if method == "faker":
			method_providers = [fake.street_name,fake.city,fake.zipcode,fake.country]
		elif method == "mimesis":
			method_providers = [generic.address.street_name,generic.address.city,generic.address.zip_code,generic.address.country]
		
		data_set = []
		data_names = ["street", "city","zip_code", "country"]
		for x in range(amount):
			data = {}
			for name,provider in zip(data_names,method_providers):
				data[name] = provider()
			data_set.append(schema_model(**data))
		return data_set





"""


class Synthesiser:
    @staticmethod
    def list_faker_methods():
        methods = []
        methods_map = {}
        fake = Faker()
        for attr in dir(fake):
            try:
                if not attr.startswith("_") and callable(getattr(fake, attr)):
                    methods.append(attr)
                    methods_map[attr] = fake
            except:
                pass
        return (methods, methods_map)

    @staticmethod
    def list_mimesis_methods():
        methods = []
        methods_map = {}
        for provider_name in sorted(generic.__dict__.keys()):
            provider_cls = getattr(generic, provider_name)
            if inspect.isclass(provider_cls):
                try:
                    sig = inspect.signature(provider_cls)
                    if all(
                        p.default != inspect.Parameter.empty
                        or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                        for p in sig.parameters.values()
                    ):
                        instance = provider_cls()
                    else:
                        continue
                except (TypeError, ValueError):
                    continue
                for attr in dir(instance):
                    if not attr.startswith("_") and callable(getattr(instance, attr)):
                        methods.append(attr)
                        methods_map[attr] = instance
        return (methods, methods_map)

    @staticmethod
    def list_match_methods(method):
        methods = []
        methods_map = {}
        if method == "faker":
            methods, methods_map = Synthesiser.list_faker_methods()
        elif method == "mimesis":
            methods, methods_map = Synthesiser.list_mimesis_methods()
        elif method == "mixed":
            methodsF, methods_mapF = Synthesiser.list_faker_methods()

            methodsM, methods_mapM = Synthesiser.list_mimesis_methods()

            methods = methodsF
            methods.extend(methodsM)

            methods_map = methods_mapF
            methods_map.update(methods_mapM)

        methods = list(set(methods))
        # for method in methods:
        # print(f"Method: {method}, Map: {methods_map[method]}")
        return (methods, methods_map)

    @staticmethod
    def get_model_data(model: BaseModel):
        model_data = []
        for field_name, model_field in model.model_fields.items():
            model_data.append([field_name, model_field])
        return model_data

    @staticmethod
    def levenshtein_distance(word1, word2, modifiers=[0, 0, 0]):
        len_word1, len_word2 = len(word1), len(word2)
        distance_point = [
            [0 for _ in range(len_word2 + 1)] for _ in range(len_word1 + 1)
        ]
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

    @staticmethod
    def calc_difference(target_word, word, target_tokens, target_tokens_set):
        if (
            ("".join([letter[0] for letter in word.split("_")])) == target_word.lower()
        ):  # abbreviation mapping    ssn -> social_security_number
            return 0
        if (
            "".join([letter[0] for letter in target_word.lower().split("_")])
        ) == word:  # abbreviation mapping    social_security_number -> ssn
            return 0

        main_distance = Synthesiser.levenshtein_distance(
            target_word, word.lower()
        )  # close early exit
        if main_distance <= 1 or main_distance >= len(target_word):
            return main_distance

        main_distance_joined = Synthesiser.levenshtein_distance(
            target_word.replace("_", ""), word.lower()
        )
        if main_distance_joined <= 1:
            return main_distance_joined

        target_tokens = [t.lower() for t in target_tokens]
        word_tokens = [w.lower() for w in word.split("_")]

        word_tokens_set = set(word_tokens)
        if target_tokens_set == word_tokens_set:  # name_first -> first_name
            return 0

        token_distance = 0
        cross_points = len(word_tokens) * len(target_tokens)
        if len(target_tokens) != len(word_tokens):
            if len(target_tokens) < len(word_tokens):
                if target_tokens_set.issubset(word_tokens_set):
                    token_distance = word_tokens.index(target_tokens[0]) * cross_points
                    distance = (main_distance / 2 + token_distance) / 2
                else:
                    token_distance = (
                        Synthesiser.levenshtein_distance(target_word, word.lower())
                        * cross_points
                    )
                    distance = (main_distance + token_distance) / 2
            else:
                if word_tokens_set.issubset(target_tokens_set):
                    token_distance = target_tokens.index(word_tokens[0]) * cross_points
                    distance = (main_distance / 2 + token_distance) / 2
                else:
                    token_distance = (
                        Synthesiser.levenshtein_distance(target_word, word.lower())
                        * cross_points
                    )
                    distance = (main_distance + token_distance) / 2
        else:
            distance = (
                Synthesiser.levenshtein_distance(target_word, word.lower())
                * cross_points
            )
        return distance

    @staticmethod
    def generate_from_constraints(field_name, field):
        print(f"Name:{field_name}\nFields:{field}")

        return "****"

    @staticmethod
    def match_fields(schema_model, method="faker"):
        word_list, methods_map = Synthesiser.list_match_methods(method)
        model_data = Synthesiser.get_model_data(schema_model)

        field_matches = []
        for x in model_data:
            closest_matches = []
            distances = []
            target_word = x[0]
            target_word = target_word.lower()
            target_tokens = target_word.split("_")
            target_tokens_set = set(target_tokens)
            for word in word_list:
                distance = Synthesiser.calc_difference(
                    target_word, word, target_tokens, target_tokens_set
                )
                distances.append([word, distance])
            sorted_by_distance = sorted(distances, key=lambda x: x[1])
            min_value = sorted_by_distance[0][1]
            if min_value > len(target_word) // 2:
                potential_matches = []
                for word in word_list:
                    if target_word in word:
                        distance = Synthesiser.levenshtein_distance(
                            target_word, word, [-0.5, 0.5, -0.5]
                        )
                        potential_matches.append([word, distance])
                sorted_match_by_distance = sorted(potential_matches, key=lambda x: x[1])
                min_value = sorted_by_distance[0][1]
                if min_value < len(target_word) // 2:
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
        # print({field_name:match for field_name,match in zip([x[0] for x in model_data],field_matches)})
        return (([x[0] for x in model_data], field_matches), methods_map)

    @staticmethod
    def check_generation_method(name, field, match, instance_provider):
        method = "match"
        applied_constraints = "length"
        # methods = ["match","constraint"]

        # match method constraints
        #
        #

        # constraint method constraints
        #
        #

        # both method constraints
        # data type
        # Type nesting (list,Union,etc)
        #
        return (method, applied_constraints)

    @staticmethod
    def synthesise(
        schema_model, method="faker", amount=1
    ):  # methods = faker or mimesis
        #model_data = Synthesiser.get_model_data(schema_model)

        name_match_pairs, methods_map = Synthesiser.match_fields(
            schema_model, method
        )  # match_fields() returns -> ((names, matches), methods_map)
        field_match_pairs = {
            name: match for name, match in zip(name_match_pairs[0], name_match_pairs[1])
        }

        for name, field in schema_model.model_fields.items():
            generation_method, applied_constraints = (
                Synthesiser.check_generation_method(
                    name,
                    field,
                    field_match_pairs[name],
                    methods_map[field_match_pairs[name]],
                )
            )

        synthesised_data = {}
        dataset = []
        for x in range(amount):
            for name, field in schema_model.model_fields.items():
                if field_match_pairs[name] != "":
                    if method == "faker":
                        synthesised_data[name] = getattr(
                            fake, field_match_pairs[name], None
                        )()
                    elif method == "mimesis":
                        provider_instance = methods_map[field_match_pairs[name]]
                        synthesised_data[name] = getattr(
                            provider_instance, field_match_pairs[name], None
                        )()
                    elif method == "mixed":
                        provider_instance = methods_map[field_match_pairs[name]]
                        try:
                            synthesised_data[name] = str(
                                getattr(
                                    provider_instance, field_match_pairs[name], None
                                )()
                            )
                        except:
                            print("Field Error:", field_match_pairs[name])
                else:
                    synthesised_data[name] = Synthesiser.generate_from_constraints(
                        name, field
                    )
            dataset.append(schema_model(**synthesised_data))
        return dataset


#'''
