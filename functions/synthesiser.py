from pydantic import BaseModel

import typing
from typing import get_origin, get_args
from typing import List, Dict, Tuple, Set, Union, Literal, Annotated

from faker import Faker
import mimesis
from mimesis import Generic


import random
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


def generate_provider_return_types(provider_names, provider_instances):
    return_types = {}

    for name in provider_names:
        try:
            value = getattr(provider_instances[name], name, None)()
        except:
            continue
        if value is None:
            return_types[name] = type(None)

        try:
            int(value)
            return_types[name] = int
        except:
            pass

        try:
            float(value)
            return_types[name] = float
        except:
            pass

        try:
            if value.lower() in {"true", "false"}:
                return_types[name] = bool
        except:
            pass

        if isinstance(value, bytes):
            return_types[name] = bytes
        elif isinstance(value, bytearray):
            return_types[name] = bytearray
        elif isinstance(value, memoryview):
            return_types[name] = memoryview
        elif isinstance(value, tuple):
            return_types[name] = tuple
        elif isinstance(value, list):
            return_types[name] = list
        elif isinstance(value, set):
            return_types[name] = set
        elif isinstance(value, frozenset):
            return_types[name] = frozenset
        elif isinstance(value, dict):
            return_types[name] = dict
        elif isinstance(value, range):
            return_types[name] = range
        elif isinstance(value, slice):
            return_types[name] = slice
        elif isinstance(value, type):
            return_types[name] = type
        else:
            return_types[name] = str

        # object has been removed

    return dict(sorted(return_types.items()))


provider_return_types = {
    "Meta": str,
    "aba": str,
    "academic_degree": str,
    "address": str,
    "administrative_unit": str,
    "alphabet": list,
    "am_pm": str,
    "android_platform_token": str,
    "answer": str,
    "ascii_company_email": str,
    "ascii_email": str,
    "ascii_free_email": str,
    "ascii_safe_email": str,
    "bank": str,
    "bank_country": str,
    "basic_phone_number": str,
    "bban": str,
    "binary": bytes,
    "birthdate": str,
    "blood_type": str,
    "boolean": str,
    "bothify": str,
    "bs": str,
    "building_number": str,
    "calling_code": str,
    "catch_phrase": str,
    "century": str,
    "chrome": str,
    "city": str,
    "city_prefix": str,
    "city_suffix": str,
    "color": str,
    "color_hsl": tuple,
    "color_hsv": tuple,
    "color_name": str,
    "color_rgb": tuple,
    "color_rgb_float": tuple,
    "company": str,
    "company_email": str,
    "company_suffix": str,
    "company_type": str,
    "continent": str,
    "coordinate": str,
    "coordinates": dict,
    "country": str,
    "country_calling_code": str,
    "country_code": str,
    "country_emoji_flag": str,
    "credit_card_expire": str,
    "credit_card_full": str,
    "credit_card_number": str,
    "credit_card_provider": str,
    "credit_card_security_code": str,
    "cryptocurrency": tuple,
    "cryptocurrency_code": str,
    "cryptocurrency_iso_code": str,
    "cryptocurrency_name": str,
    "cryptocurrency_symbol": str,
    "csv": str,
    "currency": tuple,
    "currency_code": str,
    "currency_iso_code": str,
    "currency_name": str,
    "currency_symbol": str,
    "current_country": str,
    "current_country_code": str,
    "date": str,
    "date_between": str,
    "date_between_dates": str,
    "date_object": str,
    "date_of_birth": str,
    "date_this_century": str,
    "date_this_decade": str,
    "date_this_month": str,
    "date_this_year": str,
    "date_time": str,
    "date_time_ad": str,
    "date_time_between": str,
    "date_time_between_dates": str,
    "date_time_this_century": str,
    "date_time_this_decade": str,
    "date_time_this_month": str,
    "date_time_this_year": str,
    "datetime": str,
    "day_of_month": str,
    "day_of_week": str,
    "default_country": str,
    "dga": str,
    "dish": str,
    "doi": str,
    "domain_name": str,
    "domain_word": str,
    "drink": str,
    "dsv": str,
    "duration": str,
    "ean": str,
    "ean13": str,
    "ean8": str,
    "ein": str,
    "email": str,
    "emoji": str,
    "federal_subject": str,
    "file_extension": str,
    "file_name": str,
    "file_path": str,
    "firefox": str,
    "first_name": str,
    "first_name_female": str,
    "first_name_male": str,
    "first_name_nonbinary": str,
    "fixed_width": str,
    "formatted_date": str,
    "formatted_datetime": str,
    "formatted_time": str,
    "free_email": str,
    "free_email_domain": str,
    "fruit": str,
    "full_name": str,
    "future_date": str,
    "future_datetime": str,
    "gender": str,
    "gender_code": str,
    "gender_symbol": str,
    "get_current_locale": str,
    "get_providers": list,
    "get_words_list": list,
    "gmt_offset": str,
    "height": str,
    "hex_color": str,
    "hexify": str,
    "hostname": str,
    "http_method": str,
    "http_status_code": str,
    "iana_id": str,
    "iata_code": str,
    "iban": str,
    "icao_code": str,
    "identifier": str,
    "image": bytes,
    "image_url": str,
    "internet_explorer": str,
    "invalid_ssn": str,
    "ios_platform_token": str,
    "ipv4": str,
    "ipv4_network_class": str,
    "ipv4_private": str,
    "ipv4_public": str,
    "ipv6": str,
    "isbn10": str,
    "isbn13": str,
    "isd_code": str,
    "iso8601": str,
    "items": list,
    "itin": str,
    "job": str,
    "job_female": str,
    "job_male": str,
    "json": str,
    "json_bytes": bytes,
    "language": str,
    "language_code": str,
    "language_name": str,
    "last_name": str,
    "last_name_female": str,
    "last_name_male": str,
    "last_name_nonbinary": str,
    "latitude": str,
    "latlng": tuple,
    "level": str,
    "lexify": str,
    "license_plate": str,
    "linux_platform_token": str,
    "linux_processor": str,
    "local_latlng": tuple,
    "locale": str,
    "localized_ean": str,
    "localized_ean13": str,
    "localized_ean8": str,
    "location_on_land": tuple,
    "longitude": str,
    "mac_address": str,
    "mac_platform_token": str,
    "mac_processor": str,
    "md5": str,
    "military_apo": str,
    "military_dpo": str,
    "military_ship": str,
    "military_state": str,
    "mime_type": str,
    "month": str,
    "month_name": str,
    "msisdn": str,
    "name": str,
    "name_female": str,
    "name_male": str,
    "name_nonbinary": str,
    "nationality": str,
    "nic_handle": str,
    "nic_handles": list,
    "null_boolean": str,
    "numerify": str,
    "occupation": str,
    "opera": str,
    "paragraph": str,
    "paragraphs": list,
    "passport_dates": tuple,
    "passport_dob": str,
    "passport_full": str,
    "passport_gender": str,
    "passport_number": str,
    "passport_owner": tuple,
    "password": str,
    "past_date": str,
    "past_datetime": str,
    "periodicity": str,
    "phone_number": str,
    "political_views": str,
    "port_number": str,
    "postal_code": str,
    "postalcode": str,
    "postalcode_in_state": str,
    "postalcode_plus4": str,
    "postcode": str,
    "postcode_in_state": str,
    "prefecture": str,
    "prefix": str,
    "prefix_female": str,
    "prefix_male": str,
    "prefix_nonbinary": str,
    "price": str,
    "price_in_btc": str,
    "pricetag": str,
    "profile": dict,
    "province": str,
    "psv": str,
    "pybool": str,
    "pydecimal": str,
    "pydict": dict,
    "pyfloat": str,
    "pyint": str,
    "pyiterable": set,
    "pylist": list,
    "pyobject": str,
    "pyset": set,
    "pystr": str,
    "pystr_format": str,
    "pystruct": tuple,
    "pytimezone": str,
    "pytuple": tuple,
    "quote": str,
    "random_choices": list,
    "random_digit": str,
    "random_digit_above_two": str,
    "random_digit_not_null": str,
    "random_digit_not_null_or_empty": str,
    "random_digit_or_empty": str,
    "random_element": str,
    "random_elements": list,
    "random_int": str,
    "random_letter": str,
    "random_letters": list,
    "random_lowercase_letter": str,
    "random_number": str,
    "random_sample": list,
    "random_uppercase_letter": str,
    "randomize_nb_elements": str,
    "region": str,
    "reseed": str,
    "rgb_color": tuple,
    "rgb_css_color": str,
    "ripe_id": str,
    "safari": str,
    "safe_color_name": str,
    "safe_domain_name": str,
    "safe_email": str,
    "safe_hex_color": str,
    "sbn9": str,
    "secondary_address": str,
    "seed_instance": str,
    "sentence": str,
    "sentences": list,
    "sex": str,
    "sha1": str,
    "sha256": str,
    "simple_profile": dict,
    "slug": str,
    "spices": str,
    "ssn": str,
    "state": str,
    "state_abbr": str,
    "stock_exchange": str,
    "stock_name": str,
    "stock_ticker": str,
    "street_address": str,
    "street_name": str,
    "street_number": str,
    "street_suffix": str,
    "suffix": str,
    "suffix_female": str,
    "suffix_male": str,
    "suffix_nonbinary": str,
    "surname": str,
    "swift": str,
    "swift11": str,
    "swift8": str,
    "tar": bytes,
    "telephone": str,
    "text": str,
    "texts": list,
    "time": str,
    "time_delta": str,
    "time_object": str,
    "time_series": str,
    "timestamp": str,
    "timezone": str,
    "title": str,
    "tld": str,
    "tsv": str,
    "university": str,
    "unix_device": str,
    "unix_partition": str,
    "unix_time": str,
    "upc_a": str,
    "upc_e": str,
    "uri": str,
    "uri_extension": str,
    "uri_page": str,
    "uri_path": str,
    "url": str,
    "user_agent": str,
    "user_name": str,
    "username": str,
    "uuid4": str,
    "vegetable": str,
    "views_on": str,
    "vin": str,
    "week_date": str,
    "weight": str,
    "windows_platform_token": str,
    "word": str,
    "words": list,
    "worldview": str,
    "year": str,
    "zip": bytes,
    "zip_code": str,
    "zipcode": str,
    "zipcode_in_state": str,
    "zipcode_plus4": str,
}


python_builtin_types = {
    str,
    int,
    float,
    bool,
    complex,
    bytes,
    bytearray,
    memoryview,
    tuple,
    list,
    set,
    frozenset,
    dict,
    range,
    slice,
    type,
    object,
}
typing_origins = {List, Dict, Tuple, Set, Union, Literal}
recursive_types = {List, Dict, Tuple, Set, Union, Literal, list, dict, tuple, set}
all_constr_attribs = {
    "strip_whitespace",
    "to_upper",
    "to_lower",
    "strict",  # first 4 are from pydantic StringConstraint aka constr()
    "default",
    "annotation",
    "min_length",
    "max_length",
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
}


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
            if min_value > (len(target_word) // 2 - 0.5):
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
    def check_generation_constraints(name, field, match, instance_provider):
        cout = False
        if cout:
            print("__")
            print(f"	Name:{name}")
            print(f"	Field:{field}")

        # constraint sieve
        field_info = field.metadata
        constraints = {}
        constraints["required"] = getattr(field, "is_required", None)()
        """
		attributes = [
			"default",
			"annotation",
			"min_length",
			"max_length",
			"max_length",
			"pattern",
			"gt",
			"lt",
			"ge",
			"le",
			"multiple_of",
			"allow_inf_nan",
			"max_digits",
			"decimal_places"
		]
		"""
        for attr in all_constr_attribs:
            return_val = getattr(field, attr, None)
            if return_val is None:
                for field_info_element in field_info:
                    return_val = getattr(field_info_element, attr, None)
                    if return_val is not None:
                        break
            constraints[attr] = return_val

        if cout:
            print(f"Constraints:{constraints}")
        annotation = constraints["annotation"]
        constraints["origin"] = get_origin(annotation)
        constraints["args"] = get_args(annotation)

        if cout:
            print(f"	Match:{match}")
            print("__")
        return constraints

    @staticmethod
    def generate_from_constraints(field_name, constraints):
        # print(f"Name:{field_name}\nFields:{constraints}")

        return "****"

    @staticmethod
    def generate_synth_data(field_name, match_name, applied_constraints):
        # print("__")
        # print(f"	Generating for: {field_name}")
        # print(applied_constraints)
        # data_type = get_origin(applied_constraints["annotation"])
        data_type = applied_constraints["origin"]
        output_data = ""
        if data_type == Annotated:
            # print(applied_constraints)
            # print("~~")
            new_info = typing.get_args(applied_constraints["annotation"])
            new_data_type = new_info[0]
            # print(new_data_type)
            constr_constraints = {}

            for attr in all_constr_attribs:
                return_val = getattr(new_info[1], attr, None)
                constr_constraints[attr] = return_val
            # print(constr_constraints)
            new_constraints = constr_constraints
            new_constraints["required"] = True
            new_constraints["annotation"] = new_data_type
            new_constraints["origin"] = get_origin(new_data_type)
            new_constraints["args"] = get_args(new_data_type)
            # print("___")
            # print(new_constraints)
            output_data = Synthesiser.generate_synth_data(
                field_name, match_name, new_constraints
            )
            # print("returned:",output_data)
            # input("wait...")

        else:
            if not data_type:  # if data_type is None:
                data_type = applied_constraints["annotation"]
            data_args = applied_constraints["args"]
            # data_args = get_args(applied_constraints["annotation"])
            # print(f"	{data_type}\n	{data_args}")
            if data_type is None:
                output_data = None
            elif data_type in recursive_types:
                if applied_constraints["min_length"] is not None:
                    min_amount = applied_constraints["min_length"]
                else:
                    min_amount = 1
                if applied_constraints["max_length"] is not None:
                    max_amount = applied_constraints["max_length"]
                else:
                    max_amount = min_amount + 10
                if data_type in [List, list]:
                    output_data = []
                    new_applied_constraints = applied_constraints.copy()
                    for x in range(random.randint(min_amount, max_amount)):
                        chosen_type = random.choice(data_args)
                        new_applied_constraints["annotation"] = chosen_type
                        new_applied_constraints["origin"] = get_origin(chosen_type)
                        new_applied_constraints["args"] = get_args(chosen_type)
                        output_data.append(
                            Synthesiser.generate_synth_data(
                                field_name, match_name, new_applied_constraints
                            )
                        )
                elif data_type in [Dict, dict]:
                    new_left_applied_constraints = applied_constraints.copy()
                    new_right_applied_constraints = applied_constraints.copy()
                    chosen_type_left = data_args[0]
                    new_left_applied_constraints["annotation"] = chosen_type_left
                    new_left_applied_constraints["origin"] = get_origin(
                        chosen_type_left
                    )
                    new_left_applied_constraints["args"] = get_args(chosen_type_left)
                    chosen_type_right = data_args[1]
                    new_right_applied_constraints["annotation"] = chosen_type_right
                    new_right_applied_constraints["origin"] = get_origin(
                        chosen_type_right
                    )
                    new_right_applied_constraints["args"] = get_args(chosen_type_right)
                    output_data = {}
                    current_amount = 0
                    target_amount = random.randint(min_amount, max_amount)
                    current_tries = 0
                    target_amountx2 = target_amount * 2
                    # for x in range(random.randint(min_amount,max_amount)):
                    while current_amount < target_amount:
                        v1 = Synthesiser.generate_synth_data(
                            field_name, match_name, new_left_applied_constraints
                        )
                        v2 = Synthesiser.generate_synth_data(
                            field_name, match_name, new_right_applied_constraints
                        )
                        output_data[v1] = v2
                        current_amount = len(output_data)
                        current_tries += 1
                        if current_tries > target_amountx2:
                            raise Exception(
                                f"Not enough provider dictionary keys for {field_name}"
                            )

                elif data_type in [Tuple, tuple]:
                    output_data = []
                    new_applied_constraints = applied_constraints.copy()
                    for x in range(random.randint(min_amount, max_amount)):
                        chosen_type = data_args[x]
                        new_applied_constraints["annotation"] = chosen_type
                        new_applied_constraints["origin"] = get_origin(chosen_type)
                        new_applied_constraints["args"] = get_args(chosen_type)
                        output_data.append(
                            Synthesiser.generate_synth_data(
                                field_name, match_name, new_applied_constraints
                            )
                        )

                elif data_type in [Set, set]:
                    output_data = []
                    new_applied_constraints = applied_constraints.copy()
                    current_amount = 0
                    target_amount = random.randint(min_amount, max_amount)
                    chosen_type = data_args[0]
                    new_applied_constraints["annotation"] = chosen_type
                    new_applied_constraints["origin"] = get_origin(chosen_type)
                    new_applied_constraints["args"] = get_args(chosen_type)
                    current_tries = 0
                    target_amountx2 = target_amount * 2
                    while current_amount < target_amount:
                        output_data.append(
                            Synthesiser.generate_synth_data(
                                field_name, match_name, new_applied_constraints
                            )
                        )
                        current_amount = len(output_data)
                        current_tries += 1
                        if current_tries > target_amountx2:
                            raise Exception(
                                f"Not enough provider dictionary keys for {field_name}"
                            )

                elif data_type == Union:
                    new_applied_constraints = applied_constraints.copy()
                    chosen_type = random.choice(data_args)
                    new_applied_constraints["annotation"] = chosen_type
                    new_applied_constraints["origin"] = get_origin(chosen_type)
                    new_applied_constraints["args"] = get_args(chosen_type)
                    output_data = Synthesiser.generate_synth_data(
                        field_name, match_name, new_applied_constraints
                    )

                elif data_type == Literal:
                    # print("Literal Here")
                    # print(data_args)
                    output_data = random.choice(data_args)
                    # print(output_data)
                else:
                    raise Exception(
                        f"Recersive data type| {data_type} : {data_args} |not handled"
                    )
                # print("	recursive")
            elif data_type in python_builtin_types:
                # print("	generate data")
                func = Synthesiser.resolved_methods.get(match_name)
                if not func:  # or applied_constraints["pattern"] is not None:
                    output_data = Synthesiser.generate_from_constraints(
                        match_name, applied_constraints
                    )
                else:
                    result = func()
                    output_data = result if isinstance(result, str) else str(result)
                    # output_data = str(getattr(provider_instance, match_name, None)())
            else:
                if inspect.isclass(data_type):
                    if issubclass(data_type, BaseModel):
                        pass  # print("	nest")
                    output_data = None
                else:
                    output_data = None

        # print(f"Data: {output_data}")
        # print("__")
        return output_data

    @staticmethod
    def synthesise(schema_model, method="faker", amount=1):
        # methods = faker or mimesis
        # model_data = Synthesiser.get_model_data(schema_model)
        if amount == 0:
            return []
        name_match_pairs, methods_map = Synthesiser.match_fields(
            schema_model, method
        )  # match_fields() returns -> ((names, matches), methods_map)

        field_match_pairs = {
            name: match for name, match in zip(name_match_pairs[0], name_match_pairs[1])
        }
        # print(field_match_pairs)
        applied_constraints = {}
        for name, field in schema_model.model_fields.items():
            if field_match_pairs[name] == "":
                mm = ""
            else:
                mm = methods_map[field_match_pairs[name]]
            applied_constraints[name] = Synthesiser.check_generation_constraints(
                name, field, field_match_pairs[name], mm
            )
        # print(applied_constraints)

        resolved_methods = {}
        for match_name in name_match_pairs[1]:
            if match_name != "":
                provider_instance = methods_map[match_name]
                resolved_methods[match_name] = getattr(provider_instance, match_name)
            else:
                resolved_methods[match_name] = None
        Synthesiser.resolved_methods = resolved_methods
        # print(resolved_methods)

        synthesised_data = {}
        dataset = []
        for x in range(amount):
            # print("__")
            for name, field in schema_model.model_fields.items():
                # print(f"Field:{name}")
                if not applied_constraints[name]["required"]:
                    if random.randint(1, 2) == 1:
                        continue

                if field_match_pairs[name] != "":
                    provider_instance = methods_map[field_match_pairs[name]]
                else:
                    provider_instance = ""
                synthesised_data[name] = Synthesiser.generate_synth_data(
                    name, field_match_pairs[name], applied_constraints[name]
                )
                # print(f"	Data:{synthesised_data[name]}")
            # print("__")
            dataset.append(schema_model(**synthesised_data))
        return dataset


#'''


def setup_func():
    names, instances = Synthesiser.list_match_methods("mixed")
    print(generate_provider_return_types(names, instances))


if __name__ == "__main__":
    setup_func()
