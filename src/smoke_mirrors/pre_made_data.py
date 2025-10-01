from typing import List, Dict, Tuple, Set, Union, Literal, Optional, Any

from decimal import Decimal

from faker import Faker
import mimesis
from mimesis import Generic
import inspect
from enum import Enum
                
                
class provider_return_types(Enum):
    Meta = str
    aba = str
    academic_degree = str
    address = str
    administrative_unit = str
    alphabet = list
    am_pm = str
    android_platform_token = str
    answer = str
    ascii_company_email = str
    ascii_email = str
    ascii_free_email = str
    ascii_safe_email = str
    bank = str
    bank_country = str
    basic_phone_number = str
    bban = str
    binary = bytes
    birthdate = str
    blood_type = str
    boolean = str
    bothify = str
    bs = str
    building_number = str
    calling_code = str
    catch_phrase = str
    century = str
    chrome = str
    city = str
    city_prefix = str
    city_suffix = str
    color = str
    color_hsl = tuple
    color_hsv = tuple
    color_name = str
    color_rgb = tuple
    color_rgb_float = tuple
    company = str
    company_email = str
    company_suffix = str
    company_type = str
    continent = str
    coordinate = str
    coordinates = dict
    country = str
    country_calling_code = str
    country_code = str
    country_emoji_flag = str
    credit_card_expire = str
    credit_card_full = str
    credit_card_number = str
    credit_card_provider = str
    credit_card_security_code = str
    cryptocurrency = tuple
    cryptocurrency_code = str
    cryptocurrency_iso_code = str
    cryptocurrency_name = str
    cryptocurrency_symbol = str
    csv = str
    currency = tuple
    currency_code = str
    currency_iso_code = str
    currency_name = str
    currency_symbol = str
    current_country = str
    current_country_code = str
    date = str
    date_between = str
    date_between_dates = str
    date_object = str
    date_of_birth = str
    date_this_century = str
    date_this_decade = str
    date_this_month = str
    date_this_year = str
    date_time = str
    date_time_ad = str
    date_time_between = str
    date_time_between_dates = str
    date_time_this_century = str
    date_time_this_decade = str
    date_time_this_month = str
    date_time_this_year = str
    datetime = str
    day_of_month = str
    day_of_week = str
    default_country = str
    dga = str
    dish = str
    doi = str
    domain_name = str
    domain_word = str
    drink = str
    dsv = str
    duration = str
    ean = str
    ean13 = str
    ean8 = str
    ein = str
    email = str
    emoji = str
    federal_subject = str
    file_extension = str
    file_name = str
    file_path = str
    firefox = str
    first_name = str
    first_name_female = str
    first_name_male = str
    first_name_nonbinary = str
    fixed_width = str
    formatted_date = str
    formatted_datetime = str
    formatted_time = str
    free_email = str
    free_email_domain = str
    fruit = str
    full_name = str
    future_date = str
    future_datetime = str
    gender = str
    gender_code = str
    gender_symbol = str
    get_current_locale = str
    get_providers = list
    get_words_list = list
    gmt_offset = str
    height = str
    hex_color = str
    hexify = str
    hostname = str
    http_method = str
    http_status_code = str
    iana_id = str
    iata_code = str
    iban = str
    icao_code = str
    identifier = str
    image = bytes
    image_url = str
    internet_explorer = str
    invalid_ssn = str
    ios_platform_token = str
    ipv4 = str
    ipv4_network_class = str
    ipv4_private = str
    ipv4_public = str
    ipv6 = str
    isbn10 = str
    isbn13 = str
    isd_code = str
    iso8601 = str
    items = list
    itin = str
    job = str
    job_female = str
    job_male = str
    json = str
    json_bytes = bytes
    language = str
    language_code = str
    language_name = str
    last_name = str
    last_name_female = str
    last_name_male = str
    last_name_nonbinary = str
    latitude = str
    latlng = tuple
    level = str
    lexify = str
    license_plate = str
    linux_platform_token = str
    linux_processor = str
    local_latlng = tuple
    locale = str
    localized_ean = str
    localized_ean13 = str
    localized_ean8 = str
    location_on_land = tuple
    longitude = str
    mac_address = str
    mac_platform_token = str
    mac_processor = str
    md5 = str
    military_apo = str
    military_dpo = str
    military_ship = str
    military_state = str
    mime_type = str
    month = str
    month_name = str
    msisdn = str
    name = str
    name_female = str
    name_male = str
    name_nonbinary = str
    nationality = str
    nic_handle = str
    nic_handles = list
    null_boolean = str
    numerify = str
    occupation = str
    opera = str
    paragraph = str
    paragraphs = list
    passport_dates = tuple
    passport_dob = str
    passport_full = str
    passport_gender = str
    passport_number = str
    passport_owner = tuple
    password = str
    past_date = str
    past_datetime = str
    periodicity = str
    phone_number = str
    political_views = str
    port_number = str
    postal_code = str
    postalcode = str
    postalcode_in_state = str
    postalcode_plus4 = str
    postcode = str
    postcode_in_state = str
    prefecture = str
    prefix = str
    prefix_female = str
    prefix_male = str
    prefix_nonbinary = str
    price = str
    price_in_btc = str
    pricetag = str
    profile = dict
    province = str
    psv = str
    pybool = str
    pydecimal = str
    pydict = dict
    pyfloat = str
    pyint = str
    pyiterable = tuple
    pylist = list
    pyobject = str
    pyset = set
    pystr = str
    pystr_format = str
    pystruct = tuple
    pytimezone = str
    pytuple = tuple
    quote = str
    random_choices = list
    random_digit = str
    random_digit_above_two = str
    random_digit_not_null = str
    random_digit_not_null_or_empty = str
    random_digit_or_empty = str
    random_element = str
    random_elements = list
    random_int = str
    random_letter = str
    random_letters = list
    random_lowercase_letter = str
    random_number = str
    random_sample = list
    random_uppercase_letter = str
    randomize_nb_elements = str
    region = str
    reseed = str
    rgb_color = tuple
    rgb_css_color = str
    ripe_id = str
    safari = str
    safe_color_name = str
    safe_domain_name = str
    safe_email = str
    safe_hex_color = str
    sbn9 = str
    secondary_address = str
    seed_instance = str
    sentence = str
    sentences = list
    sex = str
    sha1 = str
    sha256 = str
    simple_profile = dict
    slug = str
    spices = str
    ssn = str
    state = str
    state_abbr = str
    stock_exchange = str
    stock_name = str
    stock_ticker = str
    street_address = str
    street_name = str
    street_number = str
    street_suffix = str
    suffix = str
    suffix_female = str
    suffix_male = str
    suffix_nonbinary = str
    surname = str
    swift = str
    swift11 = str
    swift8 = str
    tar = bytes
    telephone = str
    text = str
    texts = list
    time = str
    time_delta = str
    time_object = str
    time_series = str
    timestamp = str
    timezone = str
    title = str
    tld = str
    tsv = str
    university = str
    unix_device = str
    unix_partition = str
    unix_time = str
    upc_a = str
    upc_e = str
    uri = str
    uri_extension = str
    uri_page = str
    uri_path = str
    url = str
    user_agent = str
    user_name = str
    username = str
    uuid4 = str
    vegetable = str
    views_on = str
    vin = str
    week_date = str
    weight = str
    windows_platform_token = str
    word = str
    words = list
    worldview = str
    year = str
    zip = bytes
    zip_code = str
    zipcode = str
    zipcode_in_state = str
    zipcode_plus4 = str
provider_methods = {
    'faker' : {
        'word_list': ['city_prefix', 'month', 'psv', 'past_datetime', 'suffix_nonbinary', 'random_digit_not_null', 'future_date', 'postalcode_in_state', 'time_object', 'safari', 'name_male', 'date_time_ad', 'pydict', 'aba', 'ascii_safe_email', 'date_time_this_decade', 'color', 'current_country', 'pytuple', 'uuid4', 'cryptocurrency_code', 'date_time_this_century', 'coordinate', 'file_extension', 'ipv6', 'locale', 'msisdn', 'last_name_female', 'ipv4_private', 'sha1', 'date_time_between_dates', 'iana_id', 'latlng', 'city_suffix', 'currency_symbol', 'domain_name', 'get_formatter', 'dsv', 'paragraphs', 'sbn9', 'user_agent', 'military_dpo', 'localized_ean13', 'suffix_female', 'date_between', 'ipv4_network_class', 'safe_color_name', 'ripe_id', 'zip', 'ean', 'zipcode_plus4', 'past_date', 'color_rgb', 'credit_card_provider', 'prefix_nonbinary', 'street_address', 'hexify', 'windows_platform_token', 'date_time', 'credit_card_number', 'android_platform_token', 'currency_code', 'first_name_nonbinary', 'future_datetime', 'local_latlng', 'seed_instance', 'time_series', 'random_digit_or_empty', 'get_arguments', 'pylist', 'localized_ean', 'currency_name', 'first_name_female', 'doi', 'format', 'safe_email', 'numerify', 'rgb_color', 'job_male', 'fixed_width', 'uri_page', 'current_country_code', 'passport_full', 'random_digit', 'location_on_land', 'domain_word', 'pydecimal', 'street_name', 'am_pm', 'ean8', 'get_words_list', 'month_name', 'date_this_month', 'company', 'passport_owner', 'port_number', 'pyfloat', 'first_name_male', 'date_this_decade', 'passport_dob', 'word', 'catch_phrase', 'pystruct', 'sentences', 'date_time_between', 'city', 'opera', 'timezone', 'time_delta', 'hex_color', 'md5', 'uri_path', 'military_state', 'pytimezone', 'name', 'secondary_address', 'passport_dates', 'swift', 'cryptocurrency_name', 'last_name_male', 'date_of_birth', 'address', 'date_time_this_year', 'date_time_this_month', 'chrome', 'dga', 'name_nonbinary', 'get_providers', 'random_uppercase_letter', 'uri_extension', 'tar', 'ein', 'binary', 'day_of_week', 'file_name', 'first_name', 'file_path', 'http_status_code', 'passport_gender', 'seed_locale', 'suffix', 'words', 'basic_phone_number', 'color_rgb_float', 'credit_card_expire', 'null_boolean', 'http_method', 'pyobject', 'random_choices', 'emoji', 'military_ship', 'day_of_month', 'language_code', 'pyiterable', 'street_suffix', 'company_email', 'text', 'password', 'cryptocurrency', 'time', 'century', 'color_hsl', 'safe_hex_color', 'internet_explorer', 'swift8', 'upc_a', 'longitude', 'isbn10', 'ipv4_public', 'license_plate', 'postcode_in_state', 'ascii_company_email', 'ean13', 'set_formatter', 'pyint', 'state_abbr', 'date_this_year', 'postcode', 'prefix_female', 'unix_device', 'date_between_dates', 'firefox', 'json_bytes', 'year', 'slug', 'zipcode', 'tsv', 'tld', 'linux_processor', 'nic_handles', 'latitude', 'random_element', 'building_number', 'credit_card_full', 'passport_number', 'bban', 'localized_ean8', 'provider', 'items', 'add_provider', 'free_email_domain', 'image', 'random_digit_above_two', 'image_url', 'country_calling_code', 'ascii_email', 'prefix_male', 'ipv4', 'job', 'pricetag', 'prefix', 'credit_card_security_code', 'zipcode_in_state', 'iso8601', 'safe_domain_name', 'iban', 'randomize_nb_elements', 'bs', 'json', 'job_female', 'military_apo', 'random_sample', 'color_hsv', 'country_code', 'currency', 'sentence', 'bank_country', 'texts', 'paragraph', 'unix_partition', 'country', 'random_int', 'csv', 'free_email', 'set_arguments', 'random_digit_not_null_or_empty', 'vin', 'postalcode_plus4', 'boolean', 'url', 'mac_platform_token', 'mime_type', 'random_elements', 'random_letter', 'random_letters', 'itin', 'lexify', 'last_name', 'nic_handle', 'sha256', 'uri', 'phone_number', 'company_suffix', 'bothify', 'del_arguments', 'parse', 'xml', 'invalid_ssn', 'last_name_nonbinary', 'mac_address', 'mac_processor', 'pyset', 'user_name', 'enum', 'hostname', 'ios_platform_token', 'name_female', 'pystr_format', 'postalcode', 'pystr', 'simple_profile', 'unix_time', 'suffix_male', 'random_lowercase_letter', 'language_name', 'administrative_unit', 'ascii_free_email', 'email', 'state', 'date_object', 'rgb_css_color', 'ssn', 'upc_e', 'linux_platform_token', 'profile', 'swift11', 'pybool', 'color_name', 'date_this_century', 'isbn13', 'random_number', 'date'],
        'word_tokens': {'city_prefix': ['city', 'prefix'], 'month': ['month'], 'psv': ['psv'], 'past_datetime': ['past', 'datetime'], 'suffix_nonbinary': ['suffix', 'nonbinary'], 'random_digit_not_null': ['random', 'digit', 'not', 'null'], 'future_date': ['future', 'date'], 'postalcode_in_state': ['postalcode', 'in', 'state'], 'time_object': ['time', 'object'], 'safari': ['safari'], 'name_male': ['name', 'male'], 'date_time_ad': ['date', 'time', 'ad'], 'pydict': ['pydict'], 'aba': ['aba'], 'ascii_safe_email': ['ascii', 'safe', 'email'], 'date_time_this_decade': ['date', 'time', 'this', 'decade'], 'color': ['color'], 'current_country': ['current', 'country'], 'pytuple': ['pytuple'], 'uuid4': ['uuid4'], 'cryptocurrency_code': ['cryptocurrency', 'code'], 'date_time_this_century': ['date', 'time', 'this', 'century'], 'coordinate': ['coordinate'], 'file_extension': ['file', 'extension'], 'ipv6': ['ipv6'], 'locale': ['locale'], 'msisdn': ['msisdn'], 'last_name_female': ['last', 'name', 'female'], 'ipv4_private': ['ipv4', 'private'], 'sha1': ['sha1'], 'date_time_between_dates': ['date', 'time', 'between', 'dates'], 'iana_id': ['iana', 'id'], 'latlng': ['latlng'], 'city_suffix': ['city', 'suffix'], 'currency_symbol': ['currency', 'symbol'], 'domain_name': ['domain', 'name'], 'get_formatter': ['get', 'formatter'], 'dsv': ['dsv'], 'paragraphs': ['paragraphs'], 'sbn9': ['sbn9'], 'user_agent': ['user', 'agent'], 'military_dpo': ['military', 'dpo'], 'localized_ean13': ['localized', 'ean13'], 'suffix_female': ['suffix', 'female'], 'date_between': ['date', 'between'], 'ipv4_network_class': ['ipv4', 'network', 'class'], 'safe_color_name': ['safe', 'color', 'name'], 'ripe_id': ['ripe', 'id'], 'zip': ['zip'], 'ean': ['ean'], 'zipcode_plus4': ['zipcode', 'plus4'], 'past_date': ['past', 'date'], 'color_rgb': ['color', 'rgb'], 'credit_card_provider': ['credit', 'card', 'provider'], 'prefix_nonbinary': ['prefix', 'nonbinary'], 'street_address': ['street', 'address'], 'hexify': ['hexify'], 'windows_platform_token': ['windows', 'platform', 'token'], 'date_time': ['date', 'time'], 'credit_card_number': ['credit', 'card', 'number'], 'android_platform_token': ['android', 'platform', 'token'], 'currency_code': ['currency', 'code'], 'first_name_nonbinary': ['first', 'name', 'nonbinary'], 'future_datetime': ['future', 'datetime'], 'local_latlng': ['local', 'latlng'], 'seed_instance': ['seed', 'instance'], 'time_series': ['time', 'series'], 'random_digit_or_empty': ['random', 'digit', 'or', 'empty'], 'get_arguments': ['get', 'arguments'], 'pylist': ['pylist'], 'localized_ean': ['localized', 'ean'], 'currency_name': ['currency', 'name'], 'first_name_female': ['first', 'name', 'female'], 'doi': ['doi'], 'format': ['format'], 'safe_email': ['safe', 'email'], 'numerify': ['numerify'], 'rgb_color': ['rgb', 'color'], 'job_male': ['job', 'male'], 'fixed_width': ['fixed', 'width'], 'uri_page': ['uri', 'page'], 'current_country_code': ['current', 'country', 'code'], 'passport_full': ['passport', 'full'], 'random_digit': ['random', 'digit'], 'location_on_land': ['location', 'on', 'land'], 'domain_word': ['domain', 'word'], 'pydecimal': ['pydecimal'], 'street_name': ['street', 'name'], 'am_pm': ['am', 'pm'], 'ean8': ['ean8'], 'get_words_list': ['get', 'words', 'list'], 'month_name': ['month', 'name'], 'date_this_month': ['date', 'this', 'month'], 'company': ['company'], 'passport_owner': ['passport', 'owner'], 'port_number': ['port', 'number'], 'pyfloat': ['pyfloat'], 'first_name_male': ['first', 'name', 'male'], 'date_this_decade': ['date', 'this', 'decade'], 'passport_dob': ['passport', 'dob'], 'word': ['word'], 'catch_phrase': ['catch', 'phrase'], 'pystruct': ['pystruct'], 'sentences': ['sentences'], 'date_time_between': ['date', 'time', 'between'], 'city': ['city'], 'opera': ['opera'], 'timezone': ['timezone'], 'time_delta': ['time', 'delta'], 'hex_color': ['hex', 'color'], 'md5': ['md5'], 'uri_path': ['uri', 'path'], 'military_state': ['military', 'state'], 'pytimezone': ['pytimezone'], 'name': ['name'], 'secondary_address': ['secondary', 'address'], 'passport_dates': ['passport', 'dates'], 'swift': ['swift'], 'cryptocurrency_name': ['cryptocurrency', 'name'], 'last_name_male': ['last', 'name', 'male'], 'date_of_birth': ['date', 'of', 'birth'], 'address': ['address'], 'date_time_this_year': ['date', 'time', 'this', 'year'], 'date_time_this_month': ['date', 'time', 'this', 'month'], 'chrome': ['chrome'], 'dga': ['dga'], 'name_nonbinary': ['name', 'nonbinary'], 'get_providers': ['get', 'providers'], 'random_uppercase_letter': ['random', 'uppercase', 'letter'], 'uri_extension': ['uri', 'extension'], 'tar': ['tar'], 'ein': ['ein'], 'binary': ['binary'], 'day_of_week': ['day', 'of', 'week'], 'file_name': ['file', 'name'], 'first_name': ['first', 'name'], 'file_path': ['file', 'path'], 'http_status_code': ['http', 'status', 'code'], 'passport_gender': ['passport', 'gender'], 'seed_locale': ['seed', 'locale'], 'suffix': ['suffix'], 'words': ['words'], 'basic_phone_number': ['basic', 'phone', 'number'], 'color_rgb_float': ['color', 'rgb', 'float'], 'credit_card_expire': ['credit', 'card', 'expire'], 'null_boolean': ['null', 'boolean'], 'http_method': ['http', 'method'], 'pyobject': ['pyobject'], 'random_choices': ['random', 'choices'], 'emoji': ['emoji'], 'military_ship': ['military', 'ship'], 'day_of_month': ['day', 'of', 'month'], 'language_code': ['language', 'code'], 'pyiterable': ['pyiterable'], 'street_suffix': ['street', 'suffix'], 'company_email': ['company', 'email'], 'text': ['text'], 'password': ['password'], 'cryptocurrency': ['cryptocurrency'], 'time': ['time'], 'century': ['century'], 'color_hsl': ['color', 'hsl'], 'safe_hex_color': ['safe', 'hex', 'color'], 'internet_explorer': ['internet', 'explorer'], 'swift8': ['swift8'], 'upc_a': ['upc', 'a'], 'longitude': ['longitude'], 'isbn10': ['isbn10'], 'ipv4_public': ['ipv4', 'public'], 'license_plate': ['license', 'plate'], 'postcode_in_state': ['postcode', 'in', 'state'], 'ascii_company_email': ['ascii', 'company', 'email'], 'ean13': ['ean13'], 'set_formatter': ['set', 'formatter'], 'pyint': ['pyint'], 'state_abbr': ['state', 'abbr'], 'date_this_year': ['date', 'this', 'year'], 'postcode': ['postcode'], 'prefix_female': ['prefix', 'female'], 'unix_device': ['unix', 'device'], 'date_between_dates': ['date', 'between', 'dates'], 'firefox': ['firefox'], 'json_bytes': ['json', 'bytes'], 'year': ['year'], 'slug': ['slug'], 'zipcode': ['zipcode'], 'tsv': ['tsv'], 'tld': ['tld'], 'linux_processor': ['linux', 'processor'], 'nic_handles': ['nic', 'handles'], 'latitude': ['latitude'], 'random_element': ['random', 'element'], 'building_number': ['building', 'number'], 'credit_card_full': ['credit', 'card', 'full'], 'passport_number': ['passport', 'number'], 'bban': ['bban'], 'localized_ean8': ['localized', 'ean8'], 'provider': ['provider'], 'items': ['items'], 'add_provider': ['add', 'provider'], 'free_email_domain': ['free', 'email', 'domain'], 'image': ['image'], 'random_digit_above_two': ['random', 'digit', 'above', 'two'], 'image_url': ['image', 'url'], 'country_calling_code': ['country', 'calling', 'code'], 'ascii_email': ['ascii', 'email'], 'prefix_male': ['prefix', 'male'], 'ipv4': ['ipv4'], 'job': ['job'], 'pricetag': ['pricetag'], 'prefix': ['prefix'], 'credit_card_security_code': ['credit', 'card', 'security', 'code'], 'zipcode_in_state': ['zipcode', 'in', 'state'], 'iso8601': ['iso8601'], 'safe_domain_name': ['safe', 'domain', 'name'], 'iban': ['iban'], 'randomize_nb_elements': ['randomize', 'nb', 'elements'], 'bs': ['bs'], 'json': ['json'], 'job_female': ['job', 'female'], 'military_apo': ['military', 'apo'], 'random_sample': ['random', 'sample'], 'color_hsv': ['color', 'hsv'], 'country_code': ['country', 'code'], 'currency': ['currency'], 'sentence': ['sentence'], 'bank_country': ['bank', 'country'], 'texts': ['texts'], 'paragraph': ['paragraph'], 'unix_partition': ['unix', 'partition'], 'country': ['country'], 'random_int': ['random', 'int'], 'csv': ['csv'], 'free_email': ['free', 'email'], 'set_arguments': ['set', 'arguments'], 'random_digit_not_null_or_empty': ['random', 'digit', 'not', 'null', 'or', 'empty'], 'vin': ['vin'], 'postalcode_plus4': ['postalcode', 'plus4'], 'boolean': ['boolean'], 'url': ['url'], 'mac_platform_token': ['mac', 'platform', 'token'], 'mime_type': ['mime', 'type'], 'random_elements': ['random', 'elements'], 'random_letter': ['random', 'letter'], 'random_letters': ['random', 'letters'], 'itin': ['itin'], 'lexify': ['lexify'], 'last_name': ['last', 'name'], 'nic_handle': ['nic', 'handle'], 'sha256': ['sha256'], 'uri': ['uri'], 'phone_number': ['phone', 'number'], 'company_suffix': ['company', 'suffix'], 'bothify': ['bothify'], 'del_arguments': ['del', 'arguments'], 'parse': ['parse'], 'xml': ['xml'], 'invalid_ssn': ['invalid', 'ssn'], 'last_name_nonbinary': ['last', 'name', 'nonbinary'], 'mac_address': ['mac', 'address'], 'mac_processor': ['mac', 'processor'], 'pyset': ['pyset'], 'user_name': ['user', 'name'], 'enum': ['enum'], 'hostname': ['hostname'], 'ios_platform_token': ['ios', 'platform', 'token'], 'name_female': ['name', 'female'], 'pystr_format': ['pystr', 'format'], 'postalcode': ['postalcode'], 'pystr': ['pystr'], 'simple_profile': ['simple', 'profile'], 'unix_time': ['unix', 'time'], 'suffix_male': ['suffix', 'male'], 'random_lowercase_letter': ['random', 'lowercase', 'letter'], 'language_name': ['language', 'name'], 'administrative_unit': ['administrative', 'unit'], 'ascii_free_email': ['ascii', 'free', 'email'], 'email': ['email'], 'state': ['state'], 'date_object': ['date', 'object'], 'rgb_css_color': ['rgb', 'css', 'color'], 'ssn': ['ssn'], 'upc_e': ['upc', 'e'], 'linux_platform_token': ['linux', 'platform', 'token'], 'profile': ['profile'], 'swift11': ['swift11'], 'pybool': ['pybool'], 'color_name': ['color', 'name'], 'date_this_century': ['date', 'this', 'century'], 'isbn13': ['isbn13'], 'random_number': ['random', 'number'], 'date': ['date']},
        'word_tokens_set': {'city_prefix': {'prefix', 'city'}, 'month': {'month'}, 'psv': {'psv'}, 'past_datetime': {'datetime', 'past'}, 'suffix_nonbinary': {'nonbinary', 'suffix'}, 'random_digit_not_null': {'null', 'not', 'random', 'digit'}, 'future_date': {'future', 'date'}, 'postalcode_in_state': {'postalcode', 'in', 'state'}, 'time_object': {'object', 'time'}, 'safari': {'safari'}, 'name_male': {'name', 'male'}, 'date_time_ad': {'ad', 'date', 'time'}, 'pydict': {'pydict'}, 'aba': {'aba'}, 'ascii_safe_email': {'ascii', 'safe', 'email'}, 'date_time_this_decade': {'this', 'decade', 'date', 'time'}, 'color': {'color'}, 'current_country': {'current', 'country'}, 'pytuple': {'pytuple'}, 'uuid4': {'uuid4'}, 'cryptocurrency_code': {'cryptocurrency', 'code'}, 'date_time_this_century': {'this', 'century', 'date', 'time'}, 'coordinate': {'coordinate'}, 'file_extension': {'extension', 'file'}, 'ipv6': {'ipv6'}, 'locale': {'locale'}, 'msisdn': {'msisdn'}, 'last_name_female': {'name', 'female', 'last'}, 'ipv4_private': {'private', 'ipv4'}, 'sha1': {'sha1'}, 'date_time_between_dates': {'dates', 'date', 'between', 'time'}, 'iana_id': {'id', 'iana'}, 'latlng': {'latlng'}, 'city_suffix': {'city', 'suffix'}, 'currency_symbol': {'currency', 'symbol'}, 'domain_name': {'domain', 'name'}, 'get_formatter': {'get', 'formatter'}, 'dsv': {'dsv'}, 'paragraphs': {'paragraphs'}, 'sbn9': {'sbn9'}, 'user_agent': {'agent', 'user'}, 'military_dpo': {'dpo', 'military'}, 'localized_ean13': {'localized', 'ean13'}, 'suffix_female': {'female', 'suffix'}, 'date_between': {'date', 'between'}, 'ipv4_network_class': {'class', 'network', 'ipv4'}, 'safe_color_name': {'safe', 'color', 'name'}, 'ripe_id': {'id', 'ripe'}, 'zip': {'zip'}, 'ean': {'ean'}, 'zipcode_plus4': {'plus4', 'zipcode'}, 'past_date': {'past', 'date'}, 'color_rgb': {'rgb', 'color'}, 'credit_card_provider': {'provider', 'credit', 'card'}, 'prefix_nonbinary': {'nonbinary', 'prefix'}, 'street_address': {'street', 'address'}, 'hexify': {'hexify'}, 'windows_platform_token': {'token', 'platform', 'windows'}, 'date_time': {'date', 'time'}, 'credit_card_number': {'credit', 'number', 'card'}, 'android_platform_token': {'token', 'platform', 'android'}, 'currency_code': {'currency', 'code'}, 'first_name_nonbinary': {'first', 'name', 'nonbinary'}, 'future_datetime': {'datetime', 'future'}, 'local_latlng': {'local', 'latlng'}, 'seed_instance': {'instance', 'seed'}, 'time_series': {'series', 'time'}, 'random_digit_or_empty': {'empty', 'or', 'random', 'digit'}, 'get_arguments': {'get', 'arguments'}, 'pylist': {'pylist'}, 'localized_ean': {'ean', 'localized'}, 'currency_name': {'currency', 'name'}, 'first_name_female': {'first', 'name', 'female'}, 'doi': {'doi'}, 'format': {'format'}, 'safe_email': {'safe', 'email'}, 'numerify': {'numerify'}, 'rgb_color': {'rgb', 'color'}, 'job_male': {'job', 'male'}, 'fixed_width': {'fixed', 'width'}, 'uri_page': {'page', 'uri'}, 'current_country_code': {'current', 'code', 'country'}, 'passport_full': {'full', 'passport'}, 'random_digit': {'random', 'digit'}, 'location_on_land': {'land', 'on', 'location'}, 'domain_word': {'domain', 'word'}, 'pydecimal': {'pydecimal'}, 'street_name': {'street', 'name'}, 'am_pm': {'am', 'pm'}, 'ean8': {'ean8'}, 'get_words_list': {'get', 'words', 'list'}, 'month_name': {'month', 'name'}, 'date_this_month': {'this', 'month', 'date'}, 'company': {'company'}, 'passport_owner': {'owner', 'passport'}, 'port_number': {'port', 'number'}, 'pyfloat': {'pyfloat'}, 'first_name_male': {'first', 'name', 'male'}, 'date_this_decade': {'this', 'date', 'decade'}, 'passport_dob': {'dob', 'passport'}, 'word': {'word'}, 'catch_phrase': {'catch', 'phrase'}, 'pystruct': {'pystruct'}, 'sentences': {'sentences'}, 'date_time_between': {'date', 'between', 'time'}, 'city': {'city'}, 'opera': {'opera'}, 'timezone': {'timezone'}, 'time_delta': {'delta', 'time'}, 'hex_color': {'hex', 'color'}, 'md5': {'md5'}, 'uri_path': {'uri', 'path'}, 'military_state': {'state', 'military'}, 'pytimezone': {'pytimezone'}, 'name': {'name'}, 'secondary_address': {'secondary', 'address'}, 'passport_dates': {'dates', 'passport'}, 'swift': {'swift'}, 'cryptocurrency_name': {'cryptocurrency', 'name'}, 'last_name_male': {'name', 'last', 'male'}, 'date_of_birth': {'birth', 'of', 'date'}, 'address': {'address'}, 'date_time_this_year': {'this', 'year', 'date', 'time'}, 'date_time_this_month': {'this', 'month', 'date', 'time'}, 'chrome': {'chrome'}, 'dga': {'dga'}, 'name_nonbinary': {'name', 'nonbinary'}, 'get_providers': {'get', 'providers'}, 'random_uppercase_letter': {'letter', 'random', 'uppercase'}, 'uri_extension': {'extension', 'uri'}, 'tar': {'tar'}, 'ein': {'ein'}, 'binary': {'binary'}, 'day_of_week': {'day', 'of', 'week'}, 'file_name': {'name', 'file'}, 'first_name': {'first', 'name'}, 'file_path': {'file', 'path'}, 'http_status_code': {'code', 'status', 'http'}, 'passport_gender': {'gender', 'passport'}, 'seed_locale': {'locale', 'seed'}, 'suffix': {'suffix'}, 'words': {'words'}, 'basic_phone_number': {'basic', 'number', 'phone'}, 'color_rgb_float': {'rgb', 'color', 'float'}, 'credit_card_expire': {'credit', 'expire', 'card'}, 'null_boolean': {'null', 'boolean'}, 'http_method': {'method', 'http'}, 'pyobject': {'pyobject'}, 'random_choices': {'random', 'choices'}, 'emoji': {'emoji'}, 'military_ship': {'ship', 'military'}, 'day_of_month': {'day', 'month', 'of'}, 'language_code': {'language', 'code'}, 'pyiterable': {'pyiterable'}, 'street_suffix': {'street', 'suffix'}, 'company_email': {'company', 'email'}, 'text': {'text'}, 'password': {'password'}, 'cryptocurrency': {'cryptocurrency'}, 'time': {'time'}, 'century': {'century'}, 'color_hsl': {'color', 'hsl'}, 'safe_hex_color': {'safe', 'color', 'hex'}, 'internet_explorer': {'internet', 'explorer'}, 'swift8': {'swift8'}, 'upc_a': {'a', 'upc'}, 'longitude': {'longitude'}, 'isbn10': {'isbn10'}, 'ipv4_public': {'public', 'ipv4'}, 'license_plate': {'plate', 'license'}, 'postcode_in_state': {'postcode', 'in', 'state'}, 'ascii_company_email': {'ascii', 'company', 'email'}, 'ean13': {'ean13'}, 'set_formatter': {'formatter', 'set'}, 'pyint': {'pyint'}, 'state_abbr': {'abbr', 'state'}, 'date_this_year': {'this', 'year', 'date'}, 'postcode': {'postcode'}, 'prefix_female': {'female', 'prefix'}, 'unix_device': {'device', 'unix'}, 'date_between_dates': {'dates', 'date', 'between'}, 'firefox': {'firefox'}, 'json_bytes': {'json', 'bytes'}, 'year': {'year'}, 'slug': {'slug'}, 'zipcode': {'zipcode'}, 'tsv': {'tsv'}, 'tld': {'tld'}, 'linux_processor': {'processor', 'linux'}, 'nic_handles': {'handles', 'nic'}, 'latitude': {'latitude'}, 'random_element': {'random', 'element'}, 'building_number': {'number', 'building'}, 'credit_card_full': {'full', 'credit', 'card'}, 'passport_number': {'number', 'passport'}, 'bban': {'bban'}, 'localized_ean8': {'localized', 'ean8'}, 'provider': {'provider'}, 'items': {'items'}, 'add_provider': {'provider', 'add'}, 'free_email_domain': {'domain', 'free', 'email'}, 'image': {'image'}, 'random_digit_above_two': {'above', 'two', 'random', 'digit'}, 'image_url': {'url', 'image'}, 'country_calling_code': {'calling', 'code', 'country'}, 'ascii_email': {'ascii', 'email'}, 'prefix_male': {'prefix', 'male'}, 'ipv4': {'ipv4'}, 'job': {'job'}, 'pricetag': {'pricetag'}, 'prefix': {'prefix'}, 'credit_card_security_code': {'credit', 'code', 'security', 'card'}, 'zipcode_in_state': {'in', 'state', 'zipcode'}, 'iso8601': {'iso8601'}, 'safe_domain_name': {'safe', 'domain', 'name'}, 'iban': {'iban'}, 'randomize_nb_elements': {'randomize', 'elements', 'nb'}, 'bs': {'bs'}, 'json': {'json'}, 'job_female': {'female', 'job'}, 'military_apo': {'apo', 'military'}, 'random_sample': {'sample', 'random'}, 'color_hsv': {'color', 'hsv'}, 'country_code': {'code', 'country'}, 'currency': {'currency'}, 'sentence': {'sentence'}, 'bank_country': {'country', 'bank'}, 'texts': {'texts'}, 'paragraph': {'paragraph'}, 'unix_partition': {'partition', 'unix'}, 'country': {'country'}, 'random_int': {'random', 'int'}, 'csv': {'csv'}, 'free_email': {'free', 'email'}, 'set_arguments': {'set', 'arguments'}, 'random_digit_not_null_or_empty': {'or', 'empty', 'not', 'null', 'random', 'digit'}, 'vin': {'vin'}, 'postalcode_plus4': {'postalcode', 'plus4'}, 'boolean': {'boolean'}, 'url': {'url'}, 'mac_platform_token': {'token', 'platform', 'mac'}, 'mime_type': {'mime', 'type'}, 'random_elements': {'elements', 'random'}, 'random_letter': {'letter', 'random'}, 'random_letters': {'random', 'letters'}, 'itin': {'itin'}, 'lexify': {'lexify'}, 'last_name': {'name', 'last'}, 'nic_handle': {'handle', 'nic'}, 'sha256': {'sha256'}, 'uri': {'uri'}, 'phone_number': {'number', 'phone'}, 'company_suffix': {'company', 'suffix'}, 'bothify': {'bothify'}, 'del_arguments': {'del', 'arguments'}, 'parse': {'parse'}, 'xml': {'xml'}, 'invalid_ssn': {'invalid', 'ssn'}, 'last_name_nonbinary': {'name', 'nonbinary', 'last'}, 'mac_address': {'address', 'mac'}, 'mac_processor': {'processor', 'mac'}, 'pyset': {'pyset'}, 'user_name': {'name', 'user'}, 'enum': {'enum'}, 'hostname': {'hostname'}, 'ios_platform_token': {'token', 'platform', 'ios'}, 'name_female': {'name', 'female'}, 'pystr_format': {'pystr', 'format'}, 'postalcode': {'postalcode'}, 'pystr': {'pystr'}, 'simple_profile': {'simple', 'profile'}, 'unix_time': {'unix', 'time'}, 'suffix_male': {'male', 'suffix'}, 'random_lowercase_letter': {'lowercase', 'letter', 'random'}, 'language_name': {'name', 'language'}, 'administrative_unit': {'administrative', 'unit'}, 'ascii_free_email': {'ascii', 'free', 'email'}, 'email': {'email'}, 'state': {'state'}, 'date_object': {'object', 'date'}, 'rgb_css_color': {'rgb', 'css', 'color'}, 'ssn': {'ssn'}, 'upc_e': {'e', 'upc'}, 'linux_platform_token': {'token', 'platform', 'linux'}, 'profile': {'profile'}, 'swift11': {'swift11'}, 'pybool': {'pybool'}, 'color_name': {'name', 'color'}, 'date_this_century': {'this', 'century', 'date'}, 'isbn13': {'isbn13'}, 'random_number': {'random', 'number'}, 'date': {'date'}}
    },
    'mimesis' : {
        'word_list': ['month', 'default_country', 'username', 'political_views', 'formatted_time', 'color', 'university', 'stock_name', 'federal_subject', 'currency_symbol', 'street_number', 'weight', 'stock_exchange', 'timestamp', 'nationality', 'occupation', 'company_type', 'cryptocurrency_symbol', 'worldview', 'rgb_color', 'province', 'views_on', 'price', 'street_name', 'prefecture', 'duration', 'update_dataset', 'company', 'birthdate', 'word', 'iata_code', 'dish', 'gender_symbol', 'city', 'timezone', 'override_locale', 'stock_ticker', 'hex_color', 'datetime', 'full_name', 'sex', 'icao_code', 'name', 'validate_enum', 'surname', 'address', 'Meta', 'postal_code', 'gender', 'day_of_week', 'first_name', 'quote', 'words', 'bulk_create_datetimes', 'vegetable', 'language', 'text', 'emoji', 'street_suffix', 'day_of_month', 'calling_code', 'time', 'password', 'drink', 'century', 'longitude', 'blood_type', 'gender_code', 'currency_iso_code', 'country_emoji_flag', 'height', 'year', 'identifier', 'latitude', 'answer', 'coordinates', 'cryptocurrency_iso_code', 'academic_degree', 'reseed', 'isd_code', 'level', 'telephone', 'continent', 'country_code', 'price_in_btc', 'sentence', 'get_current_locale', 'formatted_date', 'country', 'zip_code', 'last_name', 'phone_number', 'title', 'region', 'spices', 'alphabet', 'state', 'email', 'periodicity', 'week_date', 'formatted_datetime', 'fruit', 'gmt_offset', 'date', 'bank'],
        'word_tokens': {'month': ['month'], 'default_country': ['default', 'country'], 'username': ['username'], 'political_views': ['political', 'views'], 'formatted_time': ['formatted', 'time'], 'color': ['color'], 'university': ['university'], 'stock_name': ['stock', 'name'], 'federal_subject': ['federal', 'subject'], 'currency_symbol': ['currency', 'symbol'], 'street_number': ['street', 'number'], 'weight': ['weight'], 'stock_exchange': ['stock', 'exchange'], 'timestamp': ['timestamp'], 'nationality': ['nationality'], 'occupation': ['occupation'], 'company_type': ['company', 'type'], 'cryptocurrency_symbol': ['cryptocurrency', 'symbol'], 'worldview': ['worldview'], 'rgb_color': ['rgb', 'color'], 'province': ['province'], 'views_on': ['views', 'on'], 'price': ['price'], 'street_name': ['street', 'name'], 'prefecture': ['prefecture'], 'duration': ['duration'], 'update_dataset': ['update', 'dataset'], 'company': ['company'], 'birthdate': ['birthdate'], 'word': ['word'], 'iata_code': ['iata', 'code'], 'dish': ['dish'], 'gender_symbol': ['gender', 'symbol'], 'city': ['city'], 'timezone': ['timezone'], 'override_locale': ['override', 'locale'], 'stock_ticker': ['stock', 'ticker'], 'hex_color': ['hex', 'color'], 'datetime': ['datetime'], 'full_name': ['full', 'name'], 'sex': ['sex'], 'icao_code': ['icao', 'code'], 'name': ['name'], 'validate_enum': ['validate', 'enum'], 'surname': ['surname'], 'address': ['address'], 'Meta': ['Meta'], 'postal_code': ['postal', 'code'], 'gender': ['gender'], 'day_of_week': ['day', 'of', 'week'], 'first_name': ['first', 'name'], 'quote': ['quote'], 'words': ['words'], 'bulk_create_datetimes': ['bulk', 'create', 'datetimes'], 'vegetable': ['vegetable'], 'language': ['language'], 'text': ['text'], 'emoji': ['emoji'], 'street_suffix': ['street', 'suffix'], 'day_of_month': ['day', 'of', 'month'], 'calling_code': ['calling', 'code'], 'time': ['time'], 'password': ['password'], 'drink': ['drink'], 'century': ['century'], 'longitude': ['longitude'], 'blood_type': ['blood', 'type'], 'gender_code': ['gender', 'code'], 'currency_iso_code': ['currency', 'iso', 'code'], 'country_emoji_flag': ['country', 'emoji', 'flag'], 'height': ['height'], 'year': ['year'], 'identifier': ['identifier'], 'latitude': ['latitude'], 'answer': ['answer'], 'coordinates': ['coordinates'], 'cryptocurrency_iso_code': ['cryptocurrency', 'iso', 'code'], 'academic_degree': ['academic', 'degree'], 'reseed': ['reseed'], 'isd_code': ['isd', 'code'], 'level': ['level'], 'telephone': ['telephone'], 'continent': ['continent'], 'country_code': ['country', 'code'], 'price_in_btc': ['price', 'in', 'btc'], 'sentence': ['sentence'], 'get_current_locale': ['get', 'current', 'locale'], 'formatted_date': ['formatted', 'date'], 'country': ['country'], 'zip_code': ['zip', 'code'], 'last_name': ['last', 'name'], 'phone_number': ['phone', 'number'], 'title': ['title'], 'region': ['region'], 'spices': ['spices'], 'alphabet': ['alphabet'], 'state': ['state'], 'email': ['email'], 'periodicity': ['periodicity'], 'week_date': ['week', 'date'], 'formatted_datetime': ['formatted', 'datetime'], 'fruit': ['fruit'], 'gmt_offset': ['gmt', 'offset'], 'date': ['date'], 'bank': ['bank']},
        'word_tokens_set': {'month': {'month'}, 'default_country': {'default', 'country'}, 'username': {'username'}, 'political_views': {'views', 'political'}, 'formatted_time': {'formatted', 'time'}, 'color': {'color'}, 'university': {'university'}, 'stock_name': {'stock', 'name'}, 'federal_subject': {'subject', 'federal'}, 'currency_symbol': {'currency', 'symbol'}, 'street_number': {'street', 'number'}, 'weight': {'weight'}, 'stock_exchange': {'stock', 'exchange'}, 'timestamp': {'timestamp'}, 'nationality': {'nationality'}, 'occupation': {'occupation'}, 'company_type': {'company', 'type'}, 'cryptocurrency_symbol': {'cryptocurrency', 'symbol'}, 'worldview': {'worldview'}, 'rgb_color': {'rgb', 'color'}, 'province': {'province'}, 'views_on': {'on', 'views'}, 'price': {'price'}, 'street_name': {'street', 'name'}, 'prefecture': {'prefecture'}, 'duration': {'duration'}, 'update_dataset': {'update', 'dataset'}, 'company': {'company'}, 'birthdate': {'birthdate'}, 'word': {'word'}, 'iata_code': {'code', 'iata'}, 'dish': {'dish'}, 'gender_symbol': {'gender', 'symbol'}, 'city': {'city'}, 'timezone': {'timezone'}, 'override_locale': {'override', 'locale'}, 'stock_ticker': {'stock', 'ticker'}, 'hex_color': {'hex', 'color'}, 'datetime': {'datetime'}, 'full_name': {'full', 'name'}, 'sex': {'sex'}, 'icao_code': {'code', 'icao'}, 'name': {'name'}, 'validate_enum': {'validate', 'enum'}, 'surname': {'surname'}, 'address': {'address'}, 'Meta': {'Meta'}, 'postal_code': {'code', 'postal'}, 'gender': {'gender'}, 'day_of_week': {'day', 'of', 'week'}, 'first_name': {'first', 'name'}, 'quote': {'quote'}, 'words': {'words'}, 'bulk_create_datetimes': {'datetimes', 'bulk', 'create'}, 'vegetable': {'vegetable'}, 'language': {'language'}, 'text': {'text'}, 'emoji': {'emoji'}, 'street_suffix': {'street', 'suffix'}, 'day_of_month': {'day', 'month', 'of'}, 'calling_code': {'calling', 'code'}, 'time': {'time'}, 'password': {'password'}, 'drink': {'drink'}, 'century': {'century'}, 'longitude': {'longitude'}, 'blood_type': {'blood', 'type'}, 'gender_code': {'gender', 'code'}, 'currency_iso_code': {'currency', 'code', 'iso'}, 'country_emoji_flag': {'flag', 'country', 'emoji'}, 'height': {'height'}, 'year': {'year'}, 'identifier': {'identifier'}, 'latitude': {'latitude'}, 'answer': {'answer'}, 'coordinates': {'coordinates'}, 'cryptocurrency_iso_code': {'cryptocurrency', 'code', 'iso'}, 'academic_degree': {'degree', 'academic'}, 'reseed': {'reseed'}, 'isd_code': {'isd', 'code'}, 'level': {'level'}, 'telephone': {'telephone'}, 'continent': {'continent'}, 'country_code': {'code', 'country'}, 'price_in_btc': {'in', 'price', 'btc'}, 'sentence': {'sentence'}, 'get_current_locale': {'current', 'locale', 'get'}, 'formatted_date': {'formatted', 'date'}, 'country': {'country'}, 'zip_code': {'zip', 'code'}, 'last_name': {'name', 'last'}, 'phone_number': {'number', 'phone'}, 'title': {'title'}, 'region': {'region'}, 'spices': {'spices'}, 'alphabet': {'alphabet'}, 'state': {'state'}, 'email': {'email'}, 'periodicity': {'periodicity'}, 'week_date': {'date', 'week'}, 'formatted_datetime': {'datetime', 'formatted'}, 'fruit': {'fruit'}, 'gmt_offset': {'offset', 'gmt'}, 'date': {'date'}, 'bank': {'bank'}}
    },
    'mixed' : {
        'word_list': ['city_prefix', 'month', 'time_object', 'random_digit_not_null', 'aba', 'cryptocurrency_code', 'date_time_this_century', 'locale', 'stock_name', 'msisdn', 'ipv4_private', 'iana_id', 'city_suffix', 'currency_symbol', 'dsv', 'street_number', 'military_dpo', 'suffix_female', 'ripe_id', 'weight', 'stock_exchange', 'past_date', 'credit_card_provider', 'credit_card_number', 'first_name_nonbinary', 'random_digit_or_empty', 'company_type', 'currency_name', 'format', 'safe_email', 'worldview', 'job_male', 'fixed_width', 'current_country_code', 'location_on_land', 'price', 'street_name', 'ean8', 'get_words_list', 'update_dataset', 'date_time_between', 'time_delta', 'stock_ticker', 'datetime', 'sex', 'pytimezone', 'name', 'gender', 'name_nonbinary', 'random_uppercase_letter', 'file_name', 'passport_gender', 'basic_phone_number', 'credit_card_expire', 'pyobject', 'vegetable', 'emoji', 'military_ship', 'street_suffix', 'color_hsl', 'swift8', 'longitude', 'ean13', 'pyint', 'state_abbr', 'date_this_year', 'country_emoji_flag', 'date_between_dates', 'year', 'zipcode', 'tld', 'building_number', 'add_provider', 'image_url', 'cryptocurrency_iso_code', 'job', 'pricetag', 'safe_domain_name', 'random_sample', 'military_apo', 'continent', 'country_code', 'texts', 'set_arguments', 'zip_code', 'random_letter', 'nic_handle', 'invalid_ssn', 'user_name', 'mac_processor', 'enum', 'postalcode', 'suffix_male', 'language_name', 'ascii_free_email', 'email', 'linux_platform_token', 'week_date', 'date_this_century', 'random_number', 'date', 'past_datetime', 'postalcode_in_state', 'username', 'safari', 'formatted_time', 'ascii_safe_email', 'date_time_this_decade', 'current_country', 'coordinate', 'sha1', 'university', 'federal_subject', 'domain_name', 'localized_ean13', 'date_between', 'zip', 'zipcode_plus4', 'color_rgb', 'street_address', 'nationality', 'uri_page', 'views_on', 'prefecture', 'duration', 'date_this_month', 'company', 'pyfloat', 'word', 'pystruct', 'gender_symbol', 'override_locale', 'secondary_address', 'passport_dates', 'swift', 'cryptocurrency_name', 'Meta', 'postal_code', 'date_time_this_month', 'seed_locale', 'binary', 'suffix', 'null_boolean', 'http_method', 'language', 'day_of_month', 'language_code', 'calling_code', 'password', 'safe_hex_color', 'internet_explorer', 'ipv4_public', 'gender_code', 'currency_iso_code', 'prefix_female', 'json_bytes', 'height', 'identifier', 'tsv', 'latitude', 'answer', 'provider', 'passport_number', 'items', 'ascii_email', 'credit_card_security_code', 'zipcode_in_state', 'vin', 'country', 'random_digit_not_null_or_empty', 'postalcode_plus4', 'mac_platform_token', 'mime_type', 'sha256', 'company_suffix', 'del_arguments', 'ios_platform_token', 'title', 'simple_profile', 'spices', 'random_lowercase_letter', 'profile', 'formatted_datetime', 'swift11', 'pybool', 'isbn13', 'bank', 'future_date', 'political_views', 'name_male', 'date_time_ad', 'pydict', 'color', 'uuid4', 'file_extension', 'latlng', 'get_formatter', 'paragraphs', 'ean', 'prefix_nonbinary', 'windows_platform_token', 'hexify', 'date_time', 'local_latlng', 'currency_code', 'time_series', 'future_datetime', 'localized_ean', 'first_name_female', 'cryptocurrency_symbol', 'doi', 'numerify', 'rgb_color', 'domain_word', 'pydecimal', 'am_pm', 'month_name', 'birthdate', 'first_name_male', 'iata_code', 'catch_phrase', 'sentences', 'opera', 'hex_color', 'uri_path', 'military_state', 'full_name', 'icao_code', 'last_name_male', 'date_of_birth', 'address', 'chrome', 'dga', 'get_providers', 'uri_extension', 'tar', 'first_name', 'http_status_code', 'bulk_create_datetimes', 'random_choices', 'century', 'isbn10', 'postcode_in_state', 'license_plate', 'ascii_company_email', 'postcode', 'unix_device', 'firefox', 'linux_processor', 'nic_handles', 'random_element', 'credit_card_full', 'coordinates', 'bban', 'localized_ean8', 'free_email_domain', 'isd_code', 'prefix', 'level', 'randomize_nb_elements', 'job_female', 'color_hsv', 'price_in_btc', 'currency', 'unix_partition', 'bank_country', 'get_current_locale', 'random_int', 'csv', 'boolean', 'url', 'random_elements', 'random_letters', 'itin', 'lexify', 'last_name', 'uri', 'phone_number', 'parse', 'xml', 'pyset', 'mac_address', 'region', 'pystr_format', 'unix_time', 'alphabet', 'administrative_unit', 'date_object', 'periodicity', 'fruit', 'gmt_offset', 'psv', 'suffix_nonbinary', 'default_country', 'pytuple', 'ipv6', 'last_name_female', 'date_time_between_dates', 'user_agent', 'sbn9', 'safe_color_name', 'ipv4_network_class', 'seed_instance', 'android_platform_token', 'timestamp', 'get_arguments', 'pylist', 'occupation', 'province', 'passport_full', 'random_digit', 'passport_owner', 'port_number', 'date_this_decade', 'passport_dob', 'dish', 'city', 'timezone', 'md5', 'validate_enum', 'surname', 'date_time_this_year', 'ein', 'day_of_week', 'quote', 'words', 'file_path', 'color_rgb_float', 'text', 'pyiterable', 'company_email', 'time', 'cryptocurrency', 'drink', 'upc_a', 'blood_type', 'set_formatter', 'slug', 'random_digit_above_two', 'image', 'academic_degree', 'country_calling_code', 'prefix_male', 'ipv4', 'reseed', 'iso8601', 'iban', 'bs', 'telephone', 'json', 'sentence', 'formatted_date', 'paragraph', 'free_email', 'bothify', 'last_name_nonbinary', 'hostname', 'name_female', 'pystr', 'state', 'ssn', 'rgb_css_color', 'upc_e', 'color_name'],
        'word_tokens': {'city_prefix': ['city', 'prefix'], 'month': ['month'], 'time_object': ['time', 'object'], 'random_digit_not_null': ['random', 'digit', 'not', 'null'], 'aba': ['aba'], 'cryptocurrency_code': ['cryptocurrency', 'code'], 'date_time_this_century': ['date', 'time', 'this', 'century'], 'locale': ['locale'], 'stock_name': ['stock', 'name'], 'msisdn': ['msisdn'], 'ipv4_private': ['ipv4', 'private'], 'iana_id': ['iana', 'id'], 'city_suffix': ['city', 'suffix'], 'currency_symbol': ['currency', 'symbol'], 'dsv': ['dsv'], 'street_number': ['street', 'number'], 'military_dpo': ['military', 'dpo'], 'suffix_female': ['suffix', 'female'], 'ripe_id': ['ripe', 'id'], 'weight': ['weight'], 'stock_exchange': ['stock', 'exchange'], 'past_date': ['past', 'date'], 'credit_card_provider': ['credit', 'card', 'provider'], 'credit_card_number': ['credit', 'card', 'number'], 'first_name_nonbinary': ['first', 'name', 'nonbinary'], 'random_digit_or_empty': ['random', 'digit', 'or', 'empty'], 'company_type': ['company', 'type'], 'currency_name': ['currency', 'name'], 'format': ['format'], 'safe_email': ['safe', 'email'], 'worldview': ['worldview'], 'job_male': ['job', 'male'], 'fixed_width': ['fixed', 'width'], 'current_country_code': ['current', 'country', 'code'], 'location_on_land': ['location', 'on', 'land'], 'price': ['price'], 'street_name': ['street', 'name'], 'ean8': ['ean8'], 'get_words_list': ['get', 'words', 'list'], 'update_dataset': ['update', 'dataset'], 'date_time_between': ['date', 'time', 'between'], 'time_delta': ['time', 'delta'], 'stock_ticker': ['stock', 'ticker'], 'datetime': ['datetime'], 'sex': ['sex'], 'pytimezone': ['pytimezone'], 'name': ['name'], 'gender': ['gender'], 'name_nonbinary': ['name', 'nonbinary'], 'random_uppercase_letter': ['random', 'uppercase', 'letter'], 'file_name': ['file', 'name'], 'passport_gender': ['passport', 'gender'], 'basic_phone_number': ['basic', 'phone', 'number'], 'credit_card_expire': ['credit', 'card', 'expire'], 'pyobject': ['pyobject'], 'vegetable': ['vegetable'], 'emoji': ['emoji'], 'military_ship': ['military', 'ship'], 'street_suffix': ['street', 'suffix'], 'color_hsl': ['color', 'hsl'], 'swift8': ['swift8'], 'longitude': ['longitude'], 'ean13': ['ean13'], 'pyint': ['pyint'], 'state_abbr': ['state', 'abbr'], 'date_this_year': ['date', 'this', 'year'], 'country_emoji_flag': ['country', 'emoji', 'flag'], 'date_between_dates': ['date', 'between', 'dates'], 'year': ['year'], 'zipcode': ['zipcode'], 'tld': ['tld'], 'building_number': ['building', 'number'], 'add_provider': ['add', 'provider'], 'image_url': ['image', 'url'], 'cryptocurrency_iso_code': ['cryptocurrency', 'iso', 'code'], 'job': ['job'], 'pricetag': ['pricetag'], 'safe_domain_name': ['safe', 'domain', 'name'], 'random_sample': ['random', 'sample'], 'military_apo': ['military', 'apo'], 'continent': ['continent'], 'country_code': ['country', 'code'], 'texts': ['texts'], 'set_arguments': ['set', 'arguments'], 'zip_code': ['zip', 'code'], 'random_letter': ['random', 'letter'], 'nic_handle': ['nic', 'handle'], 'invalid_ssn': ['invalid', 'ssn'], 'user_name': ['user', 'name'], 'mac_processor': ['mac', 'processor'], 'enum': ['enum'], 'postalcode': ['postalcode'], 'suffix_male': ['suffix', 'male'], 'language_name': ['language', 'name'], 'ascii_free_email': ['ascii', 'free', 'email'], 'email': ['email'], 'linux_platform_token': ['linux', 'platform', 'token'], 'week_date': ['week', 'date'], 'date_this_century': ['date', 'this', 'century'], 'random_number': ['random', 'number'], 'date': ['date'], 'past_datetime': ['past', 'datetime'], 'postalcode_in_state': ['postalcode', 'in', 'state'], 'username': ['username'], 'safari': ['safari'], 'formatted_time': ['formatted', 'time'], 'ascii_safe_email': ['ascii', 'safe', 'email'], 'date_time_this_decade': ['date', 'time', 'this', 'decade'], 'current_country': ['current', 'country'], 'coordinate': ['coordinate'], 'sha1': ['sha1'], 'university': ['university'], 'federal_subject': ['federal', 'subject'], 'domain_name': ['domain', 'name'], 'localized_ean13': ['localized', 'ean13'], 'date_between': ['date', 'between'], 'zip': ['zip'], 'zipcode_plus4': ['zipcode', 'plus4'], 'color_rgb': ['color', 'rgb'], 'street_address': ['street', 'address'], 'nationality': ['nationality'], 'uri_page': ['uri', 'page'], 'views_on': ['views', 'on'], 'prefecture': ['prefecture'], 'duration': ['duration'], 'date_this_month': ['date', 'this', 'month'], 'company': ['company'], 'pyfloat': ['pyfloat'], 'word': ['word'], 'pystruct': ['pystruct'], 'gender_symbol': ['gender', 'symbol'], 'override_locale': ['override', 'locale'], 'secondary_address': ['secondary', 'address'], 'passport_dates': ['passport', 'dates'], 'swift': ['swift'], 'cryptocurrency_name': ['cryptocurrency', 'name'], 'Meta': ['Meta'], 'postal_code': ['postal', 'code'], 'date_time_this_month': ['date', 'time', 'this', 'month'], 'seed_locale': ['seed', 'locale'], 'binary': ['binary'], 'suffix': ['suffix'], 'null_boolean': ['null', 'boolean'], 'http_method': ['http', 'method'], 'language': ['language'], 'day_of_month': ['day', 'of', 'month'], 'language_code': ['language', 'code'], 'calling_code': ['calling', 'code'], 'password': ['password'], 'safe_hex_color': ['safe', 'hex', 'color'], 'internet_explorer': ['internet', 'explorer'], 'ipv4_public': ['ipv4', 'public'], 'gender_code': ['gender', 'code'], 'currency_iso_code': ['currency', 'iso', 'code'], 'prefix_female': ['prefix', 'female'], 'json_bytes': ['json', 'bytes'], 'height': ['height'], 'identifier': ['identifier'], 'tsv': ['tsv'], 'latitude': ['latitude'], 'answer': ['answer'], 'provider': ['provider'], 'passport_number': ['passport', 'number'], 'items': ['items'], 'ascii_email': ['ascii', 'email'], 'credit_card_security_code': ['credit', 'card', 'security', 'code'], 'zipcode_in_state': ['zipcode', 'in', 'state'], 'vin': ['vin'], 'country': ['country'], 'random_digit_not_null_or_empty': ['random', 'digit', 'not', 'null', 'or', 'empty'], 'postalcode_plus4': ['postalcode', 'plus4'], 'mac_platform_token': ['mac', 'platform', 'token'], 'mime_type': ['mime', 'type'], 'sha256': ['sha256'], 'company_suffix': ['company', 'suffix'], 'del_arguments': ['del', 'arguments'], 'ios_platform_token': ['ios', 'platform', 'token'], 'title': ['title'], 'simple_profile': ['simple', 'profile'], 'spices': ['spices'], 'random_lowercase_letter': ['random', 'lowercase', 'letter'], 'profile': ['profile'], 'formatted_datetime': ['formatted', 'datetime'], 'swift11': ['swift11'], 'pybool': ['pybool'], 'isbn13': ['isbn13'], 'bank': ['bank'], 'future_date': ['future', 'date'], 'political_views': ['political', 'views'], 'name_male': ['name', 'male'], 'date_time_ad': ['date', 'time', 'ad'], 'pydict': ['pydict'], 'color': ['color'], 'uuid4': ['uuid4'], 'file_extension': ['file', 'extension'], 'latlng': ['latlng'], 'get_formatter': ['get', 'formatter'], 'paragraphs': ['paragraphs'], 'ean': ['ean'], 'prefix_nonbinary': ['prefix', 'nonbinary'], 'windows_platform_token': ['windows', 'platform', 'token'], 'hexify': ['hexify'], 'date_time': ['date', 'time'], 'local_latlng': ['local', 'latlng'], 'currency_code': ['currency', 'code'], 'time_series': ['time', 'series'], 'future_datetime': ['future', 'datetime'], 'localized_ean': ['localized', 'ean'], 'first_name_female': ['first', 'name', 'female'], 'cryptocurrency_symbol': ['cryptocurrency', 'symbol'], 'doi': ['doi'], 'numerify': ['numerify'], 'rgb_color': ['rgb', 'color'], 'domain_word': ['domain', 'word'], 'pydecimal': ['pydecimal'], 'am_pm': ['am', 'pm'], 'month_name': ['month', 'name'], 'birthdate': ['birthdate'], 'first_name_male': ['first', 'name', 'male'], 'iata_code': ['iata', 'code'], 'catch_phrase': ['catch', 'phrase'], 'sentences': ['sentences'], 'opera': ['opera'], 'hex_color': ['hex', 'color'], 'uri_path': ['uri', 'path'], 'military_state': ['military', 'state'], 'full_name': ['full', 'name'], 'icao_code': ['icao', 'code'], 'last_name_male': ['last', 'name', 'male'], 'date_of_birth': ['date', 'of', 'birth'], 'address': ['address'], 'chrome': ['chrome'], 'dga': ['dga'], 'get_providers': ['get', 'providers'], 'uri_extension': ['uri', 'extension'], 'tar': ['tar'], 'first_name': ['first', 'name'], 'http_status_code': ['http', 'status', 'code'], 'bulk_create_datetimes': ['bulk', 'create', 'datetimes'], 'random_choices': ['random', 'choices'], 'century': ['century'], 'isbn10': ['isbn10'], 'postcode_in_state': ['postcode', 'in', 'state'], 'license_plate': ['license', 'plate'], 'ascii_company_email': ['ascii', 'company', 'email'], 'postcode': ['postcode'], 'unix_device': ['unix', 'device'], 'firefox': ['firefox'], 'linux_processor': ['linux', 'processor'], 'nic_handles': ['nic', 'handles'], 'random_element': ['random', 'element'], 'credit_card_full': ['credit', 'card', 'full'], 'coordinates': ['coordinates'], 'bban': ['bban'], 'localized_ean8': ['localized', 'ean8'], 'free_email_domain': ['free', 'email', 'domain'], 'isd_code': ['isd', 'code'], 'prefix': ['prefix'], 'level': ['level'], 'randomize_nb_elements': ['randomize', 'nb', 'elements'], 'job_female': ['job', 'female'], 'color_hsv': ['color', 'hsv'], 'price_in_btc': ['price', 'in', 'btc'], 'currency': ['currency'], 'unix_partition': ['unix', 'partition'], 'bank_country': ['bank', 'country'], 'get_current_locale': ['get', 'current', 'locale'], 'random_int': ['random', 'int'], 'csv': ['csv'], 'boolean': ['boolean'], 'url': ['url'], 'random_elements': ['random', 'elements'], 'random_letters': ['random', 'letters'], 'itin': ['itin'], 'lexify': ['lexify'], 'last_name': ['last', 'name'], 'uri': ['uri'], 'phone_number': ['phone', 'number'], 'parse': ['parse'], 'xml': ['xml'], 'pyset': ['pyset'], 'mac_address': ['mac', 'address'], 'region': ['region'], 'pystr_format': ['pystr', 'format'], 'unix_time': ['unix', 'time'], 'alphabet': ['alphabet'], 'administrative_unit': ['administrative', 'unit'], 'date_object': ['date', 'object'], 'periodicity': ['periodicity'], 'fruit': ['fruit'], 'gmt_offset': ['gmt', 'offset'], 'psv': ['psv'], 'suffix_nonbinary': ['suffix', 'nonbinary'], 'default_country': ['default', 'country'], 'pytuple': ['pytuple'], 'ipv6': ['ipv6'], 'last_name_female': ['last', 'name', 'female'], 'date_time_between_dates': ['date', 'time', 'between', 'dates'], 'user_agent': ['user', 'agent'], 'sbn9': ['sbn9'], 'safe_color_name': ['safe', 'color', 'name'], 'ipv4_network_class': ['ipv4', 'network', 'class'], 'seed_instance': ['seed', 'instance'], 'android_platform_token': ['android', 'platform', 'token'], 'timestamp': ['timestamp'], 'get_arguments': ['get', 'arguments'], 'pylist': ['pylist'], 'occupation': ['occupation'], 'province': ['province'], 'passport_full': ['passport', 'full'], 'random_digit': ['random', 'digit'], 'passport_owner': ['passport', 'owner'], 'port_number': ['port', 'number'], 'date_this_decade': ['date', 'this', 'decade'], 'passport_dob': ['passport', 'dob'], 'dish': ['dish'], 'city': ['city'], 'timezone': ['timezone'], 'md5': ['md5'], 'validate_enum': ['validate', 'enum'], 'surname': ['surname'], 'date_time_this_year': ['date', 'time', 'this', 'year'], 'ein': ['ein'], 'day_of_week': ['day', 'of', 'week'], 'quote': ['quote'], 'words': ['words'], 'file_path': ['file', 'path'], 'color_rgb_float': ['color', 'rgb', 'float'], 'text': ['text'], 'pyiterable': ['pyiterable'], 'company_email': ['company', 'email'], 'time': ['time'], 'cryptocurrency': ['cryptocurrency'], 'drink': ['drink'], 'upc_a': ['upc', 'a'], 'blood_type': ['blood', 'type'], 'set_formatter': ['set', 'formatter'], 'slug': ['slug'], 'random_digit_above_two': ['random', 'digit', 'above', 'two'], 'image': ['image'], 'academic_degree': ['academic', 'degree'], 'country_calling_code': ['country', 'calling', 'code'], 'prefix_male': ['prefix', 'male'], 'ipv4': ['ipv4'], 'reseed': ['reseed'], 'iso8601': ['iso8601'], 'iban': ['iban'], 'bs': ['bs'], 'telephone': ['telephone'], 'json': ['json'], 'sentence': ['sentence'], 'formatted_date': ['formatted', 'date'], 'paragraph': ['paragraph'], 'free_email': ['free', 'email'], 'bothify': ['bothify'], 'last_name_nonbinary': ['last', 'name', 'nonbinary'], 'hostname': ['hostname'], 'name_female': ['name', 'female'], 'pystr': ['pystr'], 'state': ['state'], 'ssn': ['ssn'], 'rgb_css_color': ['rgb', 'css', 'color'], 'upc_e': ['upc', 'e'], 'color_name': ['color', 'name']},
        'word_tokens_set': {'city_prefix': {'prefix', 'city'}, 'month': {'month'}, 'time_object': {'object', 'time'}, 'random_digit_not_null': {'null', 'not', 'random', 'digit'}, 'aba': {'aba'}, 'cryptocurrency_code': {'cryptocurrency', 'code'}, 'date_time_this_century': {'this', 'century', 'date', 'time'}, 'locale': {'locale'}, 'stock_name': {'stock', 'name'}, 'msisdn': {'msisdn'}, 'ipv4_private': {'private', 'ipv4'}, 'iana_id': {'id', 'iana'}, 'city_suffix': {'city', 'suffix'}, 'currency_symbol': {'currency', 'symbol'}, 'dsv': {'dsv'}, 'street_number': {'street', 'number'}, 'military_dpo': {'dpo', 'military'}, 'suffix_female': {'female', 'suffix'}, 'ripe_id': {'id', 'ripe'}, 'weight': {'weight'}, 'stock_exchange': {'stock', 'exchange'}, 'past_date': {'past', 'date'}, 'credit_card_provider': {'provider', 'credit', 'card'}, 'credit_card_number': {'credit', 'number', 'card'}, 'first_name_nonbinary': {'first', 'name', 'nonbinary'}, 'random_digit_or_empty': {'empty', 'or', 'random', 'digit'}, 'company_type': {'company', 'type'}, 'currency_name': {'currency', 'name'}, 'format': {'format'}, 'safe_email': {'safe', 'email'}, 'worldview': {'worldview'}, 'job_male': {'job', 'male'}, 'fixed_width': {'fixed', 'width'}, 'current_country_code': {'current', 'code', 'country'}, 'location_on_land': {'land', 'on', 'location'}, 'price': {'price'}, 'street_name': {'street', 'name'}, 'ean8': {'ean8'}, 'get_words_list': {'get', 'words', 'list'}, 'update_dataset': {'update', 'dataset'}, 'date_time_between': {'date', 'between', 'time'}, 'time_delta': {'delta', 'time'}, 'stock_ticker': {'stock', 'ticker'}, 'datetime': {'datetime'}, 'sex': {'sex'}, 'pytimezone': {'pytimezone'}, 'name': {'name'}, 'gender': {'gender'}, 'name_nonbinary': {'name', 'nonbinary'}, 'random_uppercase_letter': {'letter', 'random', 'uppercase'}, 'file_name': {'name', 'file'}, 'passport_gender': {'gender', 'passport'}, 'basic_phone_number': {'basic', 'number', 'phone'}, 'credit_card_expire': {'credit', 'expire', 'card'}, 'pyobject': {'pyobject'}, 'vegetable': {'vegetable'}, 'emoji': {'emoji'}, 'military_ship': {'ship', 'military'}, 'street_suffix': {'street', 'suffix'}, 'color_hsl': {'color', 'hsl'}, 'swift8': {'swift8'}, 'longitude': {'longitude'}, 'ean13': {'ean13'}, 'pyint': {'pyint'}, 'state_abbr': {'abbr', 'state'}, 'date_this_year': {'this', 'year', 'date'}, 'country_emoji_flag': {'flag', 'country', 'emoji'}, 'date_between_dates': {'dates', 'date', 'between'}, 'year': {'year'}, 'zipcode': {'zipcode'}, 'tld': {'tld'}, 'building_number': {'number', 'building'}, 'add_provider': {'provider', 'add'}, 'image_url': {'url', 'image'}, 'cryptocurrency_iso_code': {'cryptocurrency', 'code', 'iso'}, 'job': {'job'}, 'pricetag': {'pricetag'}, 'safe_domain_name': {'safe', 'domain', 'name'}, 'random_sample': {'sample', 'random'}, 'military_apo': {'apo', 'military'}, 'continent': {'continent'}, 'country_code': {'code', 'country'}, 'texts': {'texts'}, 'set_arguments': {'set', 'arguments'}, 'zip_code': {'zip', 'code'}, 'random_letter': {'letter', 'random'}, 'nic_handle': {'handle', 'nic'}, 'invalid_ssn': {'invalid', 'ssn'}, 'user_name': {'name', 'user'}, 'mac_processor': {'processor', 'mac'}, 'enum': {'enum'}, 'postalcode': {'postalcode'}, 'suffix_male': {'male', 'suffix'}, 'language_name': {'name', 'language'}, 'ascii_free_email': {'ascii', 'free', 'email'}, 'email': {'email'}, 'linux_platform_token': {'token', 'platform', 'linux'}, 'week_date': {'date', 'week'}, 'date_this_century': {'this', 'century', 'date'}, 'random_number': {'random', 'number'}, 'date': {'date'}, 'past_datetime': {'datetime', 'past'}, 'postalcode_in_state': {'postalcode', 'in', 'state'}, 'username': {'username'}, 'safari': {'safari'}, 'formatted_time': {'formatted', 'time'}, 'ascii_safe_email': {'ascii', 'safe', 'email'}, 'date_time_this_decade': {'this', 'decade', 'date', 'time'}, 'current_country': {'current', 'country'}, 'coordinate': {'coordinate'}, 'sha1': {'sha1'}, 'university': {'university'}, 'federal_subject': {'subject', 'federal'}, 'domain_name': {'domain', 'name'}, 'localized_ean13': {'localized', 'ean13'}, 'date_between': {'date', 'between'}, 'zip': {'zip'}, 'zipcode_plus4': {'plus4', 'zipcode'}, 'color_rgb': {'rgb', 'color'}, 'street_address': {'street', 'address'}, 'nationality': {'nationality'}, 'uri_page': {'page', 'uri'}, 'views_on': {'on', 'views'}, 'prefecture': {'prefecture'}, 'duration': {'duration'}, 'date_this_month': {'this', 'month', 'date'}, 'company': {'company'}, 'pyfloat': {'pyfloat'}, 'word': {'word'}, 'pystruct': {'pystruct'}, 'gender_symbol': {'gender', 'symbol'}, 'override_locale': {'override', 'locale'}, 'secondary_address': {'secondary', 'address'}, 'passport_dates': {'dates', 'passport'}, 'swift': {'swift'}, 'cryptocurrency_name': {'cryptocurrency', 'name'}, 'Meta': {'Meta'}, 'postal_code': {'code', 'postal'}, 'date_time_this_month': {'this', 'month', 'date', 'time'}, 'seed_locale': {'locale', 'seed'}, 'binary': {'binary'}, 'suffix': {'suffix'}, 'null_boolean': {'null', 'boolean'}, 'http_method': {'method', 'http'}, 'language': {'language'}, 'day_of_month': {'day', 'month', 'of'}, 'language_code': {'language', 'code'}, 'calling_code': {'calling', 'code'}, 'password': {'password'}, 'safe_hex_color': {'safe', 'color', 'hex'}, 'internet_explorer': {'internet', 'explorer'}, 'ipv4_public': {'public', 'ipv4'}, 'gender_code': {'gender', 'code'}, 'currency_iso_code': {'currency', 'code', 'iso'}, 'prefix_female': {'female', 'prefix'}, 'json_bytes': {'json', 'bytes'}, 'height': {'height'}, 'identifier': {'identifier'}, 'tsv': {'tsv'}, 'latitude': {'latitude'}, 'answer': {'answer'}, 'provider': {'provider'}, 'passport_number': {'number', 'passport'}, 'items': {'items'}, 'ascii_email': {'ascii', 'email'}, 'credit_card_security_code': {'credit', 'code', 'security', 'card'}, 'zipcode_in_state': {'in', 'state', 'zipcode'}, 'vin': {'vin'}, 'country': {'country'}, 'random_digit_not_null_or_empty': {'or', 'empty', 'not', 'null', 'random', 'digit'}, 'postalcode_plus4': {'postalcode', 'plus4'}, 'mac_platform_token': {'token', 'platform', 'mac'}, 'mime_type': {'mime', 'type'}, 'sha256': {'sha256'}, 'company_suffix': {'company', 'suffix'}, 'del_arguments': {'del', 'arguments'}, 'ios_platform_token': {'token', 'platform', 'ios'}, 'title': {'title'}, 'simple_profile': {'simple', 'profile'}, 'spices': {'spices'}, 'random_lowercase_letter': {'lowercase', 'letter', 'random'}, 'profile': {'profile'}, 'formatted_datetime': {'datetime', 'formatted'}, 'swift11': {'swift11'}, 'pybool': {'pybool'}, 'isbn13': {'isbn13'}, 'bank': {'bank'}, 'future_date': {'future', 'date'}, 'political_views': {'views', 'political'}, 'name_male': {'name', 'male'}, 'date_time_ad': {'ad', 'date', 'time'}, 'pydict': {'pydict'}, 'color': {'color'}, 'uuid4': {'uuid4'}, 'file_extension': {'extension', 'file'}, 'latlng': {'latlng'}, 'get_formatter': {'get', 'formatter'}, 'paragraphs': {'paragraphs'}, 'ean': {'ean'}, 'prefix_nonbinary': {'nonbinary', 'prefix'}, 'windows_platform_token': {'token', 'platform', 'windows'}, 'hexify': {'hexify'}, 'date_time': {'date', 'time'}, 'local_latlng': {'local', 'latlng'}, 'currency_code': {'currency', 'code'}, 'time_series': {'series', 'time'}, 'future_datetime': {'datetime', 'future'}, 'localized_ean': {'ean', 'localized'}, 'first_name_female': {'first', 'name', 'female'}, 'cryptocurrency_symbol': {'cryptocurrency', 'symbol'}, 'doi': {'doi'}, 'numerify': {'numerify'}, 'rgb_color': {'rgb', 'color'}, 'domain_word': {'domain', 'word'}, 'pydecimal': {'pydecimal'}, 'am_pm': {'am', 'pm'}, 'month_name': {'month', 'name'}, 'birthdate': {'birthdate'}, 'first_name_male': {'first', 'name', 'male'}, 'iata_code': {'code', 'iata'}, 'catch_phrase': {'catch', 'phrase'}, 'sentences': {'sentences'}, 'opera': {'opera'}, 'hex_color': {'hex', 'color'}, 'uri_path': {'uri', 'path'}, 'military_state': {'state', 'military'}, 'full_name': {'full', 'name'}, 'icao_code': {'code', 'icao'}, 'last_name_male': {'name', 'last', 'male'}, 'date_of_birth': {'birth', 'of', 'date'}, 'address': {'address'}, 'chrome': {'chrome'}, 'dga': {'dga'}, 'get_providers': {'get', 'providers'}, 'uri_extension': {'extension', 'uri'}, 'tar': {'tar'}, 'first_name': {'first', 'name'}, 'http_status_code': {'code', 'status', 'http'}, 'bulk_create_datetimes': {'datetimes', 'bulk', 'create'}, 'random_choices': {'random', 'choices'}, 'century': {'century'}, 'isbn10': {'isbn10'}, 'postcode_in_state': {'postcode', 'in', 'state'}, 'license_plate': {'plate', 'license'}, 'ascii_company_email': {'ascii', 'company', 'email'}, 'postcode': {'postcode'}, 'unix_device': {'device', 'unix'}, 'firefox': {'firefox'}, 'linux_processor': {'processor', 'linux'}, 'nic_handles': {'handles', 'nic'}, 'random_element': {'random', 'element'}, 'credit_card_full': {'full', 'credit', 'card'}, 'coordinates': {'coordinates'}, 'bban': {'bban'}, 'localized_ean8': {'localized', 'ean8'}, 'free_email_domain': {'domain', 'free', 'email'}, 'isd_code': {'isd', 'code'}, 'prefix': {'prefix'}, 'level': {'level'}, 'randomize_nb_elements': {'randomize', 'elements', 'nb'}, 'job_female': {'female', 'job'}, 'color_hsv': {'color', 'hsv'}, 'price_in_btc': {'in', 'price', 'btc'}, 'currency': {'currency'}, 'unix_partition': {'partition', 'unix'}, 'bank_country': {'country', 'bank'}, 'get_current_locale': {'current', 'locale', 'get'}, 'random_int': {'random', 'int'}, 'csv': {'csv'}, 'boolean': {'boolean'}, 'url': {'url'}, 'random_elements': {'elements', 'random'}, 'random_letters': {'random', 'letters'}, 'itin': {'itin'}, 'lexify': {'lexify'}, 'last_name': {'name', 'last'}, 'uri': {'uri'}, 'phone_number': {'number', 'phone'}, 'parse': {'parse'}, 'xml': {'xml'}, 'pyset': {'pyset'}, 'mac_address': {'address', 'mac'}, 'region': {'region'}, 'pystr_format': {'pystr', 'format'}, 'unix_time': {'unix', 'time'}, 'alphabet': {'alphabet'}, 'administrative_unit': {'administrative', 'unit'}, 'date_object': {'object', 'date'}, 'periodicity': {'periodicity'}, 'fruit': {'fruit'}, 'gmt_offset': {'offset', 'gmt'}, 'psv': {'psv'}, 'suffix_nonbinary': {'nonbinary', 'suffix'}, 'default_country': {'default', 'country'}, 'pytuple': {'pytuple'}, 'ipv6': {'ipv6'}, 'last_name_female': {'name', 'female', 'last'}, 'date_time_between_dates': {'dates', 'date', 'between', 'time'}, 'user_agent': {'agent', 'user'}, 'sbn9': {'sbn9'}, 'safe_color_name': {'safe', 'color', 'name'}, 'ipv4_network_class': {'class', 'network', 'ipv4'}, 'seed_instance': {'instance', 'seed'}, 'android_platform_token': {'token', 'platform', 'android'}, 'timestamp': {'timestamp'}, 'get_arguments': {'get', 'arguments'}, 'pylist': {'pylist'}, 'occupation': {'occupation'}, 'province': {'province'}, 'passport_full': {'full', 'passport'}, 'random_digit': {'random', 'digit'}, 'passport_owner': {'owner', 'passport'}, 'port_number': {'port', 'number'}, 'date_this_decade': {'this', 'date', 'decade'}, 'passport_dob': {'dob', 'passport'}, 'dish': {'dish'}, 'city': {'city'}, 'timezone': {'timezone'}, 'md5': {'md5'}, 'validate_enum': {'validate', 'enum'}, 'surname': {'surname'}, 'date_time_this_year': {'this', 'year', 'date', 'time'}, 'ein': {'ein'}, 'day_of_week': {'day', 'of', 'week'}, 'quote': {'quote'}, 'words': {'words'}, 'file_path': {'file', 'path'}, 'color_rgb_float': {'rgb', 'color', 'float'}, 'text': {'text'}, 'pyiterable': {'pyiterable'}, 'company_email': {'company', 'email'}, 'time': {'time'}, 'cryptocurrency': {'cryptocurrency'}, 'drink': {'drink'}, 'upc_a': {'a', 'upc'}, 'blood_type': {'blood', 'type'}, 'set_formatter': {'formatter', 'set'}, 'slug': {'slug'}, 'random_digit_above_two': {'above', 'two', 'random', 'digit'}, 'image': {'image'}, 'academic_degree': {'degree', 'academic'}, 'country_calling_code': {'calling', 'code', 'country'}, 'prefix_male': {'prefix', 'male'}, 'ipv4': {'ipv4'}, 'reseed': {'reseed'}, 'iso8601': {'iso8601'}, 'iban': {'iban'}, 'bs': {'bs'}, 'telephone': {'telephone'}, 'json': {'json'}, 'sentence': {'sentence'}, 'formatted_date': {'formatted', 'date'}, 'paragraph': {'paragraph'}, 'free_email': {'free', 'email'}, 'bothify': {'bothify'}, 'last_name_nonbinary': {'name', 'nonbinary', 'last'}, 'hostname': {'hostname'}, 'name_female': {'name', 'female'}, 'pystr': {'pystr'}, 'state': {'state'}, 'ssn': {'ssn'}, 'rgb_css_color': {'rgb', 'css', 'color'}, 'upc_e': {'e', 'upc'}, 'color_name': {'name', 'color'}}
    },
}

python_builtin_types = {
    str,
    int,
    float,
    Decimal,
    bool,
    complex,
    bytes,
    tuple,
    list,
    set,
    frozenset,
    dict,
}
typing_origins = {List, Dict, Tuple, Set, Union, Literal, Optional}
recursive_types = {
    List,
    Dict,
    Tuple,
    Set,
    Union,
    Literal,
    Optional,
    list,
    dict,
    tuple,
    set,
    frozenset,
}
all_constr_attribs = {
    "default",
    "annotation",
    "min_length",
    "max_length",
    "pattern",
    "gt",
    "lt",
    "ge",
    "le",
    "multiple_of"
}

default_constr_dict = {
    "required": True,
    "default": None,
    "annotation": None,
    "min_length": None,
    "max_length": None,
    "pattern": None,
    "gt": None,
    "lt": None,
    "ge": None,
    "le": None,
    "multiple_of": None,
    "origin": None,
    "args": None,
}
        