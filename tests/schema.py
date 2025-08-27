from pydantic import BaseModel


class USER(BaseModel):
    name: str


#'''
class Address(BaseModel):
    user: USER
    street: str


#'''

"""
class Address(BaseModel):
    username: str = Field(min_length=3, max_length=20, pattern="^[a-zA-Z0-9_]+$")
    email: Optional[str]
    age: int = Field(gt=12, lt=100)
    bio: Optional[str] = Field(default=None, max_length=250)
    interests: List[str] = Field(default_factory=list)
#"""

"""
class Address(BaseModel):
	field_decimal: decimal.Decimal = Field(gt=3, lt=15, multiple_of=0.000005, decimal_places = 2)
	field_float: float = Field(gt=15, allow_inf_nan=True)
#"""

"""
class Address(BaseModel):
	street: str
	city: str
	zip_code: List[Dict[constr(pattern=r'^\d{3}(-\d{6})?$'),List[constr(pattern=r'^\d{5}(-\d{4})?$')]]]  #either ddddd or ddddd-dddd
	country: str = Field(default="USA")
	social_security_number: str
	continent: Optional[str]
	phone_number: List[Dict[str,List[str]]]
	title: Set[str]
	direction: Literal["north", "south", "east", "west"]
	bananas: str
	#user: USER
	name: Annotated[Dict[constr(max_length = 1000), constr(max_length = 2000)],Field(min_length=30)]
#"""

"""
class Address(BaseModel):
	street: str
	city: str
	zip_code: List[Dict[constr(pattern=r'^\d{3}(-\d{6})?$'),List[constr(pattern=r'^\d{5}(-\d{4})?$')]]]  #either ddddd or ddddd-dddd
	country: str = Field(default="USA")
	social_security_number: str
	continent: Optional[str]
	phone_number: List[Dict[str,List[str]]]
	title: Set[str]
	direction: Literal["north", "south", "east", "west"]
	bananas: str
	#user: USER
	name: Annotated[Dict[constr(max_length = 1000), constr(max_length = 2000)],Field(min_length=30)]
	field_decimal_constr: decimal.Decimal = Field(gt=3, lt=15, multiple_of=0.000005, decimal_places = 2)
	field_float_constr: float = Field(gt=15, allow_inf_nan=True)
	field_str: str
	field_int: int
	field_float: float
	field_bool: bool
	field_complex: complex
	field_bytes: bytes
	field_tuple: tuple
	field_list: list #List[constr(min_length = 20)] = Field(min_length = 50)
	field_set: set
	field_frozenset: frozenset
	field_dict: dict
	username: str = Field(min_length=3, max_length=20, pattern="^[a-zA-Z0-9_]+$")
	email: Optional[str]
	age: int = Field(gt=12, lt=100)
	bio: Optional[str] = Field(default=None, max_length=250)
	interests: List[str] = Field(default_factory=list)
#"""


"""
	Meta: str
	aba: str
	academic_degree: str
	address: str
	administrative_unit: str
	alphabet: str
	am_pm: str
	android_platform_token: str
	answer: str
	ascii_company_email: str
	ascii_email: str
	ascii_free_email: str
	ascii_safe_email: str
	bank: str
	bank_country: str
	basic_phone_number: str
	bban: str
	binary: str
	birthdate: str
	blood_type: str
	boolean: str
	bothify: str
	bs: str
	building_number: str
	calling_code: str
	catch_phrase: str
	century: str
	chrome: str
	city: str
	city_prefix: str
	city_suffix: str
	color: str
	color_hsl: str
	color_hsv: str
	color_name: str
	color_rgb: str
	color_rgb_float: str
	company: str
	company_email: str
	company_suffix: str
	company_type: str
	continent: str
	coordinate: str
	coordinates: str
	country: str
	country_calling_code: str
	country_code: str
	country_emoji_flag: str
	credit_card_expire: str
	credit_card_full: str
	credit_card_number: str
	credit_card_provider: str
	credit_card_security_code: str
	cryptocurrency: str
	cryptocurrency_code: str
	cryptocurrency_iso_code: str
	cryptocurrency_name: str
	cryptocurrency_symbol: str
	csv: str
	currency: str
	currency_code: str
	currency_iso_code: str
	currency_name: str
	currency_symbol: str
	current_country: str
	current_country_code: str
	date: str
	date_between: str
	date_between_dates: str
	date_object: str
	date_of_birth: str
	date_this_century: str
	date_this_decade: str
	date_this_month: str
	date_this_year: str
	date_time: str
	date_time_ad: str
	date_time_between: str
	date_time_between_dates: str
	date_time_this_century: str
	date_time_this_decade: str
	date_time_this_month: str
	date_time_this_year: str
	datetime: str
	day_of_month: str
	day_of_week: str
	default_country: str
	dga: str
	dish: str
	doi: str
	domain_name: str
	domain_word: str
	drink: str
	dsv: str
	duration: str
	ean: str
	ean13: str
	ean8: str
	ein: str
	email: str
	emoji: str
	federal_subject: str
	file_extension: str
	file_name: str
	file_path: str
	firefox: str
	first_name: str
	first_name_female: str
	first_name_male: str
	first_name_nonbinary: str
	fixed_width: str
	formatted_date: str
	formatted_datetime: str
	formatted_time: str
	free_email: str
	free_email_domain: str
	fruit: str
	full_name: str
	future_date: str
	future_datetime: str
	gender: str
	gender_code: str
	gender_symbol: str
	get_current_locale: str
	get_providers: str
	get_words_list: str
	gmt_offset: str
	height: str
	hex_color: str
	hexify: str
	hostname: str
	http_method: str
	http_status_code: str
	iana_id: str
	iata_code: str
	iban: str
	icao_code: str
	identifier: str
	image: str
	image_url: str
	internet_explorer: str
	invalid_ssn: str
	ios_platform_token: str
	ipv4: str
	ipv4_network_class: str
	ipv4_private: str
	ipv4_public: str
	ipv6: str
	isbn10: str
	isbn13: str
	isd_code: str
	iso8601: str
	items: str
	itin: str
	job: str
	job_female: str
	job_male: str
	json: str
	json_bytes: str
	language: str
	language_code: str
	language_name: str
	last_name: str
	last_name_female: str
	last_name_male: str
	last_name_nonbinary: str
	latitude: str
	latlng: str
	level: str
	lexify: str
	license_plate: str
	linux_platform_token: str
	linux_processor: str
	local_latlng: str
	locale: str
	localized_ean: str
	localized_ean13: str
	localized_ean8: str
	location_on_land: str
	longitude: str
	mac_address: str
	mac_platform_token: str
	mac_processor: str
	md5: str
	military_apo: str
	military_dpo: str
	military_ship: str
	military_state: str
	mime_type: str
	month: str
	month_name: str
	msisdn: str
	name: str
	name_female: str
	name_male: str
	name_nonbinary: str
	nationality: str
	nic_handle: str
	nic_handles: str
	null_boolean: str
	numerify: str
	occupation: str
	opera: str
	paragraph: str
	paragraphs: str
	passport_dates: str
	passport_dob: str
	passport_full: str
	passport_gender: str
	passport_number: str
	passport_owner: str
	password: str
	past_date: str
	past_datetime: str
	periodicity: str
	phone_number: str
	political_views: str
	port_number: str
	postal_code: str
	postalcode: str
	postalcode_in_state: str
	postalcode_plus4: str
	postcode: str
	postcode_in_state: str
	prefecture: str
	prefix: str
	prefix_female: str
	prefix_male: str
	prefix_nonbinary: str
	price: str
	price_in_btc: str
	pricetag: str
	profile: str
	province: str
	psv: str
	pybool: str
	pydecimal: str
	pydict: str
	pyfloat: str
	pyint: str
	pyiterable: str
	pylist: str
	pyobject: str
	pyset: str
	pystr: str
	pystr_format: str
	pystruct: str
	pytimezone: str
	pytuple: str
	quote: str
	random_choices: str
	random_digit: str
	random_digit_above_two: str
	random_digit_not_null: str
	random_digit_not_null_or_empty: str
	random_digit_or_empty: str
	random_element: str
	random_elements: str
	random_int: str
	random_letter: str
	random_letters: str
	random_lowercase_letter: str
	random_number: str
	random_sample: str
	random_uppercase_letter: str
	randomize_nb_elements: str
	region: str
	reseed: str
	rgb_color: str
	rgb_css_color: str
	ripe_id: str
	safari: str
	safe_color_name: str
	safe_domain_name: str
	safe_email: str
	safe_hex_color: str
	sbn9: str
	secondary_address: str
	seed_instance: str
	sentence: str
	sentences: str
	sex: str
	sha1: str
	sha256: str
	simple_profile: str
	slug: str
	spices: str
	ssn: str
	state: str
	state_abbr: str
	stock_exchange: str
	stock_name: str
	stock_ticker: str
	street_address: str
	street_name: str
	street_number: str
	street_suffix: str
	suffix: str
	suffix_female: str
	suffix_male: str
	suffix_nonbinary: str
	surname: str
	swift: str
	swift11: str
	swift8: str
	tar: str
	telephone: str
	text: str
	texts: str
	time: str
	time_delta: str
	time_object: str
	time_series: str
	timestamp: str
	timezone: str
	title: str
	tld: str
	tsv: str
	university: str
	unix_device: str
	unix_partition: str
	unix_time: str
	upc_a: str
	upc_e: str
	uri: str
	uri_extension: str
	uri_page: str
	uri_path: str
	url: str
	user_agent: str
	user_name: str
	username: str
	uuid4: str
	vegetable: str
	views_on: str
	vin: str
	week_date: str
	weight: str
	windows_platform_token: str
	word: str
	words: str
	worldview: str
	year: str
	zip: str
	zip_code: str
	zipcode: str
	zipcode_in_state: str
	zipcode_plus4: str
#"""

"""
Field Error: add_provider
Field Error: bulk_create_datetimes
Field Error: del_arguments
Field Error: enum
Field Error: format
Field Error: get_arguments
Field Error: get_formatter
Field Error: override_locale
Field Error: parse
Field Error: provider
Field Error: seed_locale
Field Error: set_arguments
Field Error: set_formatter
Field Error: update_dataset
Field Error: validate_enum
Field Error: xml
"""
