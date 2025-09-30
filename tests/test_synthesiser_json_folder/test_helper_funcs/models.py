from sm.library.jsonschemaclass import JsonSchemaClass


test_Address = JsonSchemaClass("test_Address",{
    "properties":{
        "street": {"type":"string"},
        "city": {"type":"string"}
    },
    "required":["street","city"],
    "type":"object",
    "title":"test_Address"
})

test_Address_2 = JsonSchemaClass("test_Address_2",{
    "properties":{
        "street": {"type":"string"},
        "city": {"type":"string"},
        "social_security_number": {"type":"string"},
        "continent": {"type":"string"}
    },
    "required":["street","city","social_security_number","continent"],
    "type":"object",
    "title":"test_Address_2"
})

test_Address_3 = JsonSchemaClass("test_Address_3",{
        "properties": {
                "street": {
                        "title": "Street",
                        "type": "string"
                },
                "city": {
                        "title": "City",
                        "type": "string"
                },
                "social_security_number": {
                        "title": "Social Security Number",
                        "type": "string"
                },
                "continent": {
                        "title": "Continent",
                        "type": "string"
                }
        },
        "required": [
                "street",
                "city",
                "social_security_number",
                "continent"
        ],
        "title": "test_Address_3",
        "type": "object"
})

test_Address_4 = JsonSchemaClass("test_Address_4",{
        "$defs": {
                "test_Address_3": {
                        "properties": {
                                "street": {
                                        "title": "Street",
                                        "type": "string"
                                },
                                "city": {
                                        "title": "City",
                                        "type": "string"
                                },
                                "social_security_number": {
                                        "title": "Social Security Number",
                                        "type": "string"
                                },
                                "continent": {
                                        "title": "Continent",
                                        "type": "string"
                                }
                        },
                        "required": [
                                "street",
                                "city",
                                "social_security_number",
                                "continent"
                        ],
                        "title": "test_Address_3",
                        "type": "object"
                }
        },
        "properties": {
                "name": {
                        "title": "Name",
                        "type": "string"
                },
                "phone_number": {
                        "title": "Phone Number",
                        "type": "string"
                },
                "social_security_number": {
                        "title": "Social Security Number",
                        "type": "string"
                },
                "extra": {
                        "$ref": "#/$defs/test_Address_3"
                }
        },
        "required": [
                "name",
                "phone_number",
                "social_security_number",
                "extra"
        ],
        "title": "test_Address_4",
        "type": "object"
})

Constraints6 = JsonSchemaClass("Constraints6",{
        "properties": {
                "constr_strip_whitespace": {
                        "title": "Constr Strip Whitespace",
                        "type": "string"
                },
                "constr_to_upper": {
                        "title": "Constr To Upper",
                        "type": "string"
                },
                "constr_to_lower": {
                        "title": "Constr To Lower",
                        "type": "string"
                },
                "constr_strict": {
                        "title": "Constr Strict",
                        "type": "string"
                },
                "constr_default": {
                        "default": "AAA",
                        "title": "Constr Default",
                        "type": "string"
                },
                "constr_annotation": {
                        "title": "Constr Annotation",
                        "type": "string"
                },
                "constr_min_length": {
                        "minLength": 5,
                        "title": "Constr Min Length",
                        "type": "string"
                },
                "constr_max_length": {
                        "maxLength": 5,
                        "title": "Constr Max Length",
                        "type": "string"
                },
                "constr_pattern": {
                        "pattern": "^\\d{5}(-\\d{4})?$",
                        "title": "Constr Pattern",
                        "type": "string"
                },
                "constr_gt": {
                        "exclusiveMinimum": 5,
                        "title": "Constr Gt",
                        "type": "integer"
                },
                "constr_lt": {
                        "exclusiveMaximum": 5,
                        "title": "Constr Lt",
                        "type": "integer"
                },
                "constr_ge": {
                        "minimum": 5,
                        "title": "Constr Ge",
                        "type": "integer"
                },
                "constr_le": {
                        "maximum": 5,
                        "title": "Constr Le",
                        "type": "integer"
                },
                "constr_multiple_of": {
                        "multipleOf": 5,
                        "title": "Constr Multiple Of",
                        "type": "integer"
                },
                "constr_allow_inf_nan": {
                        "title": "Constr Allow Inf Nan",
                        "type": "integer"
                },
                "constr_max_digits": {
                        "anyOf": [
                                {
                                        "type": "number"
                                },
                                {
                                        "type": "string"
                                }
                        ],
                        "title": "Constr Max Digits"
                },
                "constr_decimal_places": {
                        "anyOf": [
                                {
                                        "type": "number"
                                },
                                {
                                        "type": "string"
                                }
                        ],
                        "title": "Constr Decimal Places"
                },
                "constr_origin": {
                        "items": {
                                "type": "string"
                        },
                        "title": "Constr Origin",
                        "type": "array"
                },
                "constr_args": {
                        "additionalProperties": {
                                "type": "string"
                        },
                        "title": "Constr Args",
                        "type": "object"
                },
                "constr_required": {
                        "title": "Constr Required",
                        "type": "string"
                }
        },
        "required": [
                "constr_strict",
                "constr_annotation",
                "constr_min_length",
                "constr_max_length",
                "constr_pattern",
                "constr_gt",
                "constr_lt",
                "constr_ge",
                "constr_le",
                "constr_multiple_of",
                "constr_allow_inf_nan",
                "constr_max_digits",
                "constr_decimal_places",
                "constr_origin",
                "constr_args",
                "constr_required"
        ],
        "title": "Constraints6",
        "type": "object"
})

Constraints = JsonSchemaClass("Constraints",{
        "$defs": {
                "ConstraintsNested": {
                        "properties": {
                                "test_none": {
                                        "title": "Test None",
                                        "type": "null"
                                },
                                "test_basic": {
                                        "title": "Test Basic",
                                        "type": "string"
                                },
                                "test_pattern": {
                                        "pattern": "^a$",
                                        "title": "Test Pattern",
                                        "type": "string"
                                },
                                "test_list": {
                                        "items": {
                                                "type": "string"
                                        },
                                        "title": "Test List",
                                        "type": "array"
                                },
                                "test_dict": {
                                        "additionalProperties": {
                                                "type": "string"
                                        },
                                        "title": "Test Dict",
                                        "type": "object"
                                },
                                "test_tuple": {
                                        "maxItems": 3,
                                        "minItems": 3,
                                        "prefixItems": [
                                                {
                                                        "type": "string"
                                                },
                                                {
                                                        "type": "string"
                                                },
                                                {
                                                        "type": "string"
                                                }
                                        ],
                                        "title": "Test Tuple",
                                        "type": "array"
                                },
                                "test_set": {
                                        "items": {
                                                "type": "string"
                                        },
                                        "title": "Test Set",
                                        "type": "array",
                                        "uniqueItems": True
                                },
                                "test_union": {
                                        "anyOf": [
                                                {
                                                        "type": "string"
                                                },
                                                {
                                                        "type": "null"
                                                }
                                        ],
                                        "title": "Test Union"
                                },
                                "test_literal": {
                                        "enum": [
                                                "1",
                                                "2",
                                                "3",
                                                "4"
                                        ],
                                        "title": "Test Literal",
                                        "type": "string"
                                },
                                "test_recursive": {
                                        "items": {
                                                "additionalProperties": {
                                                        "items": {
                                                                "maxItems": 2,
                                                                "minItems": 2,
                                                                "prefixItems": [
                                                                        {
                                                                                "type": "string"
                                                                        },
                                                                        {
                                                                                "type": "string"
                                                                        }
                                                                ],
                                                                "type": "array"
                                                        },
                                                        "type": "array"
                                                },
                                                "type": "object"
                                        },
                                        "title": "Test Recursive",
                                        "type": "array"
                                },
                                "test_list_length": {
                                        "items": {
                                                "type": "string"
                                        },
                                        "minItems": 20,
                                        "title": "Test List Length",
                                        "type": "array",
                                        "uniqueItems": True
                                }
                        },
                        "required": [
                                "test_none",
                                "test_basic",
                                "test_pattern",
                                "test_list",
                                "test_dict",
                                "test_tuple",
                                "test_set",
                                "test_union",
                                "test_literal",
                                "test_recursive",
                                "test_list_length"
                        ],
                        "title": "ConstraintsNested",
                        "type": "object"
                }
        },
        "properties": {
                "constr_strip_whitespace": {
                        "title": "Constr Strip Whitespace",
                        "type": "string"
                },
                "constr_to_upper": {
                        "title": "Constr To Upper",
                        "type": "string"
                },
                "constr_to_lower": {
                        "title": "Constr To Lower",
                        "type": "string"
                },
                "constr_strict": {
                        "title": "Constr Strict",
                        "type": "string"
                },
                "constr_default": {
                        "default": "AAA",
                        "title": "Constr Default",
                        "type": "string"
                },
                "constr_annotation": {
                        "title": "Constr Annotation",
                        "type": "string"
                },
                "constr_min_length": {
                        "minLength": 5,
                        "title": "Constr Min Length",
                        "type": "string"
                },
                "constr_max_length": {
                        "maxLength": 5,
                        "title": "Constr Max Length",
                        "type": "string"
                },
                "constr_pattern": {
                        "pattern": "^\\d{5}(-\\d{4})?$",
                        "title": "Constr Pattern",
                        "type": "string"
                },
                "constr_gt": {
                        "exclusiveMinimum": 5,
                        "title": "Constr Gt",
                        "type": "integer"
                },
                "constr_lt": {
                        "exclusiveMaximum": 5,
                        "title": "Constr Lt",
                        "type": "integer"
                },
                "constr_ge": {
                        "minimum": 5,
                        "title": "Constr Ge",
                        "type": "integer"
                },
                "constr_le": {
                        "maximum": 5,
                        "title": "Constr Le",
                        "type": "integer"
                },
                "constr_multiple_of": {
                        "multipleOf": 5,
                        "title": "Constr Multiple Of",
                        "type": "integer"
                },
                "constr_allow_inf_nan": {
                        "title": "Constr Allow Inf Nan",
                        "type": "integer"
                },
                "constr_max_digits": {
                        "anyOf": [
                                {
                                        "type": "number"
                                },
                                {
                                        "type": "string"
                                }
                        ],
                        "title": "Constr Max Digits"
                },
                "constr_decimal_places": {
                        "anyOf": [
                                {
                                        "type": "number"
                                },
                                {
                                        "type": "string"
                                }
                        ],
                        "title": "Constr Decimal Places"
                },
                "constr_origin": {
                        "items": {
                                "type": "string"
                        },
                        "title": "Constr Origin",
                        "type": "array"
                },
                "constr_args": {
                        "additionalProperties": {
                                "type": "string"
                        },
                        "title": "Constr Args",
                        "type": "object"
                },
                "constr_required": {
                        "title": "Constr Required",
                        "type": "string"
                },
                "nested_1": {
                        "$ref": "#/$defs/ConstraintsNested"
                },
                "nested_2": {
                        "$ref": "#/$defs/ConstraintsNested"
                }
        },
        "required": [
                "constr_strict",
                "constr_annotation",
                "constr_min_length",
                "constr_max_length",
                "constr_pattern",
                "constr_gt",
                "constr_lt",
                "constr_ge",
                "constr_le",
                "constr_multiple_of",
                "constr_allow_inf_nan",
                "constr_max_digits",
                "constr_decimal_places",
                "constr_origin",
                "constr_args",
                "constr_required",
                "nested_1",
                "nested_2"
        ],
        "title": "Constraints",
        "type": "object"
})

ConstraintsNested = JsonSchemaClass("ConstraintsNested",{
        "properties": {
                "test_none": {
                        "title": "Test None",
                        "type": "null"
                },
                "test_basic": {
                        "title": "Test Basic",
                        "type": "string"
                },
                "test_pattern": {
                        "pattern": "^a$",
                        "title": "Test Pattern",
                        "type": "string"
                },
                "test_list": {
                        "items": {
                                "type": "string"
                        },
                        "title": "Test List",
                        "type": "array"
                },
                "test_dict": {
                        "additionalProperties": {
                                "type": "string"
                        },
                        "title": "Test Dict",
                        "type": "object"
                },
                "test_tuple": {
                        "maxItems": 3,
                        "minItems": 3,
                        "prefixItems": [
                                {
                                        "type": "string"
                                },
                                {
                                        "type": "string"
                                },
                                {
                                        "type": "string"
                                }
                        ],
                        "title": "Test Tuple",
                        "type": "array"
                },
                "test_set": {
                        "items": {
                                "type": "string"
                        },
                        "title": "Test Set",
                        "type": "array",
                        "uniqueItems": True
                },
                "test_union": {
                        "anyOf": [
                                {
                                        "type": "string"
                                },
                                {
                                        "type": "null"
                                }
                        ],
                        "title": "Test Union"
                },
                "test_literal": {
                        "enum": [
                                "1",
                                "2",
                                "3",
                                "4"
                        ],
                        "title": "Test Literal",
                        "type": "string"
                },
                "test_recursive": {
                        "items": {
                                "additionalProperties": {
                                        "items": {
                                                "maxItems": 2,
                                                "minItems": 2,
                                                "prefixItems": [
                                                        {
                                                                "type": "string"
                                                        },
                                                        {
                                                                "type": "string"
                                                        }
                                                ],
                                                "type": "array"
                                        },
                                        "type": "array"
                                },
                                "type": "object"
                        },
                        "title": "Test Recursive",
                        "type": "array"
                },
                "test_list_length": {
                        "items": {
                                "type": "string"
                        },
                        "minItems": 20,
                        "title": "Test List Length",
                        "type": "array",
                        "uniqueItems": True
                }
        },
        "required": [
                "test_none",
                "test_basic",
                "test_pattern",
                "test_list",
                "test_dict",
                "test_tuple",
                "test_set",
                "test_union",
                "test_literal",
                "test_recursive",
                "test_list_length"
        ],
        "title": "ConstraintsNested",
        "type": "object"
})

generate_test1 = JsonSchemaClass("generate_test1",{
        "properties": {
                "test_none": {
                        "title": "Test None",
                        "type": "null"
                },
                "test_basic": {
                        "title": "Test Basic",
                        "type": "string"
                },
                "test_pattern": {
                        "pattern": "^a$",
                        "title": "Test Pattern",
                        "type": "string"
                },
                "test_list": {
                        "items": {
                                "type": "string"
                        },
                        "title": "Test List",
                        "type": "array"
                },
                "test_dict": {
                        "additionalProperties": {
                                "type": "string"
                        },
                        "title": "Test Dict",
                        "type": "object"
                },
                "test_tuple": {
                        "maxItems": 3,
                        "minItems": 3,
                        "prefixItems": [
                                {
                                        "type": "string"
                                },
                                {
                                        "type": "string"
                                },
                                {
                                        "type": "string"
                                }
                        ],
                        "title": "Test Tuple",
                        "type": "array"
                },
                "test_set": {
                        "items": {
                                "type": "string"
                        },
                        "title": "Test Set",
                        "type": "array",
                        "uniqueItems": True
                },
                "test_union": {
                        "anyOf": [
                                {
                                        "type": "string"
                                },
                                {
                                        "type": "null"
                                }
                        ],
                        "title": "Test Union"
                },
                "test_literal": {
                        "enum": [
                                "1",
                                "2",
                                "3",
                                "4"
                        ],
                        "title": "Test Literal",
                        "type": "string"
                },
                "test_recursive": {
                        "items": {
                                "additionalProperties": {
                                        "items": {
                                                "maxItems": 2,
                                                "minItems": 2,
                                                "prefixItems": [
                                                        {
                                                                "type": "string"
                                                        },
                                                        {
                                                                "type": "string"
                                                        }
                                                ],
                                                "type": "array"
                                        },
                                        "type": "array"
                                },
                                "type": "object"
                        },
                        "title": "Test Recursive",
                        "type": "array"
                },
                "test_list_length": {
                        "items": {
                                "type": "string"
                        },
                        "minItems": 20,
                        "title": "Test List Length",
                        "type": "array",
                        "uniqueItems": True
                }
        },
        "required": [
                "test_none",
                "test_basic",
                "test_pattern",
                "test_list",
                "test_dict",
                "test_tuple",
                "test_set",
                "test_union",
                "test_literal",
                "test_recursive",
                "test_list_length"
        ],
        "title": "generate_test1",
        "type": "object"
})

generate_test2 = JsonSchemaClass("generate_test2",{
        "properties": {
                "test_dict_fail": {
                        "additionalProperties": {
                                "type": "string"
                        },
                        "minProperties": 20,
                        "title": "Test Dict Fail",
                        "type": "object"
                }
        },
        "required": [
                "test_dict_fail"
        ],
        "title": "generate_test2",
        "type": "object"
})

User = JsonSchemaClass("User",{
    "properties":{
        "id": {"type":"integer"},
        "name": {"type":"string"}
    },
    "required": ["id","name"],
    "type":"object",
    "title":"User"
})




