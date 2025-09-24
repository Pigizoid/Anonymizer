import typer
from typing import Annotated, Optional
from sm.app.synth.funcs import synth_func
from sm.app.helper_funcs import (
    return_flags, 
    load_recursed_path, 
    load_schema, 
    load_file_path, 
    close_folder, 
    recursive_folder_schema_handler, 
    load_output_path_flag
)
from sm.app.models import SynthesiserConfig
from pathlib import Path


synth_batch_subcommand = typer.Typer()


def synth_batch_func(schema_model,output_file_path,flags):
    seed = flags.seed
    synth_flags = flags.synth
    load_file_path(output_file_path)

    method = synth_flags.method
    amount = synth_flags.amount
    if synth_flags.batch == 0:
        batch = amount
    else:
        batch = synth_flags.batch
    cout = synth_flags.cout
    batch_index = 0
    for y in range(amount // batch):
        print(
            f"Batch num: {y + 1} of {amount // batch} | Batch amount: {batch} | Completed: {batch_index}/{amount}  {'-' * 50}"
        )
        synth_func(
            schema_model,
            method,
            batch,
            output_file_path,
            cout=cout,
            start_index=batch_index,
            seed=seed,
        )
        batch_index += batch
    if amount - batch_index != 0:
        synth_func(
            schema_model,
            method,
            amount - batch_index,
            output_file_path,
            cout=cout,
            start_index=batch_index,
            seed=seed,
        )

    close_folder(output_file_path)


@synth_batch_subcommand.command(name="batch")
def synth_batch_command(
    ctx: typer.Context,  # contains ctx.config
    method: str = None,
    amount: int = None,
    batch: int = None,
    output: str = None,
    cout: Annotated[Optional[bool], typer.Option("--cout/--no-cout")] = None,
):
    """
    A subcommand for the synth command\n
    Inputs:\n
        a method of the methods "mixed","mimesis","faker"
        an amount to generate per schema as an int
        a batch amount as an int
            generate "amount" total in "batch" sizes
            e.g. amount=100 batch=50 means 2 batches of 50
        a filename as str for the output file (.json added by default)
        cout boolean to toggle verbose printing
    Runs the synthesiser tool in batch mode\n
    """
    flags = return_flags(ctx, SynthesiserConfig)
    print(f"Args: {flags}")
    schema_models = load_recursed_path(Path(flags.schema_path),flags.schema_type,load_schema)
    output_path_name = load_output_path_flag(flags.synth.output)
    recursive_folder_schema_handler(schema_models,synth_batch_func,flags,output_path_name)


'''
{
    "$defs": {
        "New_Address1": {
            "properties": {
                "street": {
                    "title": "Street",
                    "type": "string"
                },
                "city": {
                    "title": "City",
                    "type": "string"
                },
                "zip_code": {
                    "items": {
                        "patternProperties": {
                            "^\\d{3}(-\\d{6})?$": {
                                "items": {
                                    "pattern": "^\\d{5}(-\\d{4})?$",
                                    "type": "string"
                                },
                                "type": "array"
                            }
                        },
                        "type": "object"
                    },
                    "title": "Zip Code",
                    "type": "array"
                },
                "country": {
                    "default": "USA",
                    "title": "Country",
                    "type": "string"
                },
                "social_security_number": {
                    "title": "Social Security Number",
                    "type": "string"
                },
                "continent": {
                    "anyOf": [
                        {
                            "type": "string"
                        },
                        {
                            "type": "null"
                        }
                    ],
                    "title": "Continent"
                },
                "phone_number": {
                    "items": {
                        "additionalProperties": {
                            "items": {
                                "type": "string"
                            },
                            "type": "array"
                        },
                        "type": "object"
                    },
                    "title": "Phone Number",
                    "type": "array"
                },
                "title": {
                    "items": {
                        "type": "string"
                    },
                    "title": "Title",
                    "type": "array",
                    "uniqueItems": true
                },
                "direction": {
                    "enum": [
                        "north",
                        "south",
                        "east",
                        "west"
                    ],
                    "title": "Direction",
                    "type": "string"
                },
                "bananas": {
                    "title": "Bananas",
                    "type": "string"
                },
                "user": {
                    "$ref": "#/$defs/USER"
                },
                "name": {
                    "additionalProperties": {
                        "maxLength": 2000,
                        "type": "string"
                    },
                    "minProperties": 30,
                    "propertyNames": {
                        "maxLength": 1000
                    },
                    "title": "Name",
                    "type": "object"
                }
            },
            "required": [
                "street",
                "city",
                "zip_code",
                "social_security_number",
                "continent",
                "phone_number",
                "title",
                "direction",
                "bananas",
                "user",
                "name"
            ],
            "title": "New_Address1",
            "type": "object"
        },
        "New_Address2": {
            "properties": {
                "field_decimal": {
                    "anyOf": [
                        {
                            "exclusiveMaximum": 15.0,
                            "exclusiveMinimum": 3.0,
                            "multipleOf": 5e-06,
                            "type": "number"
                        },
                        {
                            "type": "string"
                        }
                    ],
                    "title": "Field Decimal"
                },
                "field_float": {
                    "exclusiveMinimum": 15,
                    "title": "Field Float",
                    "type": "number"
                }
            },
            "required": [
                "field_decimal",
                "field_float"
            ],
            "title": "New_Address2",
            "type": "object"
        },
        "New_Address3": {
            "properties": {
                "username": {
                    "maxLength": 20,
                    "minLength": 3,
                    "pattern": "^[a-zA-Z0-9_]+$",
                    "title": "Username",
                    "type": "string"
                },
                "email": {
                    "anyOf": [
                        {
                            "type": "string"
                        },
                        {
                            "type": "null"
                        }
                    ],
                    "title": "Email"
                },
                "age": {
                    "exclusiveMaximum": 100,
                    "exclusiveMinimum": 12,
                    "title": "Age",
                    "type": "integer"
                },
                "bio": {
                    "anyOf": [
                        {
                            "maxLength": 250,
                            "type": "string"
                        },
                        {
                            "type": "null"
                        }
                    ],
                    "default": null,
                    "title": "Bio"
                },
                "interests": {
                    "items": {
                        "type": "string"
                    },
                    "title": "Interests",
                    "type": "array"
                }
            },
            "required": [
                "username",
                "email",
                "age"
            ],
            "title": "New_Address3",
            "type": "object"
        },
        "New_Address4": {
            "properties": {
                "user": {
                    "$ref": "#/$defs/USER"
                },
                "street": {
                    "title": "Street",
                    "type": "string"
                }
            },
            "required": [
                "user",
                "street"
            ],
            "title": "New_Address4",
            "type": "object"
        },
        "USER": {
            "properties": {
                "name": {
                    "title": "Name",
                    "type": "string"
                }
            },
            "required": [
                "name"
            ],
            "title": "USER",
            "type": "object"
        }
    },
    "properties": {
        "street": {
            "title": "Street",
            "type": "string"
        },
        "city": {
            "title": "City",
            "type": "string"
        },
        "zip_code": {
            "items": {
                "patternProperties": {
                    "^\\d{3}(-\\d{6})?$": {
                        "items": {
                            "pattern": "^\\d{5}(-\\d{4})?$",
                            "type": "string"
                        },
                        "type": "array"
                    }
                },
                "type": "object"
            },
            "title": "Zip Code",
            "type": "array"
        },
        "country": {
            "default": "USA",
            "title": "Country",
            "type": "string"
        },
        "social_security_number": {
            "title": "Social Security Number",
            "type": "string"
        },
        "continent": {
            "anyOf": [
                {
                    "type": "string"
                },
                {
                    "type": "null"
                }
            ],
            "title": "Continent"
        },
        "phone_number": {
            "items": {
                "additionalProperties": {
                    "items": {
                        "type": "string"
                    },
                    "type": "array"
                },
                "type": "object"
            },
            "title": "Phone Number",
            "type": "array"
        },
        "title": {
            "items": {
                "type": "string"
            },
            "title": "Title",
            "type": "array",
            "uniqueItems": true
        },
        "direction": {
            "enum": [
                "north",
                "south",
                "east",
                "west"
            ],
            "title": "Direction",
            "type": "string"
        },
        "bananas": {
            "title": "Bananas",
            "type": "string"
        },
        "user": {
            "$ref": "#/$defs/USER"
        },
        "name": {
            "additionalProperties": {
                "maxLength": 2000,
                "type": "string"
            },
            "minProperties": 30,
            "propertyNames": {
                "maxLength": 1000
            },
            "title": "Name",
            "type": "object"
        },
        "field_decimal_constr": {
            "anyOf": [
                {
                    "exclusiveMaximum": 15.0,
                    "exclusiveMinimum": 3.0,
                    "multipleOf": 5e-06,
                    "type": "number"
                },
                {
                    "type": "string"
                }
            ],
            "title": "Field Decimal Constr"
        },
        "field_float_constr": {
            "exclusiveMinimum": 15,
            "title": "Field Float Constr",
            "type": "number"
        },
        "field_str": {
            "title": "Field Str",
            "type": "string"
        },
        "field_int": {
            "title": "Field Int",
            "type": "integer"
        },
        "field_float": {
            "title": "Field Float",
            "type": "number"
        },
        "field_bool": {
            "title": "Field Bool",
            "type": "boolean"
        },
        "field_complex": {
            "title": "Field Complex",
            "type": "string"
        },
        "field_bytes": {
            "format": "binary",
            "title": "Field Bytes",
            "type": "string"
        },
        "field_tuple": {
            "items": {},
            "title": "Field Tuple",
            "type": "array"
        },
        "field_list": {
            "items": {},
            "title": "Field List",
            "type": "array"
        },
        "field_set": {
            "items": {},
            "title": "Field Set",
            "type": "array",
            "uniqueItems": true
        },
        "field_frozenset": {
            "items": {},
            "title": "Field Frozenset",
            "type": "array",
            "uniqueItems": true
        },
        "field_dict": {
            "additionalProperties": true,
            "title": "Field Dict",
            "type": "object"
        },
        "username": {
            "maxLength": 20,
            "minLength": 3,
            "pattern": "^[a-zA-Z0-9_]+$",
            "title": "Username",
            "type": "string"
        },
        "email": {
            "anyOf": [
                {
                    "type": "string"
                },
                {
                    "type": "null"
                }
            ],
            "title": "Email"
        },
        "age": {
            "exclusiveMaximum": 100,
            "exclusiveMinimum": 12,
            "title": "Age",
            "type": "integer"
        },
        "bio": {
            "anyOf": [
                {
                    "maxLength": 250,
                    "type": "string"
                },
                {
                    "type": "null"
                }
            ],
            "default": null,
            "title": "Bio"
        },
        "interests": {
            "items": {
                "type": "string"
            },
            "title": "Interests",
            "type": "array"
        },
        "n1": {
            "$ref": "#/$defs/New_Address1"
        },
        "n2": {
            "$ref": "#/$defs/New_Address2"
        },
        "n3": {
            "$ref": "#/$defs/New_Address3"
        },
        "n4": {
            "$ref": "#/$defs/New_Address4"
        }
    },
    "required": [
        "street",
        "city",
        "zip_code",
        "social_security_number",
        "continent",
        "phone_number",
        "title",
        "direction",
        "bananas",
        "user",
        "name",
        "field_decimal_constr",
        "field_float_constr",
        "field_str",
        "field_int",
        "field_float",
        "field_bool",
        "field_complex",
        "field_bytes",
        "field_tuple",
        "field_list",
        "field_set",
        "field_frozenset",
        "field_dict",
        "username",
        "email",
        "age",
        "n1",
        "n2",
        "n3",
        "n4"
    ],
    "title": "Address",
    "type": "object"
}
'''

