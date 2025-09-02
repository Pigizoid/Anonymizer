import pydantic

from .synthesiser import Synthesiser


class Anonymiser:
    @staticmethod
    def subset_model(schema_model, field_names):
        fields = {
            name: (field.annotation, field.default)
            for name, field in schema_model.model_fields.items()
            if name in field_names
        }
        return pydantic.create_model("new_schema_model", **fields)

    @staticmethod
    def guess_type(value):
        if value is None:
            return_type = type(None)
        try:
            int(value)
            return_type = int
        except:
            pass
        try:
            float(value)
            return_type = float
        except:
            pass
        try:
            if value.lower() in {"true", "false"}:
                return_type = bool
        except:
            pass
        if isinstance(value, bytes):
            return_type = bytes
        elif isinstance(value, tuple):
            return_type = tuple
        elif isinstance(value, list):
            return_type = list
        elif isinstance(value, set):
            return_type = set
        elif isinstance(value, frozenset):
            return_type = frozenset
        elif isinstance(value, dict):
            return_type = dict
        else:
            return_type = str
        return return_type

    @staticmethod
    def new_model(data):
        fields = {
            name: (Anonymiser.guess_type(content))
            for name, content in data.items()
        }
        return pydantic.create_model("new_schema_model", **fields)

    @staticmethod
    def anonymise(schema_model, data, method, manual, fields):
        print(data)
        synth = Synthesiser(method=method)
        schema_match = True
        if manual == True:
            field_names = fields.keys()
        else:  # auto
            try:
                schema_model(**data)
                field_names = [x[0] for x in synth.get_model_data(schema_model)]
            except:
                field_names = data.keys()
                print(f"Schema '{schema_model.__name__}' does not match data, defaulting to data keys")
                schema_match = False
            name_match_pairs = synth.match_fields(field_names)
            field_names = [ field for field,match in name_match_pairs.items() if match != ""]
        if schema_match:
            new_schema_model = Anonymiser.subset_model(schema_model, field_names)
        else:
            new_schema_model = Anonymiser.new_model(data)
        return_data = [dict(x) for x in synth.synthesise(new_schema_model, method)]
        anonymised_data = []
        for return_entry in return_data:
            new_fields = data.copy()
            for field in field_names:
                new_fields[field] = return_entry[field]
            anonymised_data.append(new_fields)
        return anonymised_data
