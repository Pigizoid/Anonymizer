import pydantic

from sm.functions.synthesiser import Synthesiser


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
    def anonymise(schema_model, data, method, manual, fields):
        if manual == True:
            field_names = fields.keys()
        else:  # auto
            name_match_pairs = Synthesiser.match_fields(schema_model, method)[0]
            field_names = [name for name in name_match_pairs[0] if name != ""]
        new_schema_model = Anonymiser.subset_model(schema_model, field_names)

        return_data = dict(Synthesiser.synthesise(new_schema_model, method)[0])
        new_fields = data.copy()
        for field in field_names:
            new_fields[field] = return_data[field]
        anonymised_data = new_fields
        return anonymised_data
