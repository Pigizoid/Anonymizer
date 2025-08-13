


import pydantic
from pydantic import BaseModel
from pydantic_core import core_schema

from functions.synthesiser import Synthesiser


class Anonymiser():
	
	@staticmethod
	def anonymise(schema_model,method,data,config):
		print("▁"*len(str(config)))
		print(config)
		print("Config:")
		if config["method"] == "manual":
			fields = config["fields"]
			field_names = fields.keys()
			
		else: #auto
			name_match_pairs = Synthesiser.match_fields(schema_model,method)
			field_names = [name for name in name_match_pairs[0] if name!=""]
		
		new_schema_model = Anonymiser.subset_model(schema_model,field_names)
		
		return_data = dict(Synthesiser.synthesise(new_schema_model,method)[0])
		new_fields = data.copy()
		for field in field_names:
			new_fields[field] = return_data[field]
		anonymised_data = new_fields
		print("▁"*len(str(config)))
		print(f"Input data:\n\t{data}")
		print(f"Anonymised:\n\t{anonymised_data}")
		return anonymised_data
	
	@staticmethod
	def subset_model(schema_model, field_names):
		fields = {
			name: (field.annotation, field.default)
			for name, field in schema_model.__fields__.items()
			if name in field_names
		}
		return pydantic.create_model("new_schema_model", **fields)
