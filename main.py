from pydantic import BaseModel, Field
import typing
from typing_extensions import Annotated

import typer

from functions.synthesiser import Synthesiser


from functions.anonymiser import Anonymiser


import yaml
import json


import importlib.util
import inspect


app = typer.Typer()

def synth_func(schema_model,method,amount,file_path,start_index=0,cout:bool=False):
	dataset = Synthesiser.synthesise(schema_model,method,amount)  #returns as [ data, data, ... ]
	flush = []
	for index,data in enumerate(dataset):
		if file_path != None:
			output_data = {key: str(value) for key, value in dict(data).items()}  #convert outputs to string
			if (index+1+start_index) != 1:
				front_string = ",\n    "
			else:
				front_string = ""
			flush.append(f'{front_string}"{index+1+start_index}": {json.dumps(output_data,indent=8)}')
		else:
			flush.append(f"{index+1+start_index}: {data}")
		if (index+1)%max((amount//10),1) == 0:
			if file_path != None:
				flush_out = "".join(flush)
				with open(f"{file_path}.json","a") as f:
					f.write(flush_out)
			else:
				flush_out = "".join(flush)
			if cout != False:
				print(flush_out)
			flush.clear()
	if flush != []:
		if file_path != None:
			flush_out = "".join(flush)
			with open(f"{file_path}.json","a") as f:
				f.write(flush_out)
		else:
			flush_out = "".join(flush)
		if cout != False:
			print(flush_out)
		flush.clear()
	if file_path != None and cout != False:
		print(f"To file_path -> {file_path}")


def anon_func(schema_model,data,method,file_path,cout,manual,fields):
	anonymised_data = Anonymiser.anonymise(schema_model,data,method,manual,fields)
	
	if cout != False:
		L = len(f"        {data}")
		print("_"*L)
		print(f"Input data:\n\t{data}")
		print(f"Anonymised:\n\t{anonymised_data}")
		print("_"*L)
		
	flush_output = f'"1":{json.dumps(anonymised_data,indent=8)}'
	if file_path!= None:
		with open(f"{file_path}.json","a") as f:
			f.write(flush_output)


def load_config(config):
	with open(config) as cf_file:
		try:
			#print(yaml.safe_load(cf_file))
			config_data = dict(yaml.safe_load(cf_file))
		except yaml.YAMLError as err:
			print(err)
	
	try:
		schema_file = config_data["schema"]
	except:
		print("Config 'schema' not defined")
		return 0
	
	if not(schema_file.endswith(".py")):
		schema_file+=".py"
	
	spec = importlib.util.spec_from_file_location("imported_schema_model", schema_file)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	
	classes = inspect.getmembers(module, inspect.isclass)
	#print(classes)
	filtered = [(name, cls) for name, cls in classes if cls.__module__ == 'imported_schema_model']
	schema_model = filtered[0][1]
	#print(filtered)
	#print(schema_model)
	#User = module.User
	
	try:
		synthesiser_config = config_data["synthesiser"]
	except:
		synthesiser_config = []
	
	try:
		anonymiser_config = config_data["anonymiser"]
	except:
		anonymiser_config = []
	
	
	return {"schema":schema_model,"s":synthesiser_config,"a":anonymiser_config}


def load_folder(output):
	import os
	dir_name = "outputs"
	file_path = f"{dir_name}\\{output}"
	if os.path.isdir(dir_name) == False:
		try:
			os.mkdir(dir_name)
			print(f"Directory '{dir_name}' created successfully")
			with open(f"{file_path}.json","w") as f:
				f.write("{")
		except PermissionError:
			print(f"Permission denied: Unable to create '{dir_name}'")
		except Exception as e:
			print(f"An error occurred: {e}")
	else:
		with open(f"{file_path}.json","w") as f:
			f.write("{")
	return file_path

def close_folder(file_path):
	if file_path != None:
		with open(f"{file_path}.json","a") as f:
			f.write("\n}\n")



class SynthesiserConfig(BaseModel):
	method: str = Field(default = "faker")
	amount: int = Field(default = 1)
	batch: int = Field(default = 0)
	output: str = Field(default = "")
	cout: bool = Field(default = False)

class AnonymiserConfig(BaseModel):
	ingest: str = Field(default = "")
	method: str = Field(default = "faker")
	output: str = Field(default = "")
	cout: bool = Field(default = False)
	manual: bool = Field(default = False)
	fields: typing.Dict[str,str] = Field(default = {})


def load_flags(func_type,flags):
	if func_type == "s":
		config_schema = SynthesiserConfig
	elif func_type == "a":
		config_schema = AnonymiserConfig
	
	unchanged = [field_name for field_name in config_schema.model_fields.keys() if flags[field_name] == None ]
	flag_data = dict(config_schema(**{}))
	
	if flags["config"] is not None:
		config_data = load_config(flags["config"])
		schema_model = config_data["schema"]
		if config_data[func_type]:
			flag_data.update(config_data[func_type])
		else:
			if func_type == "s":
				print("Config 'synthesiser' not defined")
			elif func_type == "a":
				print("Config 'anonymiser' not defined")
	else:
		print("Schema auto loader")
		import schema
		schema_model = schema.Address
	
	cli_args = { field_name: flags[field_name] for field_name in config_schema.model_fields.keys()}
	flag_data.update({k: v for k, v in cli_args.items() if k not in unchanged})
	
	return (flag_data,schema_model)




@app.command()
def synthesise(
		method:str=None,
		amount:int=None,
		batch:int=None,
		output:str=None,
		cout:Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
		config:str=None
	):
	
	flags = {"method":method,"amount":amount,"batch":batch,"output":output,"cout":cout,"config":config}
	
	synthesiser_data,schema_model = load_flags("s",flags)
	
	method=synthesiser_data["method"]
	amount=synthesiser_data["amount"]
	batch=synthesiser_data["batch"]
	output=synthesiser_data["output"]
	cout=synthesiser_data["cout"]
	
	print("Args:",synthesiser_data)
	
	
	if output != None:
		file_path = load_folder(output)
	else:
		file_path = None
	##	load folder		##
	
	
	if batch != 0:
		batch_index = 0
		for y in range(amount//batch):
			if cout != False:
				print("Batch: ",y+1)
			synth_func(schema_model,method,batch,file_path,start_index=batch_index,cout=cout)
			batch_index+=batch
		synth_func(schema_model,method,amount-batch_index,file_path,start_index=batch_index,cout=cout)
	else:
		synth_func(schema_model,method,amount,file_path,cout=cout)
	
	
	##	close folder	##
	close_folder(file_path)


@app.command()
def anonymise(
		ingest:str=None,
		method:str=None,
		output:str=None,
		manual:Annotated[typing.Optional[bool], typer.Option("--manual/--no-manual")] = None,
		cout:Annotated[typing.Optional[bool], typer.Option("--cout/--no-cout")] = None,
		config:str=None
	):
	fields=None
	flags = {"ingest":ingest,"method":method,"output":output,"manual":manual,"cout":cout,"config":config,"fields":fields}
	
	anonymiser_data,schema_model = load_flags("a",flags)
	
	ingest = anonymiser_data["ingest"]
	method = anonymiser_data["method"]
	output = anonymiser_data["output"]
	cout = anonymiser_data["cout"]
	manual = anonymiser_data["manual"]
	fields = anonymiser_data["fields"]
	
	print("Args:",anonymiser_data)
	
	
	if output != None:
		file_path = load_folder(output)
	else:
		file_path = None
	##	load folder		##
	
	
	if ingest == None:
		raise("Config 'ingest' required")
	with open(ingest) as dt_file: 
		try:
			data = json.load(dt_file)
		except Exception as e:
			print(e)
			data = {}
	
	anon_func(schema_model,data,method,file_path,cout,manual,fields)
	
	
	##	close folder	##
	close_folder(file_path)
    

if __name__ == "__main__":
    app()
