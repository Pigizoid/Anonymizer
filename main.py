
import typer

from functions.synthesiser import Synthesiser
import schema

from functions.anonymiser import Anonymiser


import yaml
import json


app = typer.Typer()

def synth_func(schema_model,method,amount,file_path,start_index=0,cout:str="true"):
	dataset = Synthesiser.synthesise(schema_model,method,amount)  #returns as [ data, data, ... ]
	flush = []
	for index,data in enumerate(dataset):
		if file_path != None:
			output_data = {key: str(value) for key, value in dict(data).items()}
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
			if cout == "true":
				print(flush_out)
			flush.clear()
	if flush != []:
		if file_path != None:
			flush_out = "".join(flush)
			with open(f"{file_path}.json","a") as f:
				f.write(flush_out)
		else:
			flush_out = "".join(flush)
		if cout == "true":
			print(flush_out)
		flush.clear()
	if file_path != None:
		print(f"To file_path -> {file_path}")


@app.command()
def synthesise(method:str = "faker",amount:int=1,file:str=None,batch:int=0,cout:str="true"):
	#methods = "faker","mimesis","mixed"
	if file != None:
		import os
		dir_name = "outputs"
		file_path = f"{dir_name}\\{file}"
		if os.path.isdir(dir_name) == False:
			try:
				os.mkdir(dir_name)
				print(f"Directory '{dir_name}' created successfully")
				with open(f"{file_path}.json","w") as f:
					f.write("[")
			except PermissionError:
				print(f"Permission denied: Unable to create '{dir_name}'")
			except Exception as e:
				print(f"An error occurred: {e}")
		else:
			with open(f"{file_path}.json","w") as f:
				f.write("{")
	
	if file == None:
		file_path = None
	
	schema_model = schema.Address
	if batch != 0:
		batch_index = 0
		for y in range(amount//batch):
			print("Batch: ",y+1)
			synth_func(schema_model,method,batch,file_path,start_index=batch_index,cout=cout)
			batch_index+=batch
		synth_func(schema_model,method,amount-batch_index,file_path,start_index=batch_index,cout=cout)
	else:
		synth_func(schema_model,method,amount,file_path,cout=cout)
	
	
	if file != None:
		with open(f"{file_path}.json","a") as f:
			f.write("\n}\n")
	
	
	#with open(f"{file_path}.json","r") as f:
	#	print(json.load(f))


@app.command()
def anonymise(data:str, method:str = "faker",config: str = None):
	
	with open(data) as dt_file: 
		try:
			data = json.load(dt_file)
		except Exception as e:
			print(e)
			data = {}
	
	if config != None:
		
		with open(config) as cf_file:
			try:
				#print(yaml.safe_load(cf_file))
				config = dict(yaml.safe_load(cf_file))
			except yaml.YAMLError as exc:
				print(exc)
	
	schema_model = schema.Address
	data = Anonymiser.anonymise(schema_model,method,data,config)
    

if __name__ == "__main__":
    app()
