# Smoke and Mirrors

## Purpose
Ingest any data from a schema and anonymize it or create synthetic data

## Scope
TODO

## Features
TODO

## Environment and Installation
Python v3.12+ <br/>

`C:\...\Smoke-and-Mirrors\`

uv build
uv sync

uv pip install smoke-mirrors
uv pip install -e . --system

uv run pytest



# System Components

## Packages:
exrex - 0.12.0
faker - 37.5.3
jsonschema - 4.25.1
mimesis - 18.0.0
pydantic - 2.11.7
pydantic-settings - 2.10.1
pyyam - l6.0.2
requests - 2.32.5
rstr - 3.2.2
typer - 0.16.0


# Usage
## CLI usage
Open selected directory in the terminal
```
Usage: sm [OPTIONS] COMMAND [ARGS]

╭─ Options
│ --config
│ --schema-path
│ --schema-type
│ --seed

╭─ Commands
│ synth
│ anon
│ extras
```
```
Usage: sm synth [OPTIONS] COMMAND [ARGS]
╭─ Commands 
│ batch
│ single
```
```
Usage: sm anon [OPTIONS] COMMAND [ARGS]
╭─ Commands 
│ auto
│ manual
```

## Package Usage
```py
from smoke_mirrors.synthesiser.synthesiser import JsonSynthesiser
from smoke_mirrors.anonymiser.anonymier import anonymise

# methods ["faker","mimesis",mixed"]
json_synth = JsonSynthesiser(method=...,amount=...) 
val = json_synth.synthesise(json_schema)

val = anonymise()

```
