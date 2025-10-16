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

or optionally
uv run --with pyinstaller pyinstaller --onefile src/smoke_mirrors/app/main.py --name sm


# System Components

## Packages:
```
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
```


# Usage
## CLI usage

### Folder Structure
Open selected directory in the terminal

|folder
|  \
|  |models
|  |data
|  |config

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

```
with exe exchange 'sm' for '.\sm'
```

## Config usage
```yaml

schema_path: schemas/user_schema.json  # Path to a schema file
schema_type: json  # The type of schemas to include (e.g. json, py) (defaults to both)
seed: 42  # Random seed

# Configuration for the synthetic data generation process
synth:
  method: mixed                     # 'faker', 'mimesis', 'mixed' 
  amount: 10                        # Number of records to generate
  batch: 2                          # Batch size, 0 means no batching
  output: data/synth_output.json    # Output file or directory
  flat_output: true                 # Whether to flatten nested output structures
  stdcout: false                       # Print extra output infornation to console
  performance: false                # Turn on quick validation (after the first 10, validation is skipped)

# Configuration for data anonymization
anon:
  ingest: data/raw_data.json        # Path to the input data
  method: mask                      # 'mask', 'synth', 'perturb'
  amount: 1                         # How many anonymized versions to create
  output: data/anonymized.json      # Output file or directory
  stdcout: false                       # Print extra output infornation to console
  manual: false                     # If true, requires manual field decisions
  default: mask                     # Default anonymization method for unspecified fields
  key_anon: true                    # Whether to anonymize keys as well
  performance: false                # Turn on quick validation (after the first 10, validation is skipped)
  fields:                           # Field-specific anonymization strategies
    name: mask
    age: perturb
    email: synth
```

## Package Usage
```py
from smoke_mirrors.synthesiser.synthesiser import JsonSynthesiser
from smoke_mirrors.anonymiser.anonymier import anonymise

# methods ["faker","mimesis",mixed"]
json_synth = JsonSynthesiser(method=...,amount=...) 
data = json_synth.synthesise(json_schema)

val = anonymise(json_schema,data)
```
