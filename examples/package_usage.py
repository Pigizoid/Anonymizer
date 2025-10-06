import pytest
import time
import psutil
import os
from typing import List, Optional, Union
from pydantic import BaseModel, Field

from smoke_mirrors.synthesiser.synthesiser import JsonSynthesiser
from smoke_mirrors.tools.model_funcs import load_schemas_from_openapi

# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------

def memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def generate_and_validate(synth, schema_or_model, n=1000):
    """Generate N samples and return duration + memory usage."""
    start_mem = memory_usage_mb()
    start = time.time()
    results = synth.synthesise(schema_or_model,amount=n,performance=True)
    duration = time.time() - start
    end_mem = memory_usage_mb()
    return results, duration, (end_mem - start_mem)


# -------------------------------------------------------------------
# 1. Stress Volume Test
# -------------------------------------------------------------------

class StressProfile(BaseModel):
    bio: Optional[str]
    score: int


class StressUser(BaseModel):
    id: int
    name: str
    profile: StressProfile


def test_stress_volume():
    json_synth = JsonSynthesiser(method="mixed",cout=True)
    user_json = StressUser.model_json_schema()

    results, duration, mem = generate_and_validate(json_synth, user_json, n=10000)
    print(f"JSON Synthesiser: 1000 users in {duration:.2f}s, +{mem:.2f} MB","\n")


# -------------------------------------------------------------------
# 2. Constraint Mix Test
# -------------------------------------------------------------------

class ConstraintModel(BaseModel):
    positive_int: int = Field(..., ge=0)
    small_float: float = Field(..., ge=0, le=100)
    short_str: str = Field(..., max_length=5)
    tags: List[str] = Field(..., min_items=1, max_items=3)


def test_constraint_mix():
    json_synth = JsonSynthesiser(method="mixed")
    json_schema = ConstraintModel.model_json_schema()

    val = json_synth.synthesise(json_schema)
    print("Constraint Model JSON:", val,"\n")


# -------------------------------------------------------------------
# 3. Wide Schema Test
# -------------------------------------------------------------------

class WideModel(BaseModel):
    f1: str
    f2: int
    f3: float
    f4: bool
    f5: Optional[str]
    f6: List[int]
    f7: dict
    f8: Union[str, int]
    f9: str = "default"
    f10: List[str] = []


def test_wide_schema():
    json_synth = JsonSynthesiser(method="mixed")
    json_schema = WideModel.model_json_schema()

    val = json_synth.synthesise(json_schema)
    print("Wide Schema JSON:", val,"\n")


# -------------------------------------------------------------------
# 4. Pattern Test
# -------------------------------------------------------------------

class PatternModel(BaseModel):
    zipcode: str = Field(..., pattern=r"^\d{5}(-\d{4})?$")       # US zip or zip+4
    uuid: str = Field(..., pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
    cc: str = Field(..., pattern=r"^4[0-9]{12}(?:[0-9]{3})?$")   # Visa 13 or 16 digits


def test_pattern_torture():
    json_synth = JsonSynthesiser(method="mixed")
    json_schema = PatternModel.model_json_schema()

    val = json_synth.synthesise(json_schema)
    print("Pattern Model JSON:", val,"\n")


openapi_schema = {"openapi":"3.1.0","info":{"title":"Todo API","version":"0.1.0"},"paths":{"/":{"get":{"tags":["Tasks"],"summary":"Get All Tasks","operationId":"get_all_tasks__get","responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{"items":{"$ref":"#/components/schemas/Task"},"type":"array","title":"Response Get All Tasks  Get"}}}}}},"post":{"tags":["Tasks"],"summary":"Create Task","operationId":"create_task__post","requestBody":{"content":{"application/json":{"schema":{"$ref":"#/components/schemas/Task"}}},"required":True},"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{"$ref":"#/components/schemas/Task"}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}},"delete":{"tags":["Tasks"],"summary":"Delete All Tasks","operationId":"delete_all_tasks__delete","responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}}}}},"/{user_id}":{"get":{"tags":["Tasks"],"summary":"Get Tasks By User Id","operationId":"get_tasks_by_user_id__user_id__get","parameters":[{"name":"user_id","in":"path","required":True,"schema":{"type":"string","title":"User Id"}}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{"type":"array","items":{"$ref":"#/components/schemas/Task"},"title":"Response Get Tasks By User Id  User Id  Get"}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/{id}":{"get":{"tags":["Tasks"],"summary":"Get Task","operationId":"get_task__id__get","parameters":[{"name":"id","in":"path","required":True,"schema":{"type":"string","title":"Id"}}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{"$ref":"#/components/schemas/Task"}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}},"patch":{"tags":["Tasks"],"summary":"Update Task","operationId":"update_task__id__patch","parameters":[{"name":"id","in":"path","required":True,"schema":{"type":"string","title":"Id"}}],"requestBody":{"required":True,"content":{"application/json":{"schema":{"$ref":"#/components/schemas/TaskUpdate"}}}},"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{"$ref":"#/components/schemas/Task"}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}},"delete":{"tags":["Tasks"],"summary":"Delete Task","operationId":"delete_task__id__delete","parameters":[{"name":"id","in":"path","required":True,"schema":{"type":"string","title":"Id"}}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/users/{id}":{"delete":{"tags":["user"],"summary":"Delete User","operationId":"delete_user_users__id__delete","parameters":[{"name":"id","in":"path","required":True,"schema":{"type":"string","title":"Id"}}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}}},"components":{"schemas":{"HTTPValidationError":{"properties":{"detail":{"items":{"$ref":"#/components/schemas/ValidationError"},"type":"array","title":"Detail"}},"type":"object","title":"HTTPValidationError"},"PydanticObjectId":{"type":"string","maxLength":24,"minLength":24,"pattern":"^[0-9a-f]{24}$","example":"5eb7cf5a86d9755df3a6c593"},"Task":{"properties":{"_id":{"anyOf":[{"$ref":"#/components/schemas/PydanticObjectId"},{"type":"null"}],"description":"MongoDB document ObjectID"},"title":{"type":"string","title":"Title"},"description":{"type":"string","title":"Description"},"completed":{"type":"boolean","title":"Completed","default":False},"user_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"User Id"},"urgent":{"type":"boolean","title":"Urgent","default":False}},"type":"object","required":["title","description"],"title":"Task"},"TaskUpdate":{"properties":{"title":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Title"},"description":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Description"},"completed":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Completed"},"user_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"User Id"},"urgent":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Urgent"}},"type":"object","title":"TaskUpdate"},"ValidationError":{"properties":{"loc":{"items":{"anyOf":[{"type":"string"},{"type":"integer"}]},"type":"array","title":"Location"},"msg":{"type":"string","title":"Message"},"type":{"type":"string","title":"Error Type"}},"type":"object","required":["loc","msg","type"],"title":"ValidationError"}}}}
# -------------------------------------------------------------------
# 5. Open API Test
# -------------------------------------------------------------------
import json
def test_api_schema():
    schemas = load_schemas_from_openapi(openapi_schema)
    json_synth = JsonSynthesiser(method="mixed")
    for name,schema in schemas.items():
        val = json_synth.synthesise(schema)
        print(f"API Model name '{name}' | JSON output data:", val,"\n")

schema = {"$id": "http://example.com/root.schema.json","$schema": "https://json-schema.org/draft/2020-12/schema","title": "Root Schema Stress Test","description": "A schema with many anchors, ids, and references for stress testing.","type": "object","$defs": {"stringDef": {"$id": "http://example.com/defs/string.schema.json","$anchor": "stringAnchor","type": "string","minLength": 1},"numberDef": {"$id": "http://example.com/defs/number.schema.json","$anchor": "numAnchor","type": "number","minimum": 0},"nested": {"$id": "http://example.com/defs/nested.schema.json","type": "object","properties": {"child": { "$ref": "string.schema.json#stringAnchor" },"value": { "$ref": "number.schema.json#numAnchor" }},"required": ["child", "value"]}},"properties": {"name": { "$ref": "#/$defs/stringDef" },"age": { "$ref": "http://example.com/defs/number.schema.json" },"profile": { "$ref": "#/$defs/nested" },"aliases": {"type": "array","items": { "$ref": "http://example.com/defs/string.schema.json#stringAnchor" }},"meta": {"$id": "http://example.com/props/meta.schema.json","type": "object","properties": {"createdAt": { "type": "string", "format": "date-time" },"tags": {"type": "array","items": { "$ref": "/defs/string.schema.json" }}}}},"required": ["name", "age", "profile"]}
# -------------------------------------------------------------------
# 6. URI Model Test
# -------------------------------------------------------------------
def test_URI_schema():
    
    synth = JsonSynthesiser(method="mixed")
    val = synth.synthesise(schema)
    print(f"URI Model | JSON output data:", val,"\n")
    
# -------------------------------------------------------------------
# Run directly without pytest
# -------------------------------------------------------------------

if __name__ == "__main__":
    test_stress_volume()
    test_constraint_mix()
    test_wide_schema()
    test_pattern_torture()
    test_api_schema()
    test_URI_schema()
