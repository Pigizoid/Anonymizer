from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from scalar_fastapi import get_scalar_api_reference

app = FastAPI()


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )


class Address(BaseModel):
    street: str
    city: str
    zip_code: str = Field(..., pattern=r"^\d{5}(-\d{4})?$")
    country: str = Field(default="USA")
    social_security_number: str
    continent: str


schema_model = Address


class ReturnEntryMessage(BaseModel):
    message: str
    entry: schema_model


class Message(BaseModel):
    message: str


database: List[schema_model] = []


def get_entry_by_id(entry_id: int):
    for index, entry in enumerate(database):
        if entry.id == entry_id:
            return index
    return None


@app.get(
    "/database/{id}", response_model=schema_model, responses={404: {"model": Message}}
)
def entry_get(id: int):
    index = get_entry_by_id(id)
    if index is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    return database[index]


@app.get("/database", response_model=List[schema_model])
def database_get():
    return database


@app.post(
    "/database/add",
    response_model=ReturnEntryMessage,
    responses={404: {"model": Message}},
)
def entry_add(data: schema_model):
    if get_entry_by_id(data.id) is not None:
        raise HTTPException(status_code=400, detail="Entry with this ID already exists")
    database.append(data)
    return {"message": "Entry added", "entry": data}


@app.put(
    "/database/update",
    response_model=ReturnEntryMessage,
    responses={404: {"model": Message}},
)
def entry_update(data: schema_model):
    index = get_entry_by_id(data.id)
    if index is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    database[index] = data
    return {"message": "Entry updated", "entry": data}


@app.delete(
    "/database/delete",
    response_model=ReturnEntryMessage,
    responses={404: {"model": Message}},
)
def entry_delete(id: int):
    index = get_entry_by_id(id)
    if index is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    removed_entry = database.pop(index)
    return {"message": "Entry deleted", "entry": removed_entry}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
