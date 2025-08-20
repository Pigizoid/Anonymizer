import csv
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
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


# database: List[schema_model] = []  #local database list


db_file = "database.csv"
temp_file = "temp_database.csv"


def append_row(index, *fields):  # append new version to the csv
    with open(db_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([index] + list(fields))


def get_model_keys(schema_model):
    return [name for name, values in schema_model.model_fields.items()]


def read_latest_ids():
    latest_ids = []
    deleted_ids = {}
    with open(db_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            idx = int(row[0])
            if row[1] == "__DELETED__":
                deleted_ids[idx] = True
            else:
                if idx in deleted_ids:
                    deleted_ids[idx] = False
                latest_ids.append(idx)
    latest_ids = [
        id_num
        for id_num in set(latest_ids)
        if (id_num not in deleted_ids) or (deleted_ids[id_num] is not True)
    ]
    return latest_ids


def read_latest():  # read the latest csv version
    latest = {}
    deleted_ids = {}
    keys = get_model_keys(schema_model)
    with open(db_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            idx = int(row[0])
            if row[1] == "__DELETED__":
                deleted_ids[idx] = True
            else:
                if idx in deleted_ids:
                    deleted_ids[idx] = False
                latest[idx] = {key: val for key, val in zip(keys, row[1:])}
    latest = {
        key: val
        for key, val in latest.items()
        if (key not in deleted_ids) or (deleted_ids[key] is not True)
    }
    return latest


def read_latest_byindex(id_num):
    latest = {}
    deleted = False
    keys = get_model_keys(schema_model)
    with open(db_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            idx = int(row[0])
            if idx == id_num:
                if row[1] == "__DELETED__":
                    deleted = True
                else:
                    if deleted:
                        deleted = False
                    latest = {key: val for key, val in zip(keys, row[1:])}
    if deleted:
        return None
    else:
        return latest


def read_all():
    logs = {}
    keys = get_model_keys(schema_model)
    log_id = 0
    with open(db_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[1] == "__DELETED__":
                logs[log_id] = {row[0]: row[1]}
            else:
                logs[log_id] = {row[0]: {key: val for key, val in zip(keys, row[1:])}}
            log_id += 1
    return logs


def read_snapshot(log_id: int):
    latest = {}
    deleted_ids = {}
    keys = get_model_keys(schema_model)
    current_log_id = 0
    with open(db_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            idx = int(row[0])
            if row[1] == "__DELETED__":
                deleted_ids[idx] = True
            else:
                if idx in deleted_ids:
                    deleted_ids[idx] = False
                latest[idx] = {key: val for key, val in zip(keys, row[1:])}
            current_log_id += 1
            if current_log_id > log_id:
                break
    latest = {
        key: val
        for key, val in latest.items()
        if (key not in deleted_ids) or (deleted_ids[key] is not True)
    }
    return latest


def refactor():  # rewrite file with latest updates
    global current_max_id
    global current_ids
    latest = read_latest()
    sorted_indexes = sorted(latest.keys())
    with open(
        temp_file, "w", newline=""
    ) as f:  # temp built to avoid read issues (w will clear upon open)
        writer = csv.writer(f)
        for idx in sorted_indexes:
            writer.writerow([str(idx)] + list(latest[idx].values()))
    current_ids = read_latest_ids()
    current_max_id = current_ids[-1]
    os.replace(temp_file, db_file)


def rollback(log_id):
    global current_max_id
    global current_ids
    logs = {}
    current_log_id = 0
    with open(db_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            logs[current_log_id] = row
            current_log_id += 1
            if current_log_id > log_id:
                break

    with open(temp_file, "w", newline="") as f:
        writer = csv.writer(f)
        for row_index, row_data in logs.items():
            writer.writerow(row_data)
    current_ids = read_latest_ids()
    current_max_id = current_ids[-1]
    os.replace(temp_file, db_file)


def get_entry_by_id(id_num: int):
    latest_ids = read_latest_ids()
    if id_num not in latest_ids:
        return None
    else:
        return_data = read_latest_byindex(id_num)
        return schema_model(**return_data)


current_max_id = 0
current_ids = []


def init_max_id():
    global current_max_id
    global current_ids
    with open(db_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[1] != "__DELETED__":
                idx = int(row[0])
                if idx > current_max_id:
                    current_max_id = idx
    current_ids = read_latest_ids()


# run once at startup
init_max_id()


def get_next_id():
    global current_max_id
    current_max_id += 1
    return current_max_id


@app.get(
    "/database/get", response_model=schema_model, responses={404: {"model": Message}}
)
def entry_get(id_num: int):
    # return database[id_num]
    if get_entry_by_id(id_num) is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    return schema_model(**read_latest_byindex(id_num))


@app.get("/database", response_model=Dict[int, schema_model])
def database_get():
    json_db = read_latest()
    database_return = {
        idx: schema_model(**db_entry) for idx, db_entry in json_db.items()
    }
    return database_return


@app.get("/database_ids", response_model=List[int])
def database_ids_get():
    return read_latest_ids()


@app.post(
    "/database/add",
    response_model=ReturnEntryMessage,
    responses={404: {"model": Message}},
)
def entry_add(data: schema_model, id_num: str = None):
    if id_num is None or id_num == "":
        id_num = get_next_id()
    else:
        id_num = int(id_num)
    if id_num in current_ids:
        raise HTTPException(status_code=400, detail="Entry with this ID already exists")
    else:
        current_ids.append(id_num)

    append_row(id_num, *data.model_dump().values())

    return {"message": f"Entry {id_num} added", "entry": data}


@app.post("/database/add_batch", response_model=Message)
def add_batch(data: List[schema_model]):
    start_id = get_next_id()
    batch_rows = []
    for i, entry in enumerate(data):
        id_num = start_id + i
        batch_rows.append([id_num] + list(entry.model_dump().values()))
        current_ids.append(id_num)

    # write all at once
    with open(db_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(batch_rows)

    # update max ID
    global current_max_id
    current_max_id += len(data)

    return {"message": f"{len(data)} entries added successfully."}


@app.put(
    "/database/update",
    response_model=ReturnEntryMessage,
    responses={404: {"model": Message}},
)
def entry_update(data: schema_model, id_num: int):
    if get_entry_by_id(id_num) is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    append_row(id_num, *data.model_dump().values())
    return {"message": "Entry updated", "entry": data}


@app.delete(
    "/database/delete",
    response_model=ReturnEntryMessage,
    responses={404: {"model": Message}},
)
def entry_delete(id_num: int):
    entry = get_entry_by_id(id_num)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    append_row(id_num, "__DELETED__")
    try:
        current_ids.remove(id_num)
    except:
        raise HTTPException(status_code=404, detail="Entry ID not found")
    return {"message": "Entry deleted", "entry": entry}


@app.post("/database/refresh", response_model=Message)
def refresh_database():
    try:
        refactor()
    except:
        return {"message": "Refresh not successful"}
    return {"message": "Refresh successful"}


@app.get("/database/log", response_model=Dict[int, Dict[int, str]])
def get_database_log():
    json_logs = read_all()
    database_return = {
        idx: {idy: str(rest) for idy, rest in inner.items()}
        for idx, inner in json_logs.items()
    }
    return database_return


@app.get("/database/logs", response_model=Dict[int, schema_model])
def get_database_at_snapshot(log_id):
    json_db = read_snapshot(int(log_id))
    database_return = {
        idx: schema_model(**db_entry) for idx, db_entry in json_db.items()
    }
    return database_return


@app.post("/database/rollback", response_model=Dict[int, schema_model])
def rolback_database_at_snapshot(log_id: int):
    try:
        rollback(log_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Rollback not successful: {e}")
    json_db = read_latest()
    database_return = {
        idx: schema_model(**db_entry) for idx, db_entry in json_db.items()
    }
    return database_return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
