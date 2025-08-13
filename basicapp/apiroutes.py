from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from scalar_fastapi import get_scalar_api_reference

app = FastAPI()


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )




class User(BaseModel):
	id: int
	name: str
	age: int


class ReturnUserMessage(BaseModel):
	message: str
	user: User

class Message(BaseModel):
	message: str


users: List[User] = []



def get_user_by_id(user_id: int):
	for index, user in enumerate(users):
		if user.id == user_id:
			return index
	return -1


@app.get('/users/{id}',response_model=User, responses={404: {"model": Message}})
def user_get(id: int):
	index = get_user_by_id(id)
	if index == -1:
		raise HTTPException(status_code=404, detail="User not found")
	return users[index]

@app.get('/users',response_model=List[User])
def user_get():
	return users

@app.post('/users/add',response_model=ReturnUserMessage, responses={404: {"model": Message}})
def user_add(data: User):
	if get_user_by_id(data.id) != -1:
		raise HTTPException(status_code=400, detail="User with this ID already exists")
	users.append(data)
	return {"message": "User added", "user": data}

@app.put('/users/update',response_model=ReturnUserMessage, responses={404: {"model": Message}})
def user_update(data: User):
	index = get_user_by_id(data.id)
	if index == -1:
		raise HTTPException(status_code=404, detail="User not found")
	users[index] = data
	return {"message": "User updated", "user": data}

@app.delete('/users/delete',response_model=ReturnUserMessage, responses={404: {"model": Message}})
def user_delete(id: int):
	index = get_user_by_id(id)
	if index == -1:
		raise HTTPException(status_code=404, detail="User not found")
	removed_user = users.pop(index)
	return {"message": "User deleted", "user": removed_user}




if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="127.0.0.1", port=8000)
















