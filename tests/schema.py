from pydantic import BaseModel

class test_user(BaseModel):
    name: str
    email: str
    age: int
