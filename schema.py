


from pydantic import BaseModel, Field, EmailStr, HttpUrl, conint
from typing import List, Optional
from datetime import datetime



class Address(BaseModel):
	street: str
	city: str
	zip_code: str = Field(..., pattern=r'^\d{5}(-\d{4})?$')
	country: str = Field(default="USA")
	social_security_number: str
	continent: str
