from pydantic import BaseModel
from typing import Optional


class StressProfile(BaseModel):
    bio: Optional[str]
    score: int


class StressUser(BaseModel):
    id: int
    name: str
    profile: StressProfile
