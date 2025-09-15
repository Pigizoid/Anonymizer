from pydantic import BaseModel
from src.sm.synthesiser.synthesiser import Synthesiser


class User(BaseModel):
    id: int
    name: str


synth = Synthesiser()


def test_synthesise_empty():
    result = synth.synthesise(User, amount=0)
    assert result == []


def test_synthesise_amount():
    result = synth.synthesise(User, amount=3)
    assert len(result) == 3
    assert all([isinstance(x, User) for x in result])


"""
def test_synthesise_seed():
    result1 = synth.synthesise(User, amount=2, seed=0)
    result2 = synth.synthesise(User, amount=2, seed=0)
    assert result1 == result2
"""


def test_synthesise_recursive_dict():
    data = synth.synthesise_recursive(User, amount=1)
    assert isinstance(data, dict)
    assert "id" in data
    assert "name" in data


def test_progress_prints(capsys):
    synth.synthesise(User, amount=5)
    captured = capsys.readouterr()
    assert "Completed:" in captured.out
