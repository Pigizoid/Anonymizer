import pytest
from src.sm.synthesiser.synthesiser import Synthesiser

methods = ["faker","mimesis","mixed"]

@pytest.mark.parametrize("method",[(method) for method in methods])
def test__init__(method):
    synth = Synthesiser(method=method)
    assert synth.outputpooling is not None
    assert synth.word_list is not None
    assert synth.word_tokens is not None
    assert synth.word_tokens_set is not None
    assert synth.resolved_methods is not None

    assert synth.outputpooling == {}
    assert all([isinstance(word,str) for word in synth.word_list])
    assert isinstance(synth.word_tokens,dict)
    assert all([isinstance(token,list) for token in synth.word_tokens.values()])
    assert isinstance(synth.word_tokens_set,dict)
    assert all([isinstance(token,set) for token in synth.word_tokens_set.values()])
    assert all([callable(method) for method in synth.resolved_methods.values()])
    for word in synth.word_list:
        assert word in synth.word_tokens.keys()
        assert word in synth.word_tokens_set.keys()
