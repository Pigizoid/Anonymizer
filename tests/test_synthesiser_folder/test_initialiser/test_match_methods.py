
from src.sm.synthesiser.synthesiser import Synthesiser
synth = Synthesiser()

def test_list_faker_methods():
        return_data = synth.list_faker_methods()
        assert isinstance(return_data,tuple)
        assert isinstance(return_data[0],list)
        assert isinstance(return_data[1],dict)
        assert all([isinstance(method,str) for method in return_data[0]])
        assert all([method in return_data[1] for method in return_data[0]])

def test_list_mimesis_methods():
        return_data = synth.list_mimesis_methods()
        assert isinstance(return_data,tuple)
        assert isinstance(return_data[0],list)
        assert isinstance(return_data[1],dict)
        assert all([isinstance(method,str) for method in return_data[0]])
        assert all([method in return_data[1] for method in return_data[0]])

def test_list_match_methods():
        return_data = synth.list_match_methods("mixed")
        assert isinstance(return_data,tuple)
        assert isinstance(return_data[0],list)
        assert isinstance(return_data[1],dict)
        assert all([isinstance(method,str) for method in return_data[0]])
        assert all([method in return_data[1] for method in return_data[0]])

        faker_list = synth.list_faker_methods()[0]
        mimesis_list = synth.list_mimesis_methods()[0]
        faker_list.extend(mimesis_list)
        match_list = set(faker_list)
        assert all([match in return_data[0] for match in match_list])