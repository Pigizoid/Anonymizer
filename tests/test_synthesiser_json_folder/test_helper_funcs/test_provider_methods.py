from src.smoke_mirrors.synthesiser_json.helper_funcs.provider_methods import list_faker_methods, list_mimesis_methods, list_match_methods, make_resolved_methods

from faker import Faker


def test_list_faker_methods():
    return_data = list_faker_methods()
    assert isinstance(return_data, tuple)
    assert isinstance(return_data[0], list)
    assert isinstance(return_data[1], dict)
    assert all([isinstance(method, str) for method in return_data[0]])
    assert all([method in return_data[1] for method in return_data[0]])


def test_list_mimesis_methods():
    return_data = list_mimesis_methods()
    assert isinstance(return_data, tuple)
    assert isinstance(return_data[0], list)
    assert isinstance(return_data[1], dict)
    assert all([isinstance(method, str) for method in return_data[0]])
    assert all([method in return_data[1] for method in return_data[0]])


def test_list_match_methods():
    return_data = list_match_methods("mixed")
    assert isinstance(return_data, tuple)
    assert isinstance(return_data[0], list)
    assert isinstance(return_data[1], dict)
    assert all([isinstance(method, str) for method in return_data[0]])
    assert all([method in return_data[1] for method in return_data[0]])

    faker_list = list_faker_methods()[0]
    mimesis_list = list_mimesis_methods()[0]
    faker_list.extend(mimesis_list)
    match_list = set(faker_list)
    assert all([match in return_data[0] for match in match_list])


fake = Faker()
def test_make_resolved_methods():
    return_value = make_resolved_methods(["name","email"],{"name":fake,"email":fake})
    assert isinstance(return_value,dict)
    assert all([callable(x) for x in return_value.values()])