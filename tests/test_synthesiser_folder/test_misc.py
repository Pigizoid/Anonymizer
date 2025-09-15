import pytest

from src.sm.synthesiser.misc import print_path


def test_print_path_simple(capsys):
    path = "[0]"
    elapsed_time = 1.2345
    print_path(path, elapsed_time)

    captured = capsys.readouterr()
    output = captured.out.strip()

    assert "Time taken: 1.23 seconds" in output
    assert path in output


def test_print_path_deeper_path(capsys):
    path = "[0][1][2]"
    elapsed_time = 12.5
    print_path(path, elapsed_time)

    captured = capsys.readouterr()
    output = captured.out

    assert "Time taken: 12.50 seconds" in output
    assert path in output
    assert "        " in output


@pytest.mark.parametrize(
    "path,elapsed,expected",
    [
        ("[3]", 0.0, "Time taken: 0.00 seconds"),
        ("[1][2]", 2.718, "Time taken: 2.72 seconds"),
        ("[9][9][9][9]", 100.1234, "Time taken: 100.12 seconds"),
    ],
)
def test_print_path_parametrized(path, elapsed, expected, capsys):
    print_path(path, elapsed)
    captured = capsys.readouterr()
    output = captured.out

    assert expected in output
    assert path in output
