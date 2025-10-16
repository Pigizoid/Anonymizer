import os
import shutil
from pathlib import Path


def clear_folder(folder: str):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.mkdir(folder)


def pytest_sessionstart(session):
    print("\n>>> Starting tests and clearing outputs folder")
    test_output = Path("tests") / "outputs"
    clear_folder(test_output)
