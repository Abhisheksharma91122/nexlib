from importlib.resources import files
from pathlib import Path


def _get_code(number):
    if number not in range(1, 9):
        raise ValueError("Enter a number between 1 and 8")

    return files("nexlib.codes").joinpath(
        f"practical{number}.py"
    ).read_text(encoding="utf-8")


def show(number):
    code = _get_code(number)

    try:
        from IPython.display import display, Code
        display(Code(code, language="python"))
    except ImportError:
        print(code)


def download(number):
    code = _get_code(number)
    filename = f"practical{number}.py"

    Path(filename).write_text(code, encoding="utf-8")
    print(f"Downloaded {filename}")


def download_all():
    for number in range(1, 9):
        download(number)