"""Tests for the import-closure helper in tools/burn_divergence_wire_identity.py.

Each case builds a tiny fake package in a temporary folder and checks which
files the closure reaches. No simulation code runs here.
"""
from pathlib import Path

from tools.burn_divergence_wire_identity import (
    chain_to,
    command_scripts,
    import_closure,
)


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _closure(root: Path, entries: list[str]) -> set[str]:
    reached, errors = import_closure(root, entries)
    assert errors == []
    return set(reached)


def test_absolute_import_pulls_in_parent_packages(tmp_path):
    _write(tmp_path, {
        "run.py": "import pkg.sub.leaf\n",
        "pkg/__init__.py": "",
        "pkg/sub/__init__.py": "",
        "pkg/sub/leaf.py": "",
        "pkg/sub/unused.py": "",
    })
    assert _closure(tmp_path, ["run.py"]) == {
        "run.py", "pkg/__init__.py", "pkg/sub/__init__.py", "pkg/sub/leaf.py"}


def test_from_import_name_may_be_a_submodule_or_a_plain_name(tmp_path):
    _write(tmp_path, {
        "run.py": "from pkg import leaf\nfrom pkg.other import helper\n",
        "pkg/__init__.py": "",
        "pkg/leaf.py": "",
        "pkg/other.py": "def helper():\n    pass\n",
    })
    # "leaf" is a submodule so it counts; "helper" is a function in other.py.
    assert _closure(tmp_path, ["run.py"]) == {
        "run.py", "pkg/__init__.py", "pkg/leaf.py", "pkg/other.py"}


def test_relative_imports_resolve_from_the_importing_package(tmp_path):
    _write(tmp_path, {
        "run.py": "import pkg.a\n",
        "pkg/__init__.py": "",
        "pkg/a.py": "from . import b\nfrom .c import thing\n",
        "pkg/b.py": "from .. import top\n",
        "pkg/c.py": "thing = 1\n",
        "pkg/d.py": "",
        "top.py": "",
    })
    assert _closure(tmp_path, ["run.py"]) == {
        "run.py", "pkg/__init__.py", "pkg/a.py", "pkg/b.py", "pkg/c.py", "top.py"}


def test_imports_inside_functions_count(tmp_path):
    _write(tmp_path, {
        "run.py": "def go():\n    from pkg.late import value\n    return value\n",
        "pkg/__init__.py": "",
        "pkg/late.py": "value = 1\n",
    })
    assert "pkg/late.py" in _closure(tmp_path, ["run.py"])


def test_outside_modules_are_ignored(tmp_path):
    _write(tmp_path, {"run.py": "import json\nimport numpy as np\nfrom os import path\n"})
    assert _closure(tmp_path, ["run.py"]) == {"run.py"}


def test_module_function_strings_and_import_module_count(tmp_path):
    _write(tmp_path, {
        "run.py": ("import importlib\n"
                   "NAME = 'pkg.worker:job'\n"
                   "importlib.import_module('pkg.dynamic')\n"),
        "pkg/__init__.py": "",
        "pkg/worker.py": "",
        "pkg/dynamic.py": "",
    })
    assert _closure(tmp_path, ["run.py"]) == {
        "run.py", "pkg/__init__.py", "pkg/worker.py", "pkg/dynamic.py"}


def test_script_folder_is_on_the_search_path(tmp_path):
    # Running tools/run.py puts tools/ on sys.path, so "import helper" finds tools/helper.py.
    _write(tmp_path, {
        "tools/run.py": "import helper\n",
        "tools/helper.py": "",
    })
    assert "tools/helper.py" in _closure(tmp_path, ["tools/run.py"])


def test_missing_entry_and_syntax_error_are_reported(tmp_path):
    _write(tmp_path, {"bad.py": "def (:\n"})
    _, errors = import_closure(tmp_path, ["bad.py", "gone.py"])
    assert any("gone.py" in e for e in errors)
    assert any("bad.py" in e for e in errors)


def test_chain_follows_the_first_importer(tmp_path):
    _write(tmp_path, {
        "run.py": "import pkg.mid\n",
        "pkg/__init__.py": "",
        "pkg/mid.py": "import pkg.leaf\n",
        "pkg/leaf.py": "",
    })
    reached, _ = import_closure(tmp_path, ["run.py"])
    chain = chain_to(reached, "pkg/leaf.py")
    assert chain[0] == "run.py" and chain[-1] == "pkg/leaf.py"
    assert "pkg/mid.py" in chain


def test_command_scripts_finds_every_py_token():
    command = ("PYTHONPATH=. {python} tools/a.py --output {out} && "
               "PYTHONPATH=. {python} tools/b.py {out}/summary.json")
    assert command_scripts(command) == ["tools/a.py", "tools/b.py"]
