"""The native module's type stub describes what the module holds at runtime.

Every class and function the stub declares exists in the module. A class is
declared an ``Enum`` exactly when it is one at runtime, and an enumeration
declares exactly the members it has, so code a type checker accepts against
the stub runs.
"""

from __future__ import annotations

import ast
import enum
from pathlib import Path

import pytest

import daft
import daft.daft as native

STUB = ast.parse((Path(daft.__file__).parent / "daft" / "__init__.pyi").read_text())
CLASSES = {node.name: node for node in STUB.body if isinstance(node, ast.ClassDef)}
FUNCTIONS = [node.name for node in STUB.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _members(kind: type) -> set[str]:
    """The members of a native enumeration: its public attributes of its own type."""
    return {name for name in dir(kind) if not name.startswith("_") and isinstance(getattr(kind, name), kind)}


def _declared_members(node: ast.ClassDef) -> set[str]:
    """The members the stub declares: class attributes typed as the class itself."""
    return {
        statement.target.id
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and ast.unparse(statement.annotation) == f"ClassVar[{node.name}]"
    }


def _declared_an_enum(node: ast.ClassDef) -> bool:
    return any(ast.unparse(base) in ("Enum", "enum.Enum") for base in node.bases)


ENUMERATIONS = sorted(name for name in CLASSES if hasattr(native, name) and _members(getattr(native, name)))


@pytest.mark.parametrize("name", sorted([*CLASSES, *FUNCTIONS]))
def test_every_name_the_stub_declares_exists(name: str) -> None:
    assert hasattr(native, name)


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_a_class_is_declared_an_enum_exactly_when_it_is_one(name: str) -> None:
    runtime = getattr(native, name)

    assert _declared_an_enum(CLASSES[name]) == issubclass(runtime, enum.Enum)


@pytest.mark.parametrize("name", ENUMERATIONS)
def test_an_enumeration_declares_exactly_its_members(name: str) -> None:
    assert _declared_members(CLASSES[name]) == _members(getattr(native, name))
