"""The hand-written stub ``_manyfaced.pyi`` must agree with the native module.

ty does not compare a stub with its runtime module (mypy's stubtest did). This test does the
comparison: the names, the kind of each class member (property, method, static method), the
parameter names and the default values.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from pathlib import Path

import pytest

from manyfaced import _manyfaced

STUB = Path(_manyfaced.__file__).with_name("_manyfaced.pyi")
TREE = ast.parse(STUB.read_text(encoding="utf-8"))
STUB_CLASSES = {n.name: n for n in TREE.body if isinstance(n, ast.ClassDef)}
STUB_FUNCTIONS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
STUB_VARIABLES = {
    n.target.id
    for n in TREE.body
    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
}
# Dunder methods that the stub declares on purpose; other dunders come from PyO3.
CHECKED_DUNDERS = {"__new__", "__call__"}


def _runtime_names(obj: object) -> set[str]:
    return {n for n in vars(obj) if not n.startswith("__") or n in CHECKED_DUNDERS}


def _decorators(fn: ast.FunctionDef) -> set[str]:
    return {d.id for d in fn.decorator_list if isinstance(d, ast.Name)}


def _stub_parameters(fn: ast.FunctionDef) -> list[tuple[str, object]]:
    """(name, default) of each parameter, without ``self``/``cls``. ``inspect.Parameter.empty``
    marks a parameter without a default."""
    args = fn.args
    positional = args.posonlyargs + args.args
    defaults = [inspect.Parameter.empty] * (len(positional) - len(args.defaults))
    defaults += [ast.literal_eval(d) for d in args.defaults]
    params: list[tuple[str, object]] = list(zip((a.arg for a in positional), defaults, strict=True))
    params += [
        (a.arg, inspect.Parameter.empty if d is None else ast.literal_eval(d))
        for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    ]
    if params and params[0][0] in ("self", "cls") and "staticmethod" not in _decorators(fn):
        params = params[1:]
    return params


def _runtime_parameters(fn: Callable[..., object]) -> list[tuple[str, object]] | None:
    """(name, default) of each parameter, or None if the runtime gives no signature (PyO3 gives
    ``__call__`` only ``(*args, **kwargs)``)."""
    sig = inspect.signature(fn)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    if params and all(p.kind in variadic for p in params):
        return None
    return [(p.name, p.default) for p in params]


def test_module_names():
    stub = set(STUB_CLASSES) | set(STUB_FUNCTIONS) | STUB_VARIABLES
    assert stub == _runtime_names(_manyfaced) | {"__version__"}


def test_module_variables():
    for name in STUB_VARIABLES:
        assert isinstance(getattr(_manyfaced, name), str), name


@pytest.mark.parametrize("name", sorted(STUB_FUNCTIONS))
def test_function_signatures(name):
    expected = _stub_parameters(STUB_FUNCTIONS[name])
    assert _runtime_parameters(getattr(_manyfaced, name)) == expected


@pytest.mark.parametrize("name", sorted(STUB_CLASSES))
def test_class_members(name):
    cls = getattr(_manyfaced, name)
    stub = STUB_CLASSES[name]
    if any(isinstance(b, ast.Name) and b.id == "ValueError" for b in stub.bases):
        assert issubclass(cls, ValueError)
    members = {n.name: n for n in stub.body if isinstance(n, ast.FunctionDef)}
    assert set(members) == _runtime_names(cls), name
    for member, fn in members.items():
        runtime = vars(cls)[member]
        decorators = _decorators(fn)
        if "property" in decorators:
            assert inspect.isdatadescriptor(runtime), f"{name}.{member} is not a property"
            continue
        assert ("staticmethod" in decorators) == isinstance(runtime, staticmethod), member
        target = cls if member == "__new__" else getattr(cls, member)
        runtime_params = _runtime_parameters(target)
        if runtime_params is not None:
            assert runtime_params == _stub_parameters(fn), f"{name}.{member}"
