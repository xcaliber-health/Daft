from __future__ import annotations

import inspect

from daft.runners.remote_runner import RemoteRunner
from daft.runners.runner import Runner


def test_remote_runner_implements_every_abstract_method() -> None:
    abstract = {name for name, member in inspect.getmembers(Runner) if getattr(member, "__isabstractmethod__", False)}
    assert abstract, "expected the runner interface to declare abstract methods"
    for name in abstract:
        implementation = getattr(RemoteRunner, name, None)
        assert implementation is not None, f"missing {name}"
        assert not getattr(implementation, "__isabstractmethod__", False), f"{name} not implemented"


def test_remote_runner_declares_its_name() -> None:
    assert RemoteRunner.name == "remote"
