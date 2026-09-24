from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# The top-level module of each optional extra. Importing Daft must neither need
# them nor load them: they are imported only when the feature using them runs.
OPTIONAL_MODULES = [
    "pyiceberg",
    "deltalake",
    "lance",
    "ray",
    "pandas",
    "torch",
    "sqlalchemy",
    "connectorx",
    "huggingface_hub",
    "boto3",
    "openai",
    "transformers",
    "unitycatalog",
    "clickhouse_connect",
]


def _run(script: str) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a fresh interpreter, so no earlier import can mask the result."""
    return subprocess.run([sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True)


def test_importing_daft_loads_no_optional_dependency() -> None:
    result = _run(
        f"""
        import sys
        import daft
        loaded = sorted({{m.split(".")[0] for m in sys.modules}} & set({OPTIONAL_MODULES!r}))
        print(",".join(loaded))
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


@pytest.mark.parametrize("module", OPTIONAL_MODULES)
def test_daft_imports_and_runs_without_an_optional_dependency(module: str) -> None:
    result = _run(
        f"""
        import importlib.abc
        import sys

        class Missing(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == {module!r} or name.startswith({module!r} + "."):
                    raise ModuleNotFoundError(f"No module named {{name!r}}")

        sys.meta_path.insert(0, Missing())
        import daft
        import daft.catalog
        import daft.functions
        import daft.io

        answer = daft.from_pydict({{"a": [1, 2, 3]}}).where(daft.col("a") > 1).to_pydict()
        assert answer == {{"a": [2, 3]}}, answer
        """
    )

    assert result.returncode == 0, result.stderr
