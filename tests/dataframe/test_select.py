from __future__ import annotations

from collections.abc import Callable

import pytest

import daft
from daft import col


def test_select_dataframe_missing_col(make_df, valid_data: list[dict[str, float]]) -> None:
    df = make_df(valid_data)

    with pytest.raises(ValueError):
        df = df.select("foo", "sepal_length")


def test_select_dataframe(make_df, valid_data: list[dict[str, float]]) -> None:
    df = make_df(valid_data)
    df = df.select("sepal_length", "sepal_width")
    assert df.column_names == ["sepal_length", "sepal_width"]


def test_multiple_select_same_col(make_df, valid_data: list[dict[str, float]]):
    df = make_df(valid_data)
    df = df.select(df["sepal_length"], df["sepal_length"].alias("sepal_length_2"))
    pdf = df.to_pandas()
    assert len(pdf.columns) == 2
    assert pdf.columns.to_list() == ["sepal_length", "sepal_length_2"]


def test_select_ordering(make_df, valid_data: list[dict[str, float]]):
    df = make_df(valid_data)
    df = df.select(
        df["variety"], df["petal_length"].alias("foo"), df["sepal_length"], df["sepal_width"], df["petal_width"]
    )
    df = df.collect()
    assert df.column_names == ["variety", "foo", "sepal_length", "sepal_width", "petal_width"]


def test_select_with_kwargs_only(make_df, valid_data: list[dict[str, float]]):
    """Test that select works with keyword arguments only."""
    df = make_df(valid_data)
    df = df.select(new_col=col("sepal_length"), another_col=col("sepal_width"))
    assert df.column_names == ["new_col", "another_col"]


def test_select_with_variadic_and_kwargs(make_df, valid_data: list[dict[str, float]]):
    """Test that select works with both variadic arguments and keyword arguments."""
    df = make_df(valid_data)
    df = df.select("sepal_length", "sepal_width", new_col=col("petal_length"), another_col=col("petal_width"))
    # Variadic arguments come first, then keyword arguments
    assert df.column_names == ["sepal_length", "sepal_width", "new_col", "another_col"]


def test_select_with_variadic_expressions_and_kwargs(make_df, valid_data: list[dict[str, float]]):
    """Test that select works with expressions in variadic args and keyword args."""
    df = make_df(valid_data)
    df = df.select(
        col("sepal_length") + col("petal_length"),
        col("sepal_width") / col("petal_width"),
        sum_col=col("sepal_length") + col("sepal_width"),
        ratio_col=col("petal_length") / col("petal_width"),
    )
    # the first two are unnamed, and daft uses the first argument name as the output field name
    assert df.column_names == ["sepal_length", "sepal_width", "sum_col", "ratio_col"]


def test_select_with_dict_single_arg(make_df, valid_data: list[dict[str, float]]):
    """Test that select works with a dictionary as the only argument."""
    df = make_df(valid_data)
    df = df.select(**{"new_col": col("sepal_length"), "another_col": col("sepal_width")})
    assert df.column_names == ["new_col", "another_col"]


def test_select_with_dict_and_other_args(make_df, valid_data: list[dict[str, float]]):
    """Test that select works when a dictionary is mixed with other arguments."""
    df = make_df(valid_data)
    df = df.select("sepal_length", **{"new_col": col("sepal_width"), "another_col": col("petal_length")})
    # Variadic arguments come first, then dictionary kwargs
    assert df.column_names == ["sepal_length", "new_col", "another_col"]


def test_select_with_dict_complex_expressions(make_df, valid_data: list[dict[str, float]]):
    """Test that select works with complex expressions in the dictionary."""
    df = make_df(valid_data)
    df = df.select(
        **{
            "length_sum": col("sepal_length") + col("petal_length"),
            "width_ratio": col("sepal_width") / col("petal_width"),
            "original_col": col("variety"),
        }
    )
    assert df.column_names == ["length_sum", "width_ratio", "original_col"]


def test_select_with_dict_alias_behavior(make_df, valid_data: list[dict[str, float]]):
    """Test that select with dictionary properly handles aliases."""
    df = make_df(valid_data)
    df = df.select(
        **{
            "renamed_length": col("sepal_length").alias("should_be_ignored"),
            "renamed_width": col("sepal_width"),
        }
    )
    # The dictionary key should override any alias in the expression
    assert df.column_names == ["renamed_length", "renamed_width"]


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda df: df.select(col("a").alias("x"), col("b").alias("x")), id="select"),
        pytest.param(lambda df: df.select(col("a"), col("a")), id="select same column twice"),
        pytest.param(lambda df: df.agg(col("a").sum().alias("x"), col("b").sum().alias("x")), id="agg"),
        pytest.param(lambda df: df.select(col("a").sum().alias("x"), col("b").sum().alias("x")), id="global select"),
        pytest.param(lambda df: df.groupby("a").agg(col("a").count()), id="agg named like its group"),
    ],
)
def test_a_repeated_output_name_is_refused(build: Callable[[daft.DataFrame], daft.DataFrame]) -> None:
    df = daft.from_pydict({"a": [1, 2], "b": [3, 4]})

    with pytest.raises(ValueError, match="more than once"):
        build(df).collect()


def test_a_repeated_output_name_is_refused_in_sql() -> None:
    df = daft.from_pydict({"a": [1, 2], "b": [3, 4]})

    with pytest.raises(Exception, match="more than once"):
        daft.sql("SELECT a AS x, b AS x FROM df", df=df).collect()


def test_one_aggregation_may_be_read_under_two_names() -> None:
    df = daft.from_pydict({"a": [1, 2, 3]})

    result = df.select(col("a").sum().alias("x"), col("a").sum().alias("y")).to_pydict()

    assert result == {"x": [6], "y": [6]}


def test_with_column_still_replaces_an_existing_column() -> None:
    df = daft.from_pydict({"a": [1, 2]})

    assert df.with_column("a", col("a") * 10).to_pydict() == {"a": [10, 20]}
