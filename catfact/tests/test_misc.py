import pytest
import polars as pl
from catfact import factor, cats


DATA = ["c", "a", "c", "b", "b"]

@pytest.mark.parametrize("x", [
    pl.Series(["c", "a", "c", "b", "b"]).cast(pl.Categorical("physical")),
])
def test_cats(x):
    assert cats(x).to_list() == ["c", "a", "b"]

