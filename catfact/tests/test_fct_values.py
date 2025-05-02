import pytest
import polars as pl
from catfact import recode, collapse, lump_n, lump_prop, lump_lowfreq, to_list, cats

DATA = ["Low", "Low-ish", "High", "Very High"]


# fct_recode
@pytest.mark.parametrize(
    "ser",
    [
        pl.Series(DATA),
        pl.Series(DATA).cast(pl.Categorical("physical")),
    ],
)
def test_collapse(ser):
    # TODO: support categoricals
    res = collapse(ser, low=["Low", "Low-ish"], high=["High", "Very High"])

    assert to_list(cats(res)) == ["low", "high"]
    assert to_list(res) == ["low", "low", "high", "high"]


@pytest.mark.parametrize(
    "ser",
    [
        pl.Series(DATA),
    ],
)
def test_collapse_other(ser):
    res = collapse(ser, "other", low=["Low", "Low-ish"])
    assert to_list(cats(res)) == ["low", "other"]
    assert to_list(res) == ["low", "low", "other", "other"]


@pytest.mark.parametrize(
    "ser",
    [
        pl.Series(DATA),
    ],
)
def test_recode(ser):
    recoded = recode(ser, low=["Low", "Low-ish"], high=["High", "Very High"])
    assert to_list(cats(recoded)) == ["low", "high"]
    assert to_list(recoded) == ["low", "low", "high", "high"]


@pytest.mark.parametrize(
    "ser",
    [
        pl.Series(["b", "a", "a", "a", "c", "c"]),
    ],
)
def test_lump_n(ser):
    res = lump_n(ser, n=2)
    assert to_list(cats(res)) == ["a", "c", "Other"]


@pytest.mark.parametrize(
    "ser",
    [
        pl.Series(["b", "a", "a", "a", "c", "c"]),
    ],
)
def test_lump_prop(ser):
    res = lump_prop(ser, prop=0.5)
    assert to_list(cats(res)) == ["a", "Other"]

    res2 = lump_prop(ser, prop=0.3)
    assert to_list(cats(res2)) == ["a", "c", "Other"]


def test_lump_lowfreq():
    res = lump_lowfreq(pl.Series(["b", "a", "c"]))
    assert to_list(cats(res)) == ["b", "a", "c"]

    # TODO: note that results currently sorted
    res2 = lump_lowfreq(pl.Series(["b", "a", "c", "c"]))
    assert to_list(cats(res2)) == ["c", "b", "a"]

    res3 = lump_lowfreq(pl.Series(["b", "a", "a", "a", "c"]))
    assert to_list(cats(res3)) == ["a", "Other"]
