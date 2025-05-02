import math
import pytest
import polars as pl
from catfact import inorder, infreq, inseq, relevel, reorder, rev, to_list, cats

DATA = ["c", "a", "c", "b", "b"]
# fct_inorder: c, a, b
# fct_infreq: c, b, a
# fct_inseq: a, b, c

# TODO: should also assert series length, content, etc..


def test_inorder():
    res = inorder(pl.Series(DATA))
    assert to_list(cats(res)) == ["c", "a", "b"]


def test_infreq():
    res = infreq(pl.Series(DATA))
    assert to_list(cats(res)) == ["c", "b", "a"]


def test_inseq():
    res = inseq(pl.Series(DATA))
    assert to_list(cats(res)) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "f",
    [
        inorder,
        infreq,
        inseq,
    ],
)
def test_infunc_on_cats(f):
    res1 = f(pl.Series(["c", "a", "b"]))
    res2 = f(res1)

    assert res1.dtype == pl.Enum
    assert res1.equals(res2)


def test_relevel():
    fct = pl.Series(["a", "a", "b", "c"]).cast(pl.Categorical)

    res = relevel(fct, "b", index=math.inf)
    assert to_list(cats(res)) == ["a", "c", "b"]

    res2 = relevel(fct, "b", index=0)
    assert to_list(cats(res2)) == ["b", "a", "c"]

    res3 = relevel(fct, "c", "b", index=0)
    assert to_list(cats(res3)) == ["c", "b", "a"]


def test_relevel_func():
    fct = pl.Series(["a", "a", "b", "c"]).cast(pl.Categorical)

    res = relevel(fct, func=lambda x: x.reverse())
    assert to_list(cats(res)) == ["c", "b", "a"]

    res = relevel(fct, func=lambda x: x.filter(x.str.starts_with("a")))
    assert to_list(cats(res)) == ["b", "c", "a"]


def test_reorder():
    # TODO: does this match forcats?
    fct = pl.Series(DATA)
    x = pl.Series([1] * len(DATA))
    res = reorder(fct, x, pl.element().sum())

    assert to_list(cats(res)) == ["a", "c", "b"]

    res_desc = reorder(fct, x, pl.element().sum(), desc=True)

    assert to_list(cats(res_desc)) == ["c", "b", "a"]


def test_rev():
    fct = pl.Series(["a", "b", "c"]).cast(pl.Categorical)
    res = rev(fct)
    assert to_list(cats(res)) == ["c", "b", "a"]
