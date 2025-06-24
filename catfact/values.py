from .misc import dispatch, _expr_map_batches, _flip_mapping, _lvls_revalue
from ._databackend import (
    polars as pl,
    pandas as pd,
    PdSeries,
    PdSeriesOrCat,
    PlExpr,
    PlSeries,
    PlFrame,
)
from typing import cast


@dispatch
def collapse(fct: PlExpr, other: str | None = None, /, **kwargs: str | list[str]):
    return _expr_map_batches(fct, collapse, other, **kwargs)


@dispatch
def collapse(
    fct: PdSeriesOrCat, other: str | None = None, /, **kwargs: str | list[str]
) -> PdSeriesOrCat:
    recat = kwargs
    group_other = other

    if not isinstance(fct, pd.Categorical):
        new_fct = pd.Categorical(fct)
    else:
        new_fct = fct

    # each existing cat will map to a new one ----
    # need to know existing to new cat
    # need to know new cat to new code
    old_to_new = _flip_mapping(**recat)
    cat_to_new: dict[str, str] = {}
    for old in new_fct.categories:
        old = cast(str, old)
        if group_other:
            cat_to_new[old] = old_to_new.get(old, group_other)
        else:
            cat_to_new[old] = old_to_new.get(old, old)

    # collapse all unspecified cats to group_other if specified ----
    for k, v in cat_to_new.items():
        if v is None:
            if group_other is not None:
                cat_to_new[k] = group_other
            else:
                cat_to_new[k] = k

    # map from old cat to new code ----

    # calculate new codes
    ordered_cats = {new: True for old, new in cat_to_new.items()}

    # move the other group to last in the ordered set
    if group_other is not None:
        try:
            del ordered_cats[group_other]
            ordered_cats[group_other] = True
        except KeyError:
            pass

    # map new category name to code
    new_cat_set = {k: ii for ii, k in enumerate(ordered_cats)}

    # at this point, we need remap codes to the other category
    # make an array, where the index is old code + 1 (so missing val index is 0)
    old_code_to_new = [-1] + [new_cat_set[new_cat] for new_cat in cat_to_new.values()]

    # map old cats to new codes
    new_codes = [old_code_to_new[code + 1] for code in new_fct.codes]
    new_cats = list(new_cat_set)

    out = pd.Categorical.from_codes(new_codes, new_cats)
    return fct.__class__(out)


@dispatch
def collapse(fct: PlSeries, other: str | None = None, /, **kwargs: str | list[str]):
    """Return copy of fct with categories renamed. Optionally group all others.

    Parameters
    ----------
    fct :
        A Series
    other :
        An optional string, specifying what all other categories should be named.
        This will always be the last category level in the result.
    **kwargs :
        Keyword arguments of form new_cat_name = old_cat_name. old_cat_name may be
        a list of existing categories, to be given the same name.

    Notes
    -----
    Resulting categories are ordered according to the earliest level replaced.
    If we rename the first and last levels to "c", then "c" is the first level.

    Examples
    --------
    >>> import pandas as pd
    >>> import catfact as fct

    >>> cat = pd.Categorical(['a', 'b', 'c'])
    >>> fct.collapse(cat, x = 'a')
    ['x', 'b', 'c']
    Categories (3, object): ['x', 'b', 'c']

    >>> fct.collapse(cat, "others", x = "a")
    ['x', 'others', 'others']
    Categories (2, object): ['x', 'others']

    >>> fct.collapse(cat, ab = ["a", "b"])
    ['ab', 'ab', 'c']
    Categories (2, object): ['ab', 'c']

    """
    # Polars does not allow using .replace on categoricals
    # so we need to change the string values themselves
    if fct.dtype.is_(pl.Categorical):
        fct = fct.cast(pl.String)
    replace_map = _flip_mapping(**kwargs)
    # TODO: should it be strict?
    # TODO: will fail for categoricals

    levels = [*kwargs, *([other] if other is not None else [])]
    return fct.replace_strict(replace_map, default=other, return_dtype=pl.Enum(levels))


@dispatch
def recode(fct: PlExpr, **kwargs):
    return _expr_map_batches(fct, recode, **kwargs)


@dispatch
def recode(fct: PdSeriesOrCat, **kwargs) -> PdSeriesOrCat:
    return collapse(fct, **kwargs)


@dispatch
def recode(fct: PlSeries, **kwargs):
    """Return copy of fct with renamed categories.

    Parameters
    ----------
    fct :
        A Series
    **kwargs :
        Arguments of form new_name = old_name.

    See Also
    --------
    fct.collapse: similar function, but allows grouping all remaining categories.

    Examples
    --------

    >>> import pandas as pd
    >>> import catfact as fct

    >>> cat = pd.Categorical(['a', 'b', 'c'])
    >>> fct.recode(cat, z = 'c')
    ['a', 'b', 'z']
    Categories (3, object): ['a', 'b', 'z']

    >>> fct.recode(cat, x = ['a', 'b'])
    ['x', 'x', 'c']
    Categories (2, object): ['x', 'c']

    >>> funky_cat = pd.Categorical(["..x", "some y"])
    >>> fct.recode(funky_cat, **{"x": "..x", "y": "some y"})
    ['x', 'y']
    Categories (2, object): ['x', 'y']
    """

    return collapse(fct, **kwargs)


def _calc_lump_sum(x: PlSeries, w: PlSeries | None = None) -> PlFrame:
    """Return a DataFrame with columns x, calc for grouped sums."""

    return (
        pl.select(x=x, w=w)
        .group_by("x", maintain_order=True)
        .agg(calc=pl.col("w").sum())
        .sort("calc", descending=True)
        .drop_nulls()
    )


@dispatch
def lump_n(fct: PlExpr, n: int = 5, weights: PlExpr | None = None, other: str = "Other") -> PlExpr:
    if weights is not None:
        return pl.map_batches(
            [fct, weights],
            lambda sers: lump_n(sers[0], n, sers[1], other=other),
            return_dtype=pl.Enum,
        )

    return _expr_map_batches(fct, lump_n, n, weights, other)


@dispatch
def lump_n(
    fct: PdSeriesOrCat, n: int = 5, weights: PdSeries | None = None, other: str = "Other"
) -> PdSeriesOrCat:
    ser = pd.Series(
        weights.array if weights is not None else 1,
        index=fct,
    )
    counts = ser.groupby(level=0).sum()

    ascending = n < 0
    sorted_arr = counts.sort_values(ascending=ascending)
    keep_cats = list(sorted_arr.iloc[: abs(n)].index)

    return collapse(fct, other, **dict(zip(keep_cats, keep_cats)))


@dispatch
def lump_n(
    fct: PlSeries, n: int = 5, weights: PlSeries | None = None, other: str = "Other"
) -> PlSeries:
    """Lump all levels except the n most frequent.

    Parameters
    ----------
    fct :
        A Series
    n :
        Number of categories to lump together.
    weights :
        Weights.
    other :
        Name of the new category.

    Returns
    -------
    Series
        A new series with the most common n categories lumped together.
    """

    # TODO: handle least frequent if n < 0
    # order by descending frequency
    if weights is None:
        # likely faster calculation
        ordered = fct.value_counts(sort=True).drop_nulls()[fct.name]
    else:
        ordered = _calc_lump_sum(fct, weights)["x"]

    new_levels = pl.select(
        res=pl.when(pl.arange(len(ordered)) < n).then(ordered).otherwise(pl.lit(other))
    )["res"]

    releveled = _lvls_revalue(fct, ordered, new_levels)
    # fct_relevel
    if other in releveled.cat.get_categories():
        uniq_levels = new_levels[: n + 1]
        return releveled.cast(pl.Enum(uniq_levels))

    return releveled


@dispatch
def lump_prop(fct: PlExpr, prop: float, weights=None, other="Other") -> PlExpr:
    return _expr_map_batches(fct, lump_prop, prop=prop, weights=weights, other=other)


@dispatch
def lump_prop(
    fct: PdSeriesOrCat, prop: float, weights: PdSeries | None = None, other: str = "Other"
) -> PdSeriesOrCat:
    ser = pd.Series(
        weights.array if weights is not None else 1,
        index=fct,
    )
    counts = ser.groupby(level=0).sum()

    sorted_arr = counts.sort_values() / counts.sum()

    if prop < 0:
        res = sorted_arr.loc[sorted_arr <= abs(prop)]
    else:
        res = sorted_arr.loc[sorted_arr > prop]

    return fct.__class__(res.index)


@dispatch
def lump_prop(fct: PlSeries, prop: float, weights=None, other="Other") -> PlSeries:
    """Lump levels that appear in fewer than some proportion in the series."""

    fct = fct.rename("x")
    if weights is None:
        props = fct.drop_nulls().value_counts(sort=True, normalize=True, name="calc")
    else:
        props = _calc_lump_sum(fct, weights).with_columns(
            calc=pl.col("calc") / pl.col("calc").sum()
        )

    ordered = props["x"]
    new_levels = props.select(
        res=pl.when(pl.col("calc") >= prop).then(pl.col("x")).otherwise(pl.lit(other))
    )["res"]

    releveled = _lvls_revalue(fct, ordered, new_levels)
    # fct_relevel
    if other in releveled.cat.get_categories():
        uniq_levels = new_levels.unique(maintain_order=True)
        return releveled.cast(pl.Enum(uniq_levels))

    return releveled


@dispatch
def lump_min(fct: PlExpr, n, weights: PlExpr | None = None, other="Other") -> PlExpr:
    if weights is not None:
        return pl.map_batches(
            [fct, weights],
            lambda sers: lump_min(sers[0], n, sers[1], other=other),
            return_dtype=pl.Enum,
        )

    return _expr_map_batches(fct, lump_min, n, weights, other)


@dispatch
def lump_min(fct: PlSeries, n, weights: PlSeries | None = None, other="Other") -> PlSeries:
    """Lump levels that appear fewer than n times in the series."""
    raise NotImplementedError()


@dispatch
def lump_lowfreq(fct: PlExpr, other="Other") -> PlExpr:
    return _expr_map_batches(fct, lump_lowfreq, other=other)


@dispatch
def lump_lowfreq(fct: PlSeries, other="Other") -> PlSeries:
    """Lump low frequency level, keeping other the smallest level."""

    counts = fct.value_counts(sort=True).drop_nulls()

    # find index for first count larger than remainder
    remain = counts["count"].sum()
    for n, crnt_count in enumerate(counts["count"]):
        remain -= crnt_count
        if crnt_count > remain:
            break

    ordered = counts[fct.name]
    new_levels = pl.select(
        res=pl.when(pl.arange(len(ordered)) <= n).then(ordered).otherwise(pl.lit(other))
    )["res"]

    releveled = _lvls_revalue(fct, ordered, new_levels)
    # fct_relevel
    if other in releveled.cat.get_categories():
        uniq_levels = new_levels.unique(maintain_order=True)
        return releveled.cast(pl.Enum(uniq_levels))

    return releveled
