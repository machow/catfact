from .misc import dispatch, _flip_mapping, _lvls_revalue
from ._databackend import polars as pl, PlSeries, PlFrame


@dispatch
def collapse(fct: PlSeries, other: str | None = None, /, **kwargs: list[str]):
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
def recode(fct: PlSeries, **kwargs):
    """Return copy of fct with renamed categories.

    Parameters
    ----------
    fct :
        A pandas.Categorical, or array(-like) used to create one.
    **kwargs :
        Arguments of form new_name = old_name.

    Examples
    --------
    >>> cat = ['a', 'b', 'c']
    >>> fct_recode(cat, z = 'c')
    ['a', 'b', 'z']
    Categories (3, object): ['a', 'b', 'z']

    >>> fct_recode(cat, x = ['a', 'b'])
    ['x', 'x', 'c']
    Categories (2, object): ['x', 'c']

    >>> fct_recode(cat, {"x": ['a', 'b']})
    ['x', 'x', 'c']
    Categories (2, object): ['x', 'c']
    """

    # TODO: is it worth keeping this function?
    # factor index is first replaced level
    # need to do fct_collapse first
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
def lump_n(fct: PlSeries, n: int = 5, weights=None, other: str = "Other") -> PlSeries:
    """Lump all levels except the n most frequent.

    Parameters
    ----------
    x :
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
        ordered = _calc_lump_sum(fct, weights, prop=False)["x"]

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
def lump_prop(x: PlSeries, prop: float, weights=None, other="Other") -> PlSeries:
    """Lump levels that appear in fewer than some proportion in the series."""

    x = x.rename("x")
    if weights is None:
        props = x.drop_nulls().value_counts(sort=True, normalize=True, name="calc")
    else:
        props = _calc_lump_sum(x, weights).with_columns(calc=pl.col("calc") / pl.col("calc").sum())

    ordered = props["x"]
    new_levels = props.select(
        res=pl.when(pl.col("calc") >= prop).then(pl.col("x")).otherwise(pl.lit(other))
    )["res"]

    releveled = _lvls_revalue(x, ordered, new_levels)
    # fct_relevel
    if other in releveled.cat.get_categories():
        uniq_levels = new_levels.unique(maintain_order=True)
        return releveled.cast(pl.Enum(uniq_levels))

    return releveled


@dispatch
def lump_min(x: PlSeries, n, weights: PlSeries | None = None, other="Other") -> PlSeries:
    """Lump levels that appear fewer than n times in the series."""


@dispatch
def lump_lowfreq(x: PlSeries, other="Other") -> PlSeries:
    """Lump low frequency level, keeping other the smallest level."""

    counts = x.value_counts(sort=True).drop_nulls()

    # find index for first count larger than remainder
    remain = counts["count"].sum()
    for n, crnt_count in enumerate(counts["count"]):
        remain -= crnt_count
        if crnt_count > remain:
            break

    ordered = counts[x.name]
    new_levels = pl.select(
        res=pl.when(pl.arange(len(ordered)) <= n).then(ordered).otherwise(pl.lit(other))
    )["res"]

    releveled = _lvls_revalue(x, ordered, new_levels)
    # fct_relevel
    if other in releveled.cat.get_categories():
        uniq_levels = new_levels.unique(maintain_order=True)
        return releveled.cast(pl.Enum(uniq_levels))

    return releveled
