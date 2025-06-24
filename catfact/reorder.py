import math

from .misc import (
    _expr_map_batches,
    _validate_type,
    _levels,
    _is_enum_or_cat,
    cats,
    dispatch,
    factor,
    is_ordered,
)
from ._databackend import (
    polars as pl,
    pandas as pd,
    PdSeriesOrCat,
    PdSeries,
    PlSeries,
    PlFrame,
    PlExpr,
)
from typing import Callable


def _apply_grouped_expr(grouping: PlSeries, x, expr: PlExpr) -> PlFrame:
    """Returns aggregation with unnamed column for groups, and calc column."""
    gdf = pl.DataFrame({"": x, "grouping": grouping}).group_by("grouping", maintain_order=True)
    return gdf.agg(calc=expr)


@dispatch
def inorder(fct: PlExpr, ordered: bool | None = None) -> PlExpr:
    return _expr_map_batches(fct, inorder, ordered=ordered)


@dispatch
def inorder(fct: PdSeriesOrCat, ordered: bool | None = None) -> PdSeriesOrCat:
    if ordered is None:
        ordered = is_ordered(fct)

    if isinstance(fct, (pd.Series, pd.Categorical)):
        uniq = fct.dropna().unique()

        if isinstance(uniq, pd.Categorical):
            # the result of .unique for a categorical is a new categorical
            # unsurprisingly, it also sorts the categories, so reorder manually
            # (note that this also applies to Series[Categorical].unique())
            categories = uniq.categories[uniq.dropna().codes]
            cat = pd.Categorical(fct, categories, ordered=ordered)
        else:
            cat = pd.Categorical(fct, uniq, ordered=ordered)

        return pd.Series(cat) if isinstance(fct, pd.Series) else cat  # type: ignore

    raise NotImplementedError()


@dispatch
def inorder(fct: PlSeries, ordered: bool | None = None) -> PlSeries:
    """Return factor with categories ordered by when they first appear.

    Parameters
    ----------
    fct : list-like
        A pandas Series, Categorical, or list-like object
    ordered : bool
        Whether to return an ordered categorical. By default a Categorical inputs'
        ordered setting is respected. Use this to override it.

    See Also
    --------
    fct.infreq : Order categories by value frequency count.

    Examples
    --------

    >>> import pandas as pd
    >>> import catfact as fct
    >>> cat = pd.Categorical(["c", "a", "b"])
    >>> cat
    ['c', 'a', 'b']
    Categories (3, object): ['a', 'b', 'c']

    Note that above the categories are sorted alphabetically. Use `fct.inorder`
    to keep the categories in first-observed order.

    >>> fct.inorder(cat)
    ['c', 'a', 'b']
    Categories (3, object): ['c', 'a', 'b']

    By default, the ordered setting of categoricals is respected. Use the ordered
    parameter to override it.

    >>> cat2 = pd.Categorical(["z", "a", "b"], ordered=True)
    >>> fct.inorder(cat2)
    ['z', 'a', 'b']
    Categories (3, object): ['z' < 'a' < 'b']

    >>> fct.inorder(cat2, ordered=False)
    ['z', 'a', 'b']
    Categories (3, object): ['z', 'a', 'b']

    """

    if ordered is not None:
        raise NotImplementedError("Polars does not support ordered categoricals.")

    if _is_enum_or_cat(fct):
        return factor(fct, levels=fct.unique(maintain_order=True))

    return factor(fct)


@dispatch
def infreq(fct: PlExpr, ordered: bool | None = None) -> PlExpr:
    return _expr_map_batches(fct, infreq, ordered=ordered)


@dispatch
def infreq(fct: PdSeriesOrCat, ordered: bool | None = None) -> PdSeriesOrCat:
    if ordered is None:
        ordered = is_ordered(fct)

    # Series sorts in descending frequency order
    ser = pd.Series(fct)
    freq = ser.value_counts()

    # freq.index can be a CategoricalIndex, which would preserve
    # the original category order, so we convert it to a list
    cat = pd.Categorical(ser, categories=list(freq.index), ordered=ordered)

    if isinstance(fct, pd.Series):
        return pd.Series(cat)

    return cat


@dispatch
def infreq(fct: PlSeries, ordered: bool | None = None) -> PlSeries:
    """Return a factor with categories ordered by frequency (largest first)

    Parameters
    ----------
    fct :
        A Series or Expression
    ordered : bool
        Whether to return an ordered categorical. By default a Categorical inputs'
        ordered setting is respected. Use this to override it.

    See Also
    --------
    fct.inorder : Order categories by when they're first observed.

    Examples
    --------
    >>> import pandas as pd
    >>> import catfact as fct
    >>> fct.infreq(pd.Categorical(["c", "a", "c", "c", "a", "b"]))
    ['c', 'a', 'c', 'c', 'a', 'b']
    Categories (3, object): ['c', 'a', 'b']

    """

    _validate_type(fct)

    levels = fct.value_counts(sort=True).drop_nulls()[fct.name]

    return factor(fct, levels)


def _insert_index(lst: list, index, value) -> list:
    """Insert element into list after index.

    Note that this is similar to list.index, but inserts after.
    Note also that value may be a list to be unpacked.
    """
    import math

    # note this preserves infinities...
    n = len(lst)
    if math.isinf(index):
        if index > 0:
            index = n + 1
        else:
            raise ValueError("Cannot insert at negative infinity. Use index of 0 instead.")

    res = lst[:]
    res[index:index] = value
    return res


@dispatch
def inseq(fct: PlExpr) -> PlExpr:
    return _expr_map_batches(fct, inseq)


@dispatch
def inseq(fct: PlSeries) -> PlSeries:
    """Return a factor with categories ordered lexically (alphabetically)."""

    if _is_enum_or_cat(fct):
        return factor(fct, levels=fct.cat.get_categories().sort())

    levels = fct.unique().drop_nulls().sort()
    return factor(fct, levels)


@dispatch
def relevel(
    fct: PlExpr,
    *args,
    func: Callable[[PlSeries], PlSeries] | None = None,
    index: int | float = math.inf,
) -> PlExpr:
    return _expr_map_batches(fct, relevel, *args, func=func, index=index)


@dispatch
def relevel(
    fct: PdSeriesOrCat,
    *args,
    func: Callable[[PdSeries], PdSeries] | None = None,
    index: int | float = math.inf,
) -> PdSeriesOrCat:
    old_levels = cats(pd.Categorical(fct))
    if func is not None:
        if args:
            raise ValueError("Cannot pass positional arguments when func is an expression.")
        first_levels = func(pd.Series(old_levels))
    else:
        first_levels = pd.Series(args)

    unmatched_levels = [lvl for lvl in old_levels if lvl not in set(first_levels)]
    new_levels = _insert_index(unmatched_levels, index, list(first_levels))

    res = pd.Categorical(fct, categories=new_levels)
    return fct.__class__(res)


@dispatch
def relevel(
    fct: PlSeries,
    *args,
    func: Callable[[PlSeries], PlSeries] | None = None,
    index: int | float = math.inf,
) -> PlSeries:
    """Manually change the order of levels in a factor."""

    # TODO: all the series -> list -> series is a bit zany.
    old_levels = _levels(fct)
    if func is not None:
        if args:
            raise ValueError("Cannot pass positional arguments when func is an expression.")
        first_levels = func(old_levels)
    else:
        first_levels = pl.Series(args)

    # polars only has a set_difference method for lists
    unmatched_levels = [lvl for lvl in old_levels if lvl not in set(first_levels)]
    new_levels = _insert_index(unmatched_levels, index, list(first_levels))

    return factor(fct, levels=pl.Series(new_levels))


@dispatch
def reorder(fct: PlExpr, x: PlExpr, func: PlExpr | None = None, desc: bool = False) -> PlExpr:
    # return pl.map_batches([fct, x], lambda sers: reorder(sers[0], sers[1], func=func, desc=desc))

    # we need to use the struct .map_batches method
    # which has is_elementwise=False, so that the function
    # does not execute once per chunk (since the final levels
    # depend on all the values in the Series).
    return pl.struct([fct.alias("field_0"), x.alias("field_1")]).map_batches(
        lambda struct_series: reorder(
            struct_series.struct.field("field_0"),
            struct_series.struct.field("field_1"),
            func=func,
            desc=desc,
        )
    )


@dispatch
def reorder(fct: PdSeriesOrCat, x, func="median", desc=False) -> PdSeriesOrCat:
    # TODO: test this concrete implementation

    x_vals = x.values if isinstance(x, pd.Series) else x
    s = pd.Series(x_vals, index=fct)

    # sort groups by calculated agg func. note that groupby uses dropna=True by default,
    # but that's okay, since pandas categoricals can't order the NA category
    ordered = s.groupby(level=0).agg(func).sort_values(ascending=not desc)

    out = pd.Categorical(fct, categories=ordered.index)
    return fct.__class__(out)


@dispatch
def reorder(fct: PlSeries, x: PlSeries, func: PlExpr | None = None, desc: bool = False) -> PlSeries:
    """Return copy of fct, with categories reordered according to values in x.

    Parameters
    ----------
    fct :
        A Series, which may be a string or factor-like.
    x :
        Values used to reorder categorical. Must be same length as fct.
    func :
        Function run over all values within a level of the categorical.
    desc :
        Whether to sort in descending order.

    Examples
    --------

    >>> import polars as pl
    >>> import catfact as fct
    >>> ord1 = fct.reorder(pl.Series(['a', 'a', 'b']), pl.Series([4, 3, 2]))
    >>> fct.cats(ord1).to_list()
    ['b', 'a']


    >>> ord2 = fct.reorder(pl.Series(['a', 'a', 'b']), pl.Series([4, 3, 2]), desc = True)
    >>> fct.cats(ord2).to_list()
    ['a', 'b']

    >>> ord3 = fct.reorder(
    ...     pl.Series(['x', 'x', 'y']),
    ...     pl.Series([4, 0, 2]),
    ...     pl.element().max()
    ... )
    >>> fct.cats(ord3).to_list()
    ['y', 'x']

    """

    if func is None:
        func = pl.element().median()

    if isinstance(x, pl.Series):
        cat_aggs = _apply_grouped_expr(fct, x, func)
    else:
        raise NotImplementedError("Currently, x must be a polars.Series")

    levels = cat_aggs.sort("calc", descending=desc)["grouping"].drop_nulls()
    res = factor(fct.cast(pl.String), levels)
    return res


@dispatch
def rev(fct: PlExpr) -> PlExpr:
    return _expr_map_batches(fct, rev)


@dispatch
def rev(fct: PdSeriesOrCat) -> PdSeriesOrCat:
    cat = pd.Categorical(fct)

    rev_levels = list(reversed(cat.categories))

    out = fct.reorder_categories(rev_levels)
    return pd.Series(out) if isinstance(fct, pd.Series) else out  # type: ignore


@dispatch
def rev(fct: PlSeries) -> PlSeries:
    """Reverse the order of a factor's levels.

    Parameters
    ----------
    fct :
        A Series

    Examples
    --------
    >>> import pandas as pd
    >>> import catfact as fct
    >>> cat = pd.Categorical(["a", "b", "c"])
    >>> cat
    ['a', 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']

    >>> fct.rev(cat)
    ['a', 'b', 'c']
    Categories (3, object): ['c', 'b', 'a']

    """

    res = factor(fct)
    return res.cast(pl.Enum(res.cat.get_categories().reverse()))
