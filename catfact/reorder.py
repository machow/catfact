import math

from .misc import dispatch, factor, _validate_type, _levels
from ._databackend import polars as pl, PlSeries, PlFrame, PlExpr
from typing import Callable


def _apply_grouped_expr(grouping: PlSeries, x, expr: PlExpr) -> PlFrame:
    """Returns aggregation with unnamed column for groups, and calc column."""
    gdf = pl.DataFrame({"": x, "grouping": grouping}).group_by("grouping", maintain_order=True)
    return gdf.agg(calc=expr)


@dispatch
def inorder(x: PlSeries, ordered=None) -> PlSeries:
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
    fct_infreq : Order categories by value frequency count.

    """
    # TODO: warn that polars has no ordered mode

    if ordered is not None:
        raise NotImplementedError()

    return factor(x)


@dispatch
def infreq(fct: PlSeries, ordered=None) -> PlSeries:
    """Return a factor with categories ordered by frequency (largest first)

    Parameters
    ----------
    fct : list-like
        A pandas Series, Categorical, or list-like object
    ordered : bool
        Whether to return an ordered categorical. By default a Categorical inputs'
        ordered setting is respected. Use this to override it.

    See Also
    --------
    fct_inorder : Order categories by when they're first observed.

    Examples
    --------

    >>> fct_infreq(["c", "a", "c", "c", "a", "b"])
    ['c', 'a', 'c', 'c', 'a', 'b']
    Categories (3, object): ['c', 'a', 'b']

    """

    _validate_type(fct)

    levels = fct.value_counts(sort=True).drop_nulls()[fct.name]

    return fct.cast(pl.Enum(levels.cast(pl.String)))


@dispatch
def inseq(x: PlSeries) -> PlSeries:
    """Return a factor with categories ordered lexically (alphabetically)."""

    levels = x.unique().drop_nulls().sort()
    return x.cast(pl.Enum(levels.cast(pl.String)))


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
def reorder(fct: PlSeries, x: PlSeries, func=None, desc=False) -> PlSeries:
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

    Notes
    -----
    NaN categories can't be ordered. When func returns NaN, sorting
    is always done with NaNs last.


    Examples
    --------

    >>> fct_reorder(['a', 'a', 'b'], [4, 3, 2])
    ['a', 'a', 'b']
    Categories (2, object): ['b', 'a']

    >>> fct_reorder(['a', 'a', 'b'], [4, 3, 2], desc = True)
    ['a', 'a', 'b']
    Categories (2, object): ['a', 'b']

    >>> fct_reorder(['x', 'x', 'y'], [4, 0, 2], np.max)
    ['x', 'x', 'y']
    Categories (2, object): ['y', 'x']

    """

    if func is None:
        func = pl.element().median()

    if isinstance(x, pl.Series):
        cat_aggs = _apply_grouped_expr(fct, x, func)
    else:
        raise NotImplementedError("Currently, x must be a polars.Series")

    levels = cat_aggs.sort("calc", descending=desc)["grouping"].drop_nulls()
    return fct.cast(pl.Enum(levels))


@dispatch
def rev(fct: PlSeries) -> PlSeries:
    """Reverse the order of a factor's levels.

    Parameters
    ----------
    fct :
        A Series to return as a factor.

    """

    res = factor(fct)
    return res.cast(pl.Enum(res.cat.get_categories().reverse()))
