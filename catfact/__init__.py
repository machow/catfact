from __future__ import annotations

import polars as pl
from ._databackend import PlSeries, PlFrame
from ddispatch import dispatch
from typing import Any


def _validate_type(x: PlSeries):
    if not x.dtype.is_(pl.String):
        raise TypeError()


def _levels(x: PlSeries) -> PlSeries:
    return x.unique(maintain_order=True).drop_nulls()


def _flip_mapping(**kwargs: str | list[str]) -> dict[str, str]:
    """Flip from new = old mappings to old = new style."""

    # TODO: validate old values not overridden in mapping
    mapping = {}
    for new, old in kwargs.items():
        if isinstance(old, str):
            mapping[old] = new
        elif isinstance(old, list):
            for o in old:
                mapping[o] = new
        else:
            raise TypeError(f"Expected str or list, got {type(old)}")
    
    return mapping

def _lvls_revalue(fct: PlSeries, old_levels: PlSeries, new_levels: PlSeries) -> PlSeries:
    """Revalue levels of a categorical series."""
    if fct.dtype.is_(pl.Categorical) or fct.dtype.is_(pl.Enum):
        fct = fct.cast(pl.String)
    
    return fct.replace_strict(old_levels, new_levels, return_dtype=pl.Enum(new_levels.unique(maintain_order=True)))


@dispatch
def to_list(x: PlSeries) -> list[Any]:
    """Convert series to a list."""
    return x.to_list()


@dispatch
def cats(x: PlSeries) -> PlSeries:
    """Return the levels of a categorical series.

    Parameters
    ----------
    x :
        A pandas Series, Categorical, or list-like object

    Returns
    -------
    list
        The levels of the categorical series.

    """
    return x.cat.get_categories()


#


def _apply_grouped_expr(grouping: PlSeries, x, expr: pl.Expr) -> PlFrame:
    """Returns aggregation with unnamed column for groups, and calc column."""
    gdf = pl.DataFrame({"": x, "grouping": grouping}).group_by("grouping", maintain_order=True)
    return gdf.agg(calc=expr)


@dispatch
def factor(x: PlSeries) -> PlSeries:
    """Create a factor, a categorical series whose level order can be specified."""

    _validate_type(x)

    levels = _levels(x)
    return x.cast(pl.Enum(levels))


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

    return fct.cast(pl.Enum(levels))

@dispatch
def inseq(x: PlSeries) -> PlSeries:
    """Return a factor with categories ordered lexically (alphabetically)."""

    levels = x.unique().drop_nulls().sort()
    return x.cast(pl.Enum(levels))

@dispatch
def reorder(fct: PlSeries, x: PlSeries, func=None, desc=False) -> PlSeries:
    """Return copy of fct, with categories reordered according to values in x.

    Parameters
    ----------
    fct :
        A pandas.Categorical, or array(-like) used to create one.
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


@dispatch
def lump_n(fct: PlSeries, n: int = 5, weights = None, other: str = "Other") -> PlSeries:
    """Lump the most common n categories into a single category.

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

    # order by descending frequency
    ordered = fct.value_counts(sort=True).drop_nulls()[fct.name]

    new_levels = pl.select(res=pl.when(pl.arange(len(ordered)) < n).then(ordered).otherwise(pl.lit(other)))["res"]
    
    releveled = _lvls_revalue(fct, ordered, new_levels)
    # fct_relevel
    if other in releveled.cat.get_categories():
        ordered_levels = [*ordered[:n], other]
        return releveled.cast(pl.Enum(ordered_levels))
    
    return releveled