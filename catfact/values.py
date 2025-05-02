from .misc import dispatch, _flip_mapping, _lvls_revalue
from ._databackend import polars as pl, PlSeries


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
def lump_n(fct: PlSeries, n: int = 5, weights=None, other: str = "Other") -> PlSeries:
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

    new_levels = pl.select(
        res=pl.when(pl.arange(len(ordered)) < n).then(ordered).otherwise(pl.lit(other))
    )["res"]

    releveled = _lvls_revalue(fct, ordered, new_levels)
    # fct_relevel
    if other in releveled.cat.get_categories():
        ordered_levels = [*ordered[:n], other]
        return releveled.cast(pl.Enum(ordered_levels))

    return releveled
