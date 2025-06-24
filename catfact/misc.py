from __future__ import annotations


from ._databackend import polars as pl, pandas as pd, PdSeriesOrCat, PlSeries, PlExpr
from ddispatch import dispatch
from typing import Any, Callable, ParamSpec, Concatenate

# For each fct function, need to handle these cases:
#   - categorical: methods like replace not available.
#   - non-categorical: methods like replace available, but levels not calculated yet.
# TODO: fct_shuffle, fct_relevel(after=...), fct_drop, fct_c
# TODO: note cannot store NA in levels

P = ParamSpec("P")


def _expr_map_batches(
    expr: PlExpr, f: Callable[Concatenate[PlSeries, P], PlSeries], *args, **kwargs
) -> PlExpr:
    """Partial function for use with map_batches."""

    # TODO: currently, there is no way to specify the return_dtype, since
    # Polars expects levels to be part of the dtype, and it appears some
    # parts of Polars fail if levels are not specified.
    return expr.map_batches(lambda x: f(x, *args, **kwargs), is_elementwise=False)


def _validate_type(x: PlSeries):
    if x.dtype == pl.String or x.dtype == pl.Categorical or x.dtype == pl.Enum:
        return

    raise TypeError(f"Unsupported Series dtype: {type(x.dtype)}.")


def _levels(x: PlSeries) -> PlSeries:
    """Return levels to use in the creation of a factor."""

    if x.dtype == pl.Categorical or x.dtype == pl.Enum:
        return x.cat.get_categories()

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

    return fct.replace_strict(
        old_levels, new_levels, return_dtype=pl.Enum(new_levels.unique(maintain_order=True))
    )


def _lvls_reorder(fct: PlSeries, idx: PlSeries) -> PlSeries: ...


def _is_enum_or_cat(levels: PlSeries) -> bool:
    return levels.dtype == pl.Categorical or levels.dtype == pl.Enum


@dispatch
def is_ordered(x: PdSeriesOrCat):
    import pandas as pd

    if isinstance(x, pd.Categorical):
        return x.ordered
    elif isinstance(x, pd.Series) and pd.api.types.is_categorical_dtype(x):
        return x.cat.ordered

    return None


@dispatch
def is_ordered(x: PlSeries):
    """Check if a Polars Series is ordered."""

    raise NotImplementedError(
        "Polars Categorical and Enum dtypes do not support an `ordered` flag attribute."
    )


@dispatch
def to_list(x: PlSeries) -> list[Any]:
    """Convert series to a list."""
    return x.to_list()


@dispatch
def cats(x: PdSeriesOrCat) -> "pd.Index":
    if isinstance(x, pd.Categorical):
        return x.categories

    return x.cat.categories


@dispatch
def cats(x: PlExpr) -> PlExpr:
    """Return the levels of a categorical series as an expression."""
    return _expr_map_batches(x, cats)


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


@dispatch
def factor(
    x: PdSeriesOrCat, levels: "pd.Index | list[str] | None" = None, ordered: bool | None = None
) -> PdSeriesOrCat:
    if levels is None:
        levels = x.unique()
    return pd.Categorical(x, categories=levels, ordered=ordered)


@dispatch
def factor(x: PlSeries, levels: PlSeries | None = None) -> PlSeries:
    """Create a factor, a categorical series whose level order can be specified."""

    if levels is None:
        levels = _levels(x)
    elif levels.dtype == pl.Categorical or levels.dtype == pl.Enum:
        levels = levels.cast(pl.String)

    return x.cast(pl.Enum(levels))
