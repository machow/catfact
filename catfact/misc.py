from __future__ import annotations

import polars as pl

from ._databackend import PlSeries
from ddispatch import dispatch
from typing import Any

# For each fct function, need to handle these cases:
#   - categorical: methods like replace not available.
#   - non-categorical: methods like replace available, but levels not calculated yet.
# TODO: fct_shuffle, fct_relevel(after=...), fct_drop, fct_c
# TODO: note cannot store NA in levels


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


@dispatch
def factor(x: PlSeries, levels: PlSeries | None = None) -> PlSeries:
    """Create a factor, a categorical series whose level order can be specified."""

    if levels is None:
        levels = _levels(x)
    elif levels.dtype == pl.Categorical or levels.dtype == pl.Enum:
        levels = levels.cast(pl.String)

    return x.cast(pl.Enum(levels))
