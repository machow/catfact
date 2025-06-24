import pandas as pd
import catfact as fct
from functools import wraps


def feed(func):
    """Decorator to register a function as a Polars Series namespace method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self._s, *args, **kwargs)

    return wrapper


@pd.api.extensions.register_series_accessor("fct")
class CatFactAccessor:
    def __init__(self, s):
        self._s = s

    cats = feed(fct.cats)
    factor = feed(fct.factor)
    infreq = feed(fct.infreq)
    inorder = feed(fct.inorder)
    inseq = feed(fct.inseq)
    relevel = feed(fct.relevel)
    reorder = feed(fct.reorder)
    rev = feed(fct.rev)
    lump_n = feed(fct.lump_n)
    lump_prop = feed(fct.lump_prop)
    lump_min = feed(fct.lump_min)
    lump_lowfreq = feed(fct.lump_lowfreq)
    collapse = feed(fct.collapse)
    recode = feed(fct.recode)
