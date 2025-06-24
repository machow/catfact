from databackend import AbstractBackend
from typing import TYPE_CHECKING, TypeVar, Any

# lazy import class ----

import importlib
from types import ModuleType


class LazyImport(ModuleType):
    """Lazily represents an imported module."""

    def __init__(self, name: str):
        self.__name = name
        self.__mod: ModuleType | None = None

    def __getattr__(self, name: str):
        # TODO: if we updated __dict__ with __mod.__dict__, we could often skip
        # this getattr call. However, the two __dict__ objects could go out of sync,
        # if the original module updates anything in its globals (not uncommon).
        if self.__mod is None:
            self.__mod = self.__import()

        return getattr(self.__mod, name)

    def __import(self) -> ModuleType:
        mod = importlib.import_module(self.__name)
        return mod


# main ----

if TYPE_CHECKING:
    import polars  # noqa
    import pandas  # noqa
    import pandas as pd
    import polars as pl

    PlFrame = pl.DataFrame
    PlSeries = pl.Series
    PlExpr = pl.Expr
    PdSeries = pd.Series
    PdSeriesOrCat = TypeVar("PdSeriesOrCat", pd.Series[Any], pd.Categorical)
    PdFrame = pd.DataFrame

else:
    polars = LazyImport("polars")
    pandas = LazyImport("pandas")

    class PlFrame(AbstractBackend):
        _backends = [("polars", "DataFrame")]

    class PlSeries(AbstractBackend):
        _backends = [("polars", "Series")]

    class PlExpr(AbstractBackend):
        _backends = [("polars", "Expr")]

    class PdSeries(AbstractBackend):
        _backends = [("pandas", "Series")]

    class PdSeriesOrCat(AbstractBackend):
        _backends = [("pandas", "Series"), ("pandas", "Categorical")]
