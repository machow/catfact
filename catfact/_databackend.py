from databackend import AbstractBackend
from typing import TYPE_CHECKING, TypeVar, Any


if TYPE_CHECKING:
    import polars  # noqa
    import pandas as pd
    import polars as pl

    PlFrame = pl.DataFrame
    PlSeries = pl.Series
    PlExpr = pl.Expr
    PdSeries = pd.Series
    PdSeriesOrCat = TypeVar("PdSeriesOrCat", pd.Series[Any], pd.Categorical)
    PdFrame = pd.DataFrame

else:
    import polars  # noqa

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
