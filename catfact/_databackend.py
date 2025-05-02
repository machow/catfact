from databackend import AbstractBackend
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import polars  # noqa
    import polars as pl

    PlFrame = pl.DataFrame
    PlSeries = pl.Series
    PlExpr = pl.Expr
else:
    import polars  # noqa

    class PlFrame(AbstractBackend):
        _backends = [("polars", "DataFrame")]

    class PlSeries(AbstractBackend):
        _backends = [("polars", "Series")]

    class PlExpr(AbstractBackend):
        _backends = [("polars", "Expr")]
