from databackend import AbstractBackend
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import polars as pl

    PlFrame = pl.DataFrame
    PlSeries = pl.Series
else:
    class PlFrame(AbstractBackend):
        _backends = [("polars", "DataFrame")]

    class PlSeries(AbstractBackend):
        _backends = [("polars", "Series")]