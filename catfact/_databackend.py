from databackend import AbstractBackend
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import polars  # noqa
    import polars as pl

    PlFrame = pl.DataFrame
    PlSeries = pl.Series
else:
    import polars  # noqa

    class PlFrame(AbstractBackend):
        _backends = [("polars", "DataFrame")]

    class PlSeries(AbstractBackend):
        _backends = [("polars", "Series")]
