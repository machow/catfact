from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

__all__ = (
    "gdp",
    "ratings_messy",
    "ratings",
    "wheels",
)


class ExampleData:
    _path: str

    def __init__(self, module: str, fname: str):
        self._path = str(files(module) / fname)

    def to_polars(self) -> pl.DataFrame:
        """Load the example data as a Polars DataFrame."""
        import polars as pl

        return pl.read_csv(self._path, infer_schema_length=None)

    def to_pandas(self) -> pd.DataFrame:
        """Load the example data as a Pandas DataFrame."""
        import pandas as pd

        return pd.read_csv(self._path)


gdp = ExampleData("catfact.data", "gdp.csv")
"""Gross Domestic Product (GDP) by country.

This data has the following columns:

* `year`: The year of the GDP data.
* `country`: The country for which the GDP is reported.
* `gdp`: The GDP value in current US dollars.

It was obtained from the World Bank and is available at
https://data.worldbank.org/indicator/NY.GDP.MKTP.CD.
"""

ratings_messy = ExampleData("catfact.data", "ratings_messy.csv")
"""Messy ratings data.

This data has the following columns:

* `rating`: one of very bad, bad, neutral, good, very good.

Moreover, some of the neutral ratings were miscoded as NEUTRAL.

"""

ratings = ExampleData("catfact.data", "ratings.csv")
"""Cleaned up ratings data.

This data is the ratings_messy data with the bad NEUTRAL values converted to neutral.
"""

starwars = ExampleData("catfact.data", "starwars.csv")
"""Star Wars character data.

This data has the following columns:

* `name`: The name of the character
* `eye_color`: The eye color of the character
* `species`: The species of the character

It comes from the dplyr library (https://dplyr.tidyverse.org/),
which fetched the original data from the Star Wars API (https://swapi.py4e.com/).
"""
