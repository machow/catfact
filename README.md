# catfact


Categorical wrangling for Python. Supports both Polars and Pandas.
Enables categorical and ordinal scales in plotting tools like Plotnine.

catfact addresses some common challenges when working categorical data.
Categorical data is useful when you want to display your data in a
specific way, like alphabetical, most frequent first, or along a scale.
It is a port of the popular R package forcats.

## Installation

``` bash
pip install catfact
```

## Basic example

``` python
import polars as pl
import catfact as fct
from catfact.polars.data import starwars

(
    starwars
    .group_by("eye_color")
    .agg(pl.len())
    .sort("len", descending=True)
)
```

<div>
<small>shape: (15, 2)</small>

| eye_color       | len |
|-----------------|-----|
| str             | u32 |
| "brown"         | 21  |
| "blue"          | 19  |
| "yellow"        | 11  |
| "black"         | 10  |
| "orange"        | 8   |
| …               | …   |
| "white"         | 1   |
| "pink"          | 1   |
| "blue-gray"     | 1   |
| "green, yellow" | 1   |
| "dark"          | 1   |

</div>

``` python
from plotnine import ggplot, aes, geom_bar, coord_flip

(
    ggplot(starwars, aes("eye_color"))
    + geom_bar()
    + coord_flip()
)
```

<img src="README_files/figure-commonmark/cell-4-output-1.png"
width="640" height="480" />

``` python
(
    starwars
    .with_columns(
        fct.infreq(pl.col("eye_color"))
    )
    >> ggplot(aes("eye_color"))
    + geom_bar()
    + coord_flip()
)
```

<img src="README_files/figure-commonmark/cell-5-output-1.png"
width="640" height="480" />
