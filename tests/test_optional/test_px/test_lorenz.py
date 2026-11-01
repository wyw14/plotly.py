import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest


def _curve(fig):
    """The Lorenz curve trace (first trace) of a `px.lorenz` figure."""
    assert fig.data[0].name == "Lorenz curve"
    return np.asarray(fig.data[0].x), np.asarray(fig.data[0].y)


def _equality_line(fig):
    assert fig.data[1].name == "Perfect equality"
    return np.asarray(fig.data[1].x), np.asarray(fig.data[1].y)


def _gini(fig):
    text = fig.layout.annotations[0].text
    match = re.search(r"Gini coefficient:\s*([0-9.]+)", text)
    assert match is not None, text
    return float(match.group(1))


def _trapz_gini(x, y):
    # Independent reference computation of the trapezoidal Gini coefficient
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return 1 - 2 * trapezoid(y, x)


def test_lorenz_returns_plain_figure():
    fig = px.lorenz(x=[1, 2, 3, 4])
    assert isinstance(fig, go.Figure)
    assert not isinstance(fig, go.FigureWidget)
    assert len(fig.data) == 2
    assert {trace.type for trace in fig.data} == {"scatter"}
    assert fig.data[0].mode == "lines"
    assert fig.data[1].mode == "lines"
    assert fig.data[1].line.dash == "dash"


def test_lorenz_endpoints_and_equality_line():
    fig = px.lorenz(x=[1, 2, 3, 4])
    x, y = _curve(fig)
    assert x[0] == 0.0 and y[0] == 0.0
    assert x[-1] == 1.0 and y[-1] == 1.0
    assert np.all(np.diff(x) >= 0) and np.all(np.diff(y) >= 0)
    assert np.all(y <= x + 1e-12)
    line_x, line_y = _equality_line(fig)
    assert list(line_x) == [0.0, 1.0]
    assert list(line_y) == [0.0, 1.0]
    assert fig.layout.xaxis.range == (0.0, 1.0)
    assert fig.layout.yaxis.range == (0.0, 1.0)


def test_lorenz_unweighted_cumulative_shares():
    fig = px.lorenz(x=[1, 2, 3, 4])
    x, y = _curve(fig)
    np.testing.assert_allclose(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(y, [0.0, 0.1, 0.3, 0.6, 1.0])
    # Without weights every observation has weight 1
    np.testing.assert_allclose(np.diff(x), 0.25)
    assert _gini(fig) == pytest.approx(0.25)
    assert _gini(fig) == pytest.approx(_trapz_gini(x, y))


def test_lorenz_identical_values_are_perfectly_equal():
    fig = px.lorenz(x=[7.5, 7.5, 7.5, 7.5])
    x, y = _curve(fig)
    np.testing.assert_allclose(x, y)
    assert _gini(fig) == 0.0
    np.testing.assert_allclose(y, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_lorenz_duplicate_values():
    fig = px.lorenz(x=[2, 1, 2, 1])
    x, y = _curve(fig)
    np.testing.assert_allclose(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(y, [0.0, 1 / 6, 2 / 6, 4 / 6, 1.0])
    assert _gini(fig) == pytest.approx(1 / 6, abs=1e-4)
    # Duplicate values may be given in any row order: the trapezoidal Gini
    # coefficient must be identical.
    shuffled = px.lorenz(x=[1, 2, 1, 2])
    assert _gini(shuffled) == pytest.approx(_gini(fig))


def test_lorenz_weighted_samples():
    values = [1.0, 2.0, 4.0]
    weights = [1.0, 1.0, 2.0]
    fig = px.lorenz(x=values, weights=weights)
    x, y = _curve(fig)
    np.testing.assert_allclose(x, [0.0, 0.25, 0.5, 1.0])
    np.testing.assert_allclose(y, [0.0, 1 / 11, 3 / 11, 1.0])
    assert _gini(fig) == pytest.approx(0.25)

    # A weighted observation is equivalent to repeating that observation:
    # weights [1, 1, 2] on [1, 2, 4] <-> unweighted [1, 2, 4, 4]
    repeated = px.lorenz(x=[1.0, 2.0, 4.0, 4.0])
    assert _gini(repeated) == pytest.approx(_gini(fig))
    # The area under both curves is the same, even though the weighted
    # representation merges the duplicated group into one segment.
    rx, ry = _curve(repeated)
    assert _trapz_gini(rx, ry) == pytest.approx(_trapz_gini(x, y))


def test_lorenz_missing_values_are_dropped_pairwise():
    df = pd.DataFrame(
        {
            "income": [1.0, np.nan, 3.0, 4.0],
            "w": [1.0, 2.0, 1.0, np.nan],
        }
    )
    fig = px.lorenz(df, x="income", weights="w")
    x, y = _curve(fig)
    # Rows 2 (missing value) and 4 (missing weight) are dropped pairwise,
    # leaving the unweighted pair (1, 3).
    np.testing.assert_allclose(x, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(y, [0.0, 0.25, 1.0])
    assert _gini(fig) == pytest.approx(0.25)


def test_lorenz_missing_values_without_weights():
    fig = px.lorenz(x=[1.0, np.nan, 3.0, 4.0])
    x, y = _curve(fig)
    np.testing.assert_allclose(x, [0.0, 1 / 3, 2 / 3, 1.0])
    np.testing.assert_allclose(y, [0.0, 1 / 8, 4 / 8, 1.0])
    assert _gini(fig) == pytest.approx(_trapz_gini(x, y))


def test_lorenz_extreme_inequality():
    n = 4
    # All the value is held by a single observation
    fig = px.lorenz(x=[0.0, 0.0, 0.0, 1.0])
    x, y = _curve(fig)
    np.testing.assert_allclose(y[:-1], 0.0)
    # Discrete Gini coefficient for maximal inequality is 1 - 1/n
    assert _gini(fig) == pytest.approx(1 - 1 / n)

    # Irrelevance of the value scale, including very large values
    large = px.lorenz(x=[0.0, 0.0, 0.0, 1e15])
    assert _gini(large) == pytest.approx(1 - 1 / n)

    # Weighted extreme inequality: a tiny fraction of the population holds all
    weighted = px.lorenz(x=[0.0, 0.0, 1.0], weights=[1e6, 1e6, 1.0])
    assert 0.0 <= _gini(weighted) <= 1.0
    assert _gini(weighted) > 0.999


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(x=[1, -2, 3]), "non-negative"),
        (dict(x=[1, np.inf, 3]), "finite"),
        (dict(x=[1, -np.inf, 3]), "finite"),
        (dict(x=[1, 2, 3], weights=[1, 0, 2]), "strictly positive"),
        (dict(x=[1, 2, 3], weights=[1, -2, 2]), "strictly positive"),
        (dict(x=[1, 2, 3], weights=[1, np.inf, 2]), "finite"),
        (dict(x=[0, 0, 0]), "total weighted value"),
        (dict(x=[1, 2, 3], weights=[1, 2, 3, 4]), "same length"),
        (dict(x=[np.nan, np.nan]), "No valid observations"),
        (dict(x=[1], weights=[np.nan]), "No valid observations"),
        (dict(), "Must pass"),
        (dict(x=["a", "b", "c"]), "numeric"),
    ],
)
def test_lorenz_invalid_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        px.lorenz(**kwargs)


def test_lorenz_missing_column_reference():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="not the name of a column"):
        px.lorenz(df, x="b")
    with pytest.raises(ValueError, match="column reference"):
        px.lorenz(x="a")


def test_lorenz_input_is_not_modified():
    df = pd.DataFrame({"income": [3.0, 1.0, 2.0, np.nan], "w": [2.0, 1.0, 3.0, 1.0]})
    df_before = df.copy(deep=True)
    values = np.array([3.0, 1.0, 2.0])
    values_before = values.copy()
    weights = np.array([2.0, 1.0, 3.0])
    weights_before = weights.copy()

    px.lorenz(df, x="income", weights="w")
    px.lorenz(x=values, weights=weights)

    pd.testing.assert_frame_equal(df, df_before)
    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(weights, weights_before)


def test_lorenz_gini_annotation_is_labeled():
    fig = px.lorenz(x=[1, 2, 3, 4])
    annotation = fig.layout.annotations[0]
    assert annotation.xref == "paper" and annotation.yref == "paper"
    assert annotation.showarrow is False
    assert re.fullmatch(r"Gini coefficient: 0\.2500", annotation.text)


def test_lorenz_hover_shows_population_and_value_shares():
    fig = px.lorenz(x=[1, 2, 3, 4])
    hovertemplate = fig.data[0].hovertemplate
    assert "%{x:.1%}" in hovertemplate
    assert "%{y:.1%}" in hovertemplate
    assert "Cumulative share of population" in hovertemplate
    assert "Cumulative share of value" in hovertemplate


def test_lorenz_labels_title_and_subtitle():
    fig = px.lorenz(
        x=[1, 2, 3, 4],
        labels={"x": "Bottom share of people", "y": "Share of income"},
        title="Income inequality",
        subtitle="Trapezoidal Gini",
    )
    assert fig.layout.xaxis.title.text == "Bottom share of people"
    assert fig.layout.yaxis.title.text == "Share of income"
    assert fig.layout.title.text == "Income inequality"
    assert fig.layout.title.subtitle.text == "Trapezoidal Gini"


def test_lorenz_value_label_from_column_or_series():
    df = pd.DataFrame({"income": [1, 2, 3, 4]})
    fig = px.lorenz(df, x="income")
    assert fig.layout.yaxis.title.text == "Cumulative share of income"

    series = pd.Series([1, 2, 3, 4], name="wealth")
    fig = px.lorenz(x=series)
    assert fig.layout.yaxis.title.text == "Cumulative share of wealth"


def test_lorenz_dataframe_backends(constructor):
    df = constructor({"income": [1.0, 2.0, None, 4.0], "w": [1.0, 1.0, 1.0, 1.0]})
    fig = px.lorenz(df, x="income", weights="w")
    x, y = _curve(fig)
    np.testing.assert_allclose(x, [0.0, 1 / 3, 2 / 3, 1.0])
    np.testing.assert_allclose(y, [0.0, 1 / 7, 3 / 7, 1.0])
    # The annotation displays 4 decimals, so compare against that precision
    assert _gini(fig) == pytest.approx(_trapz_gini(x, y), abs=1e-4)


def test_lorenz_positional_integer_column_reference():
    df = pd.DataFrame([[1, 9], [2, 8], [3, 7]])
    fig = px.lorenz(df, x=0, weights=1)
    x, y = _curve(fig)
    np.testing.assert_allclose(x, [0.0, 9 / 24, 17 / 24, 1.0])
    np.testing.assert_allclose(y, [0.0, 9 / 46, 25 / 46, 1.0])
    assert fig.layout.yaxis.title.text == "Cumulative share of 0"


def test_lorenz_docstring_documents_parameters():
    doc = px.lorenz.__doc__
    assert "Lorenz curve" in doc
    for param in ["data_frame", "x", "weights", "labels", "title", "template"]:
        assert re.search(r"\n\s*%s:" % param, doc)
    assert "(documentation missing from map)" not in doc
