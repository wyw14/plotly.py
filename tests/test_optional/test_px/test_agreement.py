import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from .conftest import (
    pandas_constructor,
    polars_eager_constructor,
    pyarrow_table_constructor,
)


@pytest.fixture(
    params=[
        pandas_constructor,
        polars_eager_constructor,
        pyarrow_table_constructor,
    ]
)
def df_constructor(request):
    return request.param


def _sample_data():
    # Differences: -0.2, 0.2, -0.5, 0.2 ; one row is missing in each column.
    return dict(
        a=[1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        b=[1.2, 1.8, 3.0, 4.5, 4.8, np.nan],
        id=["s1", "s2", "s3", "s4", "s5", "s6"],
        group=["x", "y", "x", "y", "x", "y"],
    )


def _complete_pairs(data, col_a="a", col_b="b"):
    frame = pd.DataFrame(data)
    complete = frame[[col_a, col_b]].notna().all(axis=1)
    a = frame.loc[complete, col_a].to_numpy(dtype=float)
    b = frame.loc[complete, col_b].to_numpy(dtype=float)
    return a, b


def test_agreement_basic(df_constructor):
    data = _sample_data()
    df = df_constructor(data)
    a, b = _complete_pairs(data)
    expected_diff = a - b
    expected_mean = (a + b) / 2

    fig = px.agreement(df, x="a", y="b")

    assert isinstance(fig, go.Figure)
    # One scatter trace with one point per complete pair, in original order.
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.type == "scatter"
    assert trace.mode == "markers"
    np.testing.assert_allclose(np.asarray(trace.x), expected_mean)
    np.testing.assert_allclose(np.asarray(trace.y), expected_diff)
    assert len(trace.x) == 4

    # Axes titles follow the "<mean of>" / "<x> - <y>" convention.
    assert fig.layout.xaxis.title.text == "Mean of a and b"
    assert fig.layout.yaxis.title.text == "a - b"

    # Three horizontal reference lines: mean and mean +/- 1.96 * sample SD.
    assert len(fig.layout.shapes) == 3
    bias = expected_diff.mean()
    sd = expected_diff.std(ddof=1)
    expected_levels = sorted([bias - 1.96 * sd, bias, bias + 1.96 * sd])
    shape_levels = sorted(float(shape.y0) for shape in fig.layout.shapes)
    assert shape_levels == pytest.approx(expected_levels)
    for shape in fig.layout.shapes:
        assert shape.type == "line"
        assert shape.x0 == 0 and shape.x1 == 1
        assert shape.y0 == shape.y1


def test_agreement_positive_and_negative_bias():
    positive = pd.DataFrame(
        {"a": [10.0, 11.0, 12.0, 13.0], "b": [8.0, 9.0, 10.0, 11.0]}
    )
    fig_pos = px.agreement(positive, x="a", y="b")

    def is_solid(shape):
        return shape.line.dash in (None, "solid")

    mean_level = [float(s.y0) for s in fig_pos.layout.shapes if is_solid(s)][0]
    assert mean_level > 0
    assert "Mean difference: 2" in [a.text for a in fig_pos.layout.annotations]

    negative = pd.DataFrame(
        {"a": [8.0, 9.0, 10.0, 11.0], "b": [10.0, 11.0, 12.0, 13.0]}
    )
    fig_neg = px.agreement(negative, x="a", y="b")
    mean_level = [float(s.y0) for s in fig_neg.layout.shapes if is_solid(s)][0]
    assert mean_level < 0
    assert "Mean difference: -2" in [a.text for a in fig_neg.layout.annotations]


def test_agreement_missing_pairs_are_skipped_pairwise(df_constructor):
    data = _sample_data()
    df = df_constructor(data)
    df_before = nw.from_native(df, eager_only=True).to_pandas()
    a, b = _complete_pairs(data)

    fig = px.agreement(df, x="a", y="b")

    # Only rows with both values present survive, in the original row order
    # (complete rows are indices 0, 1, 3, 4).
    np.testing.assert_allclose(np.asarray(fig.data[0].y), [-0.2, 0.2, -0.5, 0.2])
    np.testing.assert_allclose(np.asarray(fig.data[0].x), [1.1, 1.9, 4.25, 4.9])
    np.testing.assert_allclose(np.asarray(fig.data[0].y), a - b)
    np.testing.assert_allclose(np.asarray(fig.data[0].x), (a + b) / 2)
    # The input dataframe was not modified.
    df_after = nw.from_native(df, eager_only=True).to_pandas()
    pd.testing.assert_frame_equal(df_before, df_after)


def test_agreement_repeated_means():
    # All four pairs share the exact same mean (11) but different differences;
    # every point must be plotted and the reference lines stay visible.
    df = pd.DataFrame({"a": [10.0, 11.0, 12.0, 13.0], "b": [12.0, 11.0, 10.0, 9.0]})
    fig = px.agreement(df, x="a", y="b")
    assert len(fig.data[0].x) == 4
    assert list(fig.data[0].x).count(11.0) == 4
    np.testing.assert_allclose(np.asarray(fig.data[0].y), [-2.0, 0.0, 2.0, 4.0])
    assert len(fig.layout.shapes) == 3


def test_agreement_identical_differences_coincident_limits():
    df = pd.DataFrame({"a": [3.0, 5.0, 8.0, 10.0], "b": [1.0, 3.0, 6.0, 8.0]})
    fig = px.agreement(df, x="a", y="b")
    levels = sorted(float(shape.y0) for shape in fig.layout.shapes)
    # All differences equal 2 -> SD of 0 -> all three lines coincide at 2.
    assert levels == pytest.approx([2.0, 2.0, 2.0])
    assert len(fig.data[0].x) == 4
    # All points are still drawn even though all means are distinct.
    np.testing.assert_allclose(np.asarray(fig.data[0].y), [2.0, 2.0, 2.0, 2.0])


@pytest.mark.parametrize(
    "a,b",
    [
        ([1.0], [2.0]),  # exactly one valid pair
        ([1.0, np.nan], [np.nan, 2.0]),  # zero complete pairs
        ([1.0, np.nan, 3.0], [np.nan, 2.0, np.nan]),
    ],
)
def test_agreement_requires_two_pairs(a, b):
    df = pd.DataFrame({"a": a, "b": b})
    with pytest.raises(ValueError, match="At least two complete pairs"):
        px.agreement(df, x="a", y="b")


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_agreement_rejects_non_finite(bad):
    df = pd.DataFrame({"a": [1.0, 2.0, bad, 4.0], "b": [1.1, 2.1, 3.1, 4.1]})
    with pytest.raises(ValueError, match="Non-finite"):
        px.agreement(df, x="a", y="b")
    # Non-finite value is reported even when the partner value is missing.
    df2 = pd.DataFrame({"a": [1.0, 2.0, bad, 4.0], "b": [1.1, 2.1, np.nan, 4.1]})
    with pytest.raises(ValueError, match="Non-finite"):
        px.agreement(df2, x="a", y="b")


def test_agreement_rejects_non_numeric():
    df = pd.DataFrame({"a": ["high", "low", "ok", "no"], "b": [1.1, 2.1, 3.1, 4.1]})
    with pytest.raises(ValueError, match="numeric"):
        px.agreement(df, x="a", y="b")


def test_agreement_missing_arguments():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Both 'x' and 'y'"):
        px.agreement(df, x="a")
    with pytest.raises(ValueError, match="not the name of a column"):
        px.agreement(df, x="a", y="missing")
    with pytest.raises(ValueError, match="same length"):
        px.agreement(x=[1.0, 2.0], y=[1.0, 2.0, 3.0])


def test_agreement_standard_deviation_ddof_one():
    # Hand-picked differences for which ddof=0 and ddof=1 give different bounds.
    df = pd.DataFrame({"a": [0.0, 1.0, 2.0, 5.0], "b": [0.0, 0.0, 0.0, 0.0]})
    diffs = df["a"].to_numpy() - df["b"].to_numpy()
    bias = diffs.mean()
    sd_n1 = diffs.std(ddof=1)
    sd_n0 = diffs.std(ddof=0)
    assert sd_n1 != pytest.approx(sd_n0)

    fig = px.agreement(df, x="a", y="b")
    levels = sorted(float(s.y0) for s in fig.layout.shapes)
    assert levels == pytest.approx(
        sorted([bias - 1.96 * sd_n1, bias, bias + 1.96 * sd_n1])
    )
    # The ddof=0 bound must not coincide with any plotted line.
    assert all(abs(bias + 1.96 * sd_n0 - v) > 1e-9 for v in levels)
    # Two-pair minimum also uses n-1: SD = |d1-d2| / sqrt(2).
    two = pd.DataFrame({"a": [0.0, 4.0], "b": [1.0, 1.0]})
    fig_two = px.agreement(two, x="a", y="b")
    two_levels = sorted(float(s.y0) for s in fig_two.layout.shapes)
    expected_sd = abs(-1.0 - 3.0) / np.sqrt(2)
    expected_bias = 1.0
    assert two_levels == pytest.approx(
        sorted(
            [
                expected_bias - 1.96 * expected_sd,
                expected_bias,
                expected_bias + 1.96 * expected_sd,
            ]
        )
    )


def test_agreement_hover_shows_raw_mean_and_difference(df_constructor):
    data = _sample_data()
    df = df_constructor(data)
    a, b = _complete_pairs(data)

    fig = px.agreement(
        df,
        x="a",
        y="b",
        hover_name="id",
        hover_data=["group"],
        labels={"a": "Method A", "b": "Method B"},
    )
    trace = fig.data[0]
    template = trace.hovertemplate

    # Both raw measurements, the mean and the difference are in the tooltip.
    assert "Method A=" in template
    assert "Method B=" in template
    assert "Mean of Method A and Method B=%{x" in template
    assert "Method A - Method B=%{y" in template
    # Extra hover field and hover name are forwarded.
    assert "group=" in template
    np.testing.assert_array_equal(np.asarray(trace.hovertext), ["s1", "s2", "s4", "s5"])
    # customdata carries raw a, raw b and the extra group column for valid rows.
    custom = np.asarray(trace.customdata, dtype=object)
    np.testing.assert_allclose(custom[:, 0].astype(float), a)
    np.testing.assert_allclose(custom[:, 1].astype(float), b)
    np.testing.assert_array_equal(custom[:, 2], ["x", "y", "y", "x"])
    # Axis labels follow the override.
    assert fig.layout.xaxis.title.text == "Mean of Method A and Method B"
    assert fig.layout.yaxis.title.text == "Method A - Method B"


def test_agreement_hover_data_dict_and_onboard_arrays(df_constructor):
    data = {"a": [1.0, 2.0, 3.0], "b": [1.5, 1.5, 1.5], "drop": [1, 2, 3]}
    df = df_constructor(data)
    extra = [10.0, 20.0, 30.0]
    fig = px.agreement(
        df,
        x="a",
        y="b",
        hover_data={"drop": False, "custom score": extra, "a": ":.2f"},
    )
    template = fig.data[0].hovertemplate
    assert "drop=" not in template
    assert "custom score=" in template
    np.testing.assert_allclose(
        np.asarray(fig.data[0].customdata, dtype=float)[:, -1], extra
    )
    # The mandatory four fields remain present even if the user hides a column.
    assert "a=%{customdata[0]" in template
    assert "b=%{customdata[1]" in template
    # The figure must serialize like any other plotly figure.
    go.Figure(fig.to_dict())
    assert len(pio.from_json(fig.to_json()).layout.shapes) == 3


def test_agreement_hover_data_not_mutated():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.5, 1.5, 1.5], "g": [1, 2, 3]})
    hover_data = {"g": True}
    hover_data_before = dict(hover_data)
    px.agreement(df, x="a", y="b", hover_data=hover_data)
    assert hover_data == hover_data_before


def test_agreement_array_inputs_without_dataframe():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.5, 2.5, 2.0, 5.0])
    fig = px.agreement(x=a, y=b, hover_name=["p", "q", "r", "s"])
    np.testing.assert_allclose(np.asarray(fig.data[0].x), (a + b) / 2)
    np.testing.assert_allclose(np.asarray(fig.data[0].y), a - b)
    np.testing.assert_array_equal(
        np.asarray(fig.data[0].hovertext), ["p", "q", "r", "s"]
    )
    assert isinstance(fig, go.Figure)


def test_agreement_returns_customizable_figure():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.5, 2.0, 3.5]})
    fig = px.agreement(df, x="a", y="b", title="Agreement", width=600, height=400)
    # A plain Figure that can be further customized like any other.
    fig.add_trace(go.Scatter(x=[1.5], y=[0.5], mode="markers", name="extra"))
    fig.update_layout(title=dict(text="Customized"))
    assert len(fig.data) == 2
    assert fig.layout.title.text == "Customized"
    assert fig.layout.width == 600
    assert fig.layout.height == 400


def test_agreement_series_references(df_constructor):
    data = _sample_data()
    df = nw.from_native(df_constructor(data), eager_only=True)
    fig = px.agreement(
        df.to_native(),
        x=df.get_column("a").to_native(),
        y=df.get_column("b").to_native(),
    )
    a, b = _complete_pairs(data)
    np.testing.assert_allclose(np.asarray(fig.data[0].y), a - b)
    np.testing.assert_allclose(np.asarray(fig.data[0].x), (a + b) / 2)


def test_agreement_integer_and_nullable_inputs():
    df = pd.DataFrame(
        {
            "a": pd.array([1, 2, pd.NA, 4], dtype="Int64"),
            "b": pd.array([2, 2, 3, 4], dtype="Int64"),
        }
    )
    fig = px.agreement(df, x="a", y="b")
    np.testing.assert_allclose(np.asarray(fig.data[0].y), [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(fig.data[0].x), [1.5, 2.0, 4.0])
