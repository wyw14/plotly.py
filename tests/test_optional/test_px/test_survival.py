import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose


def _line_traces(fig):
    return [trace for trace in fig.data if trace.mode == "lines"]


def _marker_traces(fig):
    return [trace for trace in fig.data if trace.mode == "markers"]


def test_survival_returns_plain_figure(constructor):
    df = constructor({"t": [1.0, 2.0], "e": [1, 0]})
    fig = px.survival(df, x="t", event="e")
    assert isinstance(fig, go.Figure)
    assert all(isinstance(trace, go.Scatter) for trace in fig.data)
    assert fig.layout.yaxis.title.text == "Survival probability"
    assert fig.layout.xaxis.title.text == "t"


def test_survival_basic_kaplan_meier(constructor):
    # t=1: 1 event + 1 censor of 4 at risk -> S = 3/4
    # t=2: 1 event of 2 at risk      -> S = 3/8
    # t=3: 1 censor of 1 at risk      -> S unchanged
    df = constructor({"t": [1.0, 1.0, 2.0, 3.0], "e": [1, 0, 1, 0]})
    fig = px.survival(df, x="t", event="e")
    (line,) = _line_traces(fig)
    assert line.line.shape == "hv"
    assert list(line.x) == [0.0, 1.0, 2.0, 3.0]
    assert_allclose(line.y, [1.0, 0.75, 0.375, 0.375])
    customdata = np.asarray(line.customdata)
    # columns: at risk, events, censored, survival probability
    assert customdata.tolist() == [
        [4, 0, 0, 1.0],
        [4, 1, 1, 0.75],
        [2, 1, 0, 0.375],
        [1, 0, 1, 0.375],
    ]


def test_survival_censor_marks(constructor):
    df = constructor({"t": [1.0, 1.0, 2.0, 3.0], "e": [1, 0, 1, 0]})
    fig = px.survival(df, x="t", event="e")
    (markers,) = _marker_traces(fig)
    assert markers.showlegend is False
    assert markers.marker.symbol == "cross"
    assert list(markers.x) == [1.0, 3.0]
    assert_allclose(markers.y, [0.75, 0.375])
    # the marker at a time with both an event and a censoring reports both
    assert np.asarray(markers.customdata)[0].tolist() == [4, 1, 1, 0.75]


def test_survival_ties_share_the_risk_set(constructor):
    # 5 subjects: at t=2 there are 2 events and 2 censorings, all 5 at risk;
    # the last subject is censored later.
    df = constructor({"t": [2.0, 2.0, 2.0, 2.0, 5.0], "e": [1, 1, 0, 0, 0]})
    fig = px.survival(df, x="t", event="e")
    (line,) = _line_traces(fig)
    customdata = np.asarray(line.customdata)
    assert customdata[1].tolist() == [5, 2, 2, 0.6]
    # censorings at t=2 do not lower the curve, but those subjects are gone
    # from the risk set at t=5
    assert customdata[2].tolist() == [1, 0, 1, 0.6]
    assert_allclose(line.y, [1.0, 0.6, 0.6])


def test_survival_all_censored_group_stays_at_one(constructor):
    df = constructor({"t": [1.0, 2.0, 3.0], "e": [0, 0, 0]})
    fig = px.survival(df, x="t", event="e")
    (line,) = _line_traces(fig)
    assert_allclose(line.y, [1.0, 1.0, 1.0, 1.0])
    (markers,) = _marker_traces(fig)
    assert list(markers.x) == [1.0, 2.0, 3.0]
    assert_allclose(markers.y, [1.0, 1.0, 1.0])


def test_survival_event_and_censor_at_time_zero(constructor):
    df = constructor({"t": [0.0, 0.0, 1.0, 2.0], "e": [1, 0, 1, 0]})
    fig = px.survival(df, x="t", event="e")
    (line,) = _line_traces(fig)
    assert_allclose(line.y, [1.0, 0.75, 0.375, 0.375])
    assert list(line.x) == [0.0, 0.0, 1.0, 2.0]


def test_survival_groups_category_order_and_colors(constructor):
    df = constructor(
        {
            "t": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "e": [1, 1, 0, 0, 0, 0],
            "g": ["b", "b", "b", "a", "a", "a"],
        }
    )
    fig = px.survival(
        df,
        x="t",
        event="e",
        color="g",
        category_orders={"g": ["a", "b"]},
        color_discrete_map={"a": "red", "b": "blue"},
    )
    lines = _line_traces(fig)
    assert [trace.name for trace in lines] == ["a", "b"]
    assert {trace.name: trace.line.color for trace in lines} == {
        "a": "red",
        "b": "blue",
    }
    assert all(trace.legendgroup == trace.name for trace in lines)
    assert fig.layout.legend.title.text == "g"
    # markers and steps of a group are tied together in the legend
    assert all(trace.legendgroup for trace in _marker_traces(fig))


def test_survival_group_order_defaults_to_first_appearance(constructor):
    df = constructor(
        {
            "t": [1.0, 1.0, 1.0],
            "e": [0, 0, 0],
            "g": ["z", "a", "m"],
        }
    )
    fig = px.survival(df, x="t", event="e", color="g")
    assert [trace.name for trace in _line_traces(fig)] == ["z", "a", "m"]


def test_survival_group_remaining_order_appended(constructor):
    df = constructor(
        {
            "t": [1.0, 2.0, 3.0, 4.0],
            "e": [0, 0, 0, 0],
            "g": ["b", "c", "a", "d"],
        }
    )
    fig = px.survival(
        df, x="t", event="e", color="g", category_orders={"g": ["a", "b"]}
    )
    assert [trace.name for trace in _line_traces(fig)] == ["a", "b", "c", "d"]


def test_survival_color_discrete_sequence(constructor):
    df = constructor(
        {"t": [1.0, 2.0, 1.0, 2.0], "e": [0, 0, 0, 0], "g": ["a", "a", "b", "b"]}
    )
    fig = px.survival(
        df, x="t", event="e", color="g", color_discrete_sequence=["black", "white"]
    )
    colors = [trace.line.color for trace in _line_traces(fig)]
    assert colors == ["black", "white"]


def test_survival_arrays_without_data_frame_and_bool_events():
    fig = px.survival(x=[1, 2, 3], event=[True, False, False])
    (line,) = _line_traces(fig)
    assert_allclose(line.y, [1.0, 2 / 3, 2 / 3, 2 / 3], atol=1e-12)
    (markers,) = _marker_traces(fig)
    assert list(markers.x) == [2.0, 3.0]
    # a single, ungrouped curve does not create a legend entry
    assert line.showlegend is False
    assert fig.layout.legend.title.text is None


def test_survival_hovertemplate_contents(constructor):
    df = constructor({"t": [1.0, 2.0], "e": [1, 0], "g": ["a", "a"]})
    fig = px.survival(
        df,
        x="t",
        event="e",
        color="g",
        labels={"t": "Follow-up (months)"},
    )
    (line,) = _line_traces(fig)
    template = line.hovertemplate
    assert "<b>a</b>" in template
    assert "Follow-up (months)=%{x}" in template
    assert "At risk=%{customdata[0]}" in template
    assert "Events=%{customdata[1]}" in template
    assert "Censored=%{customdata[2]}" in template
    assert "Survival probability=%{customdata[3]:.3f}" in template


def test_survival_labels_title_and_size(constructor):
    df = constructor({"t": [1.0], "e": [1], "g": ["a"]})
    fig = px.survival(
        df,
        x="t",
        event="e",
        color="g",
        labels={"t": "Time", "g": "Arm"},
        title="Overall survival",
        subtitle="ITT population",
        width=640,
        height=480,
    )
    assert fig.layout.xaxis.title.text == "Time"
    assert fig.layout.legend.title.text == "Arm"
    assert fig.layout.title.text == "Overall survival"
    assert fig.layout.title.subtitle.text == "ITT population"
    assert fig.layout.width == 640
    assert fig.layout.height == 480


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(x=[1.0, -1.0], event=[1, 0]), "non-negative"),
        (dict(x=[1.0, np.nan], event=[1, 0]), "finite"),
        (dict(x=[1.0, np.inf], event=[1, 0]), "finite"),
        (dict(x=[1.0, -np.inf], event=[1, 0]), "finite"),
        (dict(x=[1, 2], event=[1, 2]), "event indicators"),
        (dict(x=[1, 2], event=[1.0, 0.5]), "event indicators"),
        (dict(x=[1, 2], event=[1, "yes"]), "event indicators"),
        (dict(x=["a", "b"], event=[1, 0]), "numeric"),
    ],
)
def test_survival_invalid_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        px.survival(**kwargs)


def test_survival_invalid_event_missing_value(constructor):
    df = constructor({"t": [1.0, 2.0], "e": [1, None]})
    with pytest.raises(ValueError, match="event indicators|missing values"):
        px.survival(df, x="t", event="e")


def test_survival_missing_arguments():
    with pytest.raises(ValueError, match="`x`"):
        px.survival(event=[1, 0])
    with pytest.raises(ValueError, match="`event`"):
        px.survival(x=[1, 2])


def test_survival_non_numeric_time_column(constructor):
    df = constructor({"t": ["a", "b"], "e": [1, 0]})
    with pytest.raises(ValueError, match="numeric observation times"):
        px.survival(df, x="t", event="e")
