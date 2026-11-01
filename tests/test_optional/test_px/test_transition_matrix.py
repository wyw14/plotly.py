import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest


def _matrix(fig):
    return np.asarray(fig.data[0].z, dtype=float)


def _customdata(fig):
    return np.asarray(fig.data[0].customdata, dtype=float)


def test_transition_matrix_returns_heatmap_figure(constructor):
    df = constructor(
        {"state": ["A", "B", "C"], "step": [1, 2, 3], "id": ["a", "a", "a"]}
    )
    fig = px.transition_matrix(df, state="state")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    # The state list is shared by both axes, in first-appearance order.
    assert tuple(fig.data[0].x) == ("A", "B", "C")
    assert tuple(fig.data[0].y) == ("A", "B", "C")
    assert fig.layout.xaxis.categoryorder == "array"
    assert list(fig.layout.xaxis.categoryarray) == ["A", "B", "C"]


def test_basic_counts_and_no_end_transition(constructor):
    # One sequence of 3 consecutive events gives exactly 2 transitions and
    # no fabricated transition out of the final state.
    df = constructor({"state": ["A", "B", "C"], "step": [1, 2, 3]})
    fig = px.transition_matrix(df, state="state", sort="step")
    z = _matrix(fig)
    states = list(fig.data[0].x)
    i = {s: k for k, s in enumerate(states)}
    assert z[i["A"], i["B"]] == 1
    assert z[i["B"], i["C"]] == 1
    assert z.sum() == 2
    # C is the final event: it has zero out-degree.
    assert z[i["C"], :].sum() == 0


def test_sequences_are_not_connected(constructor):
    # Sort values interleave between the two sequences. If the sequences were
    # wrongly treated as one, X->B and A->Y cross-connections would appear.
    df = constructor(
        {
            "sequence": ["s1", "s2", "s1", "s2"],
            "step": [2, 1, 4, 3],
            "state": ["X", "A", "Y", "B"],
        }
    )
    fig = px.transition_matrix(df, state="state", sequence="sequence", sort="step")
    z = _matrix(fig)
    states = list(fig.data[0].x)
    i = {s: k for k, s in enumerate(states)}
    # Within-sequence transitions only: s1 X->Y, s2 A->B.
    assert z[i["X"], i["Y"]] == 1
    assert z[i["A"], i["B"]] == 1
    # No cross-sequence transitions, whatever the global sort order is.
    assert z[i["X"], i["B"]] == 0
    assert z[i["A"], i["Y"]] == 0
    assert z[i["X"], i["A"]] == 0
    assert z.sum() == 2


def test_sequence_boundaries_without_sort(constructor):
    # No sort column: input order is used, independently per sequence.
    df = constructor(
        {
            "sequence": ["b", "a", "b", "a", "b"],
            "state": ["B1", "A1", "B2", "A2", "B3"],
        }
    )
    fig = px.transition_matrix(df, state="state", sequence="sequence")
    z = _matrix(fig)
    states = list(fig.data[0].x)
    i = {s: k for k, s in enumerate(states)}
    assert z[i["A1"], i["A2"]] == 1
    assert z[i["B1"], i["B2"]] == 1
    assert z[i["B2"], i["B3"]] == 1
    # The rows happen to interleave in the input, but sequences never connect.
    assert z[i["B1"], i["A1"]] == 0
    assert z[i["A2"], i["B3"]] == 0
    assert z.sum() == 3


def test_on_missing_break_vs_skip(constructor):
    df = constructor(
        {
            "sequence": ["s"] * 5,
            "step": [1, 2, 3, 4, 5],
            "state": ["A", "B", None, "C", "D"],
        }
    )
    break_fig = px.transition_matrix(
        df,
        state="state",
        sequence="sequence",
        sort="step",
        on_missing="break",
    )
    break_z = _matrix(break_fig)
    states = list(break_fig.data[0].x)
    i = {s: k for k, s in enumerate(states)}
    # The missing event breaks the sequence: B->C is not counted.
    assert break_z[i["A"], i["B"]] == 1
    assert break_z[i["B"], i["C"]] == 0
    assert break_z[i["C"], i["D"]] == 1
    assert break_z.sum() == 2

    skip_fig = px.transition_matrix(
        df,
        state="state",
        sequence="sequence",
        sort="step",
        on_missing="skip",
    )
    skip_z = _matrix(skip_fig)
    # The missing event is ignored, so B and C become consecutive.
    assert skip_z[i["A"], i["B"]] == 1
    assert skip_z[i["B"], i["C"]] == 1
    assert skip_z[i["C"], i["D"]] == 1
    assert skip_z.sum() == 3


def test_on_missing_default_is_break(constructor):
    df = constructor({"state": ["A", None, "B"]})
    fig = px.transition_matrix(df, state="state")
    assert _matrix(fig).sum() == 0


def test_missing_at_sequence_endpoints(constructor):
    df = constructor(
        {
            "sequence": ["s1", "s1", "s2", "s2"],
            "state": ["A", None, None, "B"],
        }
    )
    fig = px.transition_matrix(df, state="state", sequence="sequence")
    # No pair of consecutive valid events anywhere.
    assert _matrix(fig).sum() == 0
    skip_fig = px.transition_matrix(
        df, state="state", sequence="sequence", on_missing="skip"
    )
    # Each sequence only contains a single valid event, still no transition.
    assert _matrix(skip_fig).sum() == 0


def test_normalize_row(constructor):
    # A has 2 outgoing transitions (A->B, A->C), C is a sink.
    df = constructor(
        {
            "sequence": ["s1", "s1", "s1", "s2", "s2"],
            "state": ["A", "B", "C", "A", "C"],
        }
    )
    counts = _matrix(px.transition_matrix(df, state="state", sequence="sequence"))
    fig = px.transition_matrix(df, state="state", sequence="sequence", normalize="row")
    z = _matrix(fig)
    states = list(fig.data[0].x)
    i = {s: k for k, s in enumerate(states)}
    assert z[i["A"], i["B"]] == pytest.approx(0.5)
    assert z[i["A"], i["C"]] == pytest.approx(0.5)
    # Every state with outgoing transitions has a row summing to 1.
    assert z[i["A"], :].sum() == pytest.approx(1)
    assert z[i["B"], :].sum() == pytest.approx(1)
    # Zero out-degree states get an entirely zero probability row.
    assert z[i["C"], :].sum() == 0
    assert np.all(z[i["C"], :] == 0)
    # Probabilities are the counts divided by the row totals.
    totals = counts.sum(axis=1, keepdims=True)
    expected = np.zeros_like(counts)
    np.divide(counts, totals, out=expected, where=totals > 0)
    np.testing.assert_allclose(z, expected)
    assert fig.layout.coloraxis.colorbar.title.text == "P(target | source)"


def test_normalize_total(constructor):
    df = constructor(
        {
            "sequence": ["s1", "s1", "s2", "s2", "s2"],
            "state": ["A", "B", "A", "B", "C"],
        }
    )
    counts = _matrix(px.transition_matrix(df, state="state", sequence="sequence"))
    fig = px.transition_matrix(
        df, state="state", sequence="sequence", normalize="total"
    )
    z = _matrix(fig)
    # 3 transitions in total: A->B twice, B->C once.
    np.testing.assert_allclose(z, counts / 3)
    assert z.sum() == pytest.approx(1)
    assert fig.layout.coloraxis.colorbar.title.text == "Fraction of all transitions"


def test_no_transitions_normalizations_are_all_zero(constructor):
    df = constructor({"state": ["A"], "sequence": ["s"]})
    for normalize in [None, "row", "total"]:
        fig = px.transition_matrix(
            df, state="state", sequence="sequence", normalize=normalize
        )
        assert np.all(_matrix(fig) == 0)


def test_hover_shows_source_target_count_and_probabilities(constructor):
    df = constructor(
        {
            "sequence": ["s1", "s1", "s1", "s2", "s2"],
            "state": ["A", "B", "C", "A", "C"],
        }
    )
    fig = px.transition_matrix(df, state="state", sequence="sequence")
    template = fig.data[0].hovertemplate
    assert "%{y}" in template  # source
    assert "%{x}" in template  # target
    assert "%{customdata[0]}" in template  # count
    assert "%{customdata[1]" in template  # P(target | source)
    assert "%{customdata[2]" in template  # fraction of all transitions

    states = list(fig.data[0].x)
    i = {s: k for k, s in enumerate(states)}
    data = _customdata(fig)
    # The A->C cell: count 1, row probability 0.5, total share 1/3.
    a_to_c = data[i["A"], i["C"]]
    assert a_to_c[0] == 1
    assert a_to_c[1] == pytest.approx(0.5)
    assert a_to_c[2] == pytest.approx(1 / 3)
    # The zero B->A cell carries zeroed probabilities too.
    assert np.all(data[i["B"], i["A"]] == 0)
    # Hover data is identical regardless of the selected normalization.
    for normalize in ["row", "total"]:
        other = _customdata(
            px.transition_matrix(
                df,
                state="state",
                sequence="sequence",
                normalize=normalize,
            )
        )
        np.testing.assert_allclose(other, data)


def test_stable_sort_keeps_input_order_on_ties(constructor):
    # Every event has the same sort value: the input order must be preserved.
    df = constructor(
        {
            "sequence": ["s"] * 4,
            "step": [1, 1, 1, 1],
            "state": ["first", "second", "third", "fourth"],
        }
    )
    fig = px.transition_matrix(df, state="state", sequence="sequence", sort="step")
    z = _matrix(fig)
    states = list(fig.data[0].x)
    i = {s: k for k, s in enumerate(states)}
    assert list(fig.data[0].y) == ["first", "second", "third", "fourth"]
    assert z[i["first"], i["second"]] == 1
    assert z[i["second"], i["third"]] == 1
    assert z[i["third"], i["fourth"]] == 1
    assert z[i["second"], i["first"]] == 0
    assert z.sum() == 3


def test_sort_reorders_events_within_sequences(constructor):
    df = constructor(
        {
            "sequence": ["s1", "s1", "s2", "s2"],
            "step": [2, 1, 9, 8],
            "state": ["later", "earlier", "late2", "early2"],
        }
    )
    fig = px.transition_matrix(df, state="state", sequence="sequence", sort="step")
    z = _matrix(fig)
    states = list(fig.data[0].x)
    i = {s: k for k, s in enumerate(states)}
    assert z[i["earlier"], i["later"]] == 1
    assert z[i["early2"], i["late2"]] == 1
    assert z.sum() == 2


def test_missing_sort_values_go_last(constructor):
    df = constructor(
        {
            "sequence": ["s"] * 3,
            "step": [None, 2, 1],
            "state": ["last", "middle", "first"],
        }
    )
    fig = px.transition_matrix(df, state="state", sequence="sequence", sort="step")
    z = _matrix(fig)
    states = list(fig.data[0].x)
    i = {s: k for k, s in enumerate(states)}
    assert z[i["first"], i["middle"]] == 1
    assert z[i["middle"], i["last"]] == 1
    assert z.sum() == 2


def test_category_orders_order_and_absent_states(constructor):
    df = constructor({"state": ["B", "A", "B"]})
    fig = px.transition_matrix(
        df, state="state", category_orders={"state": ["B", "A", "Z"]}
    )
    # Explicit order is honored on both axes, including absent state Z.
    assert list(fig.data[0].x) == ["B", "A", "Z"]
    assert list(fig.data[0].y) == ["B", "A", "Z"]
    z = _matrix(fig)
    assert z.shape == (3, 3)
    i = {"B": 0, "A": 1, "Z": 2}
    assert z[i["A"], i["B"]] == 1
    # The unobserved state keeps a full zero row and column.
    assert z[i["Z"], :].sum() == 0
    assert z[:, i["Z"]].sum() == 0


def test_category_orders_partial_appends_observed_states(constructor):
    df = constructor({"state": ["A", "B", "C"]})
    fig = px.transition_matrix(df, state="state", category_orders={"state": ["C"]})
    # Requested states first, remaining observed states appended.
    assert list(fig.data[0].x) == ["C", "A", "B"]


def test_labels_are_applied_to_axes_and_colorbar(constructor):
    df = constructor({"phase": ["A", "B"]})
    fig = px.transition_matrix(
        df,
        state="phase",
        labels={"phase": "Phase", "color": "Number of transitions"},
    )
    assert fig.layout.xaxis.title.text == "Phase (target)"
    assert fig.layout.yaxis.title.text == "Phase (source)"
    assert fig.layout.coloraxis.colorbar.title.text == "Number of transitions"


def test_colorscale_range_and_midpoint(constructor):
    df = constructor({"state": ["A", "B", "C"]})
    fig = px.transition_matrix(
        df,
        state="state",
        color_continuous_scale="Reds",
        range_color=(2, 8),
        color_continuous_midpoint=5,
    )
    coloraxis = fig.layout.coloraxis
    assert coloraxis.cmin == 2
    assert coloraxis.cmax == 8
    assert coloraxis.cmid == 5
    assert coloraxis.autocolorscale is False
    # The named colorscale got expanded to a concrete list of color stops.
    assert len(coloraxis.colorscale) > 1


def test_input_dataframe_is_not_modified(constructor):
    df = constructor(
        {
            "sequence": ["s2", "s1", "s1", "s2"],
            "step": [1, 2, 1, 2],
            "state": ["D", "A", "B", "C"],
        }
    )
    before = nw.from_native(df).to_pandas().copy()
    px.transition_matrix(
        df,
        state="state",
        sequence="sequence",
        sort="step",
        on_missing="skip",
        normalize="row",
        category_orders={"state": ["A", "B", "C", "D", "E"]},
    )
    after = nw.from_native(df).to_pandas()
    pd.testing.assert_frame_equal(before, after)


def test_array_like_inputs_without_dataframe():
    fig = px.transition_matrix(
        state=["A", "B", "A", "B"],
        sequence=["s1", "s1", "s2", "s2"],
        sort=[1, 2, 1, 2],
    )
    z = _matrix(fig)
    states = list(fig.data[0].x)
    i = {s: k for k, s in enumerate(states)}
    assert z[i["A"], i["B"]] == 2
    assert z.sum() == 2


def test_validation_errors(constructor):
    df = constructor({"state": ["A", "B"]})
    with pytest.raises(ValueError, match="`state` argument is required"):
        px.transition_matrix()
    with pytest.raises(ValueError, match="not the name of a column"):
        px.transition_matrix(df, state="missing")
    with pytest.raises(ValueError, match="`normalize` must be one of"):
        px.transition_matrix(df, state="state", normalize="percent")
    with pytest.raises(ValueError, match="`on_missing` must be one of"):
        px.transition_matrix(df, state="state", on_missing="remove")
    with pytest.raises(ValueError, match="`category_orders` must be a dict"):
        px.transition_matrix(df, state="state", category_orders=["A", "B"])
    with pytest.raises(ValueError, match="`labels` must be a dict"):
        px.transition_matrix(df, state="state", labels="state")


def test_empty_dataframe():
    df = pd.DataFrame({"state": pd.Series([], dtype=object)})
    fig = px.transition_matrix(df, state="state")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "heatmap"


def test_self_transitions_are_counted(constructor):
    df = constructor({"state": ["A", "A", "A"], "sequence": ["s"] * 3})
    fig = px.transition_matrix(df, state="state", sequence="sequence")
    z = _matrix(fig)
    assert z[0, 0] == 2


def test_repeated_transitions_aggregate(constructor):
    # The same transition observed twice in two different sequences counts 2.
    df = constructor(
        {
            "sequence": ["s1", "s1", "s2", "s2"],
            "state": ["A", "B", "A", "B"],
        }
    )
    fig = px.transition_matrix(df, state="state", sequence="sequence")
    z = _matrix(fig)
    assert z[0, 1] == 2
