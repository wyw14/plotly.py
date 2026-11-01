import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pytest


@pytest.fixture
def df_mixed():
    # Covers None/NaN/NaT across float, nullable and datetime dtypes, plus an
    # all-missing column and a fully-complete column.
    frame = pd.DataFrame(
        {
            "a": pd.array([1, None, 3, None], dtype="Int64"),
            "b": pd.array(["x", "", None, "y"], dtype="string"),
            "c": pd.to_datetime(["2020-01-01", None, "2020-02-01", None]),
            "d": [1.0, np.nan, 3.0, 4.0],
            "e": pd.array([None, None, None, None], dtype="boolean"),
            "f": [1, 2, 3, 4],
        },
        index=pd.Index([7, 7, 9, 9], name="rid"),
    )
    return frame


def _traces(fig):
    assert len(fig.data) == 2
    assert fig.data[0].type == "bar"
    assert fig.data[1].type == "heatmap"
    return fig.data[0], fig.data[1]


def test_returns_plain_figure(df_mixed):
    fig = px.missing_data(df_mixed)
    assert type(fig) is go.Figure
    # Regular figures serialize like any other go.Figure.
    pio.to_json(fig)


def test_default_mask_and_per_column_stats(df_mixed):
    bar, heatmap = _traces(px.missing_data(df_mixed))
    expected_mask = df_mixed.isna().to_numpy()
    np.testing.assert_array_equal(np.array(heatmap.z), expected_mask.astype("int8"))
    # bar chart carries the missing counts ...
    np.testing.assert_array_equal(np.array(bar.y), expected_mask.sum(axis=0))
    # ... and the percentages, including the all-null and complete columns.
    np.testing.assert_allclose(
        np.array(bar.customdata)[:, 1].astype(float),
        100.0 * expected_mask.sum(axis=0) / len(df_mixed),
    )
    assert list(bar.customdata[:, 0]) == list(df_mixed.columns.astype(str))
    assert bar.customdata[4, 1] == pytest.approx(100.0)  # all-null column 'e'
    assert bar.customdata[5, 1] == pytest.approx(0.0)  # complete column 'f'


def test_columns_and_rows_keep_original_order(df_mixed):
    fig = px.missing_data(df_mixed)
    heatmap = fig.data[1]
    assert list(heatmap.x) == [0, 1, 2, 3, 4, 5]
    assert list(heatmap.y) == [0, 1, 2, 3]
    # Bottom x axis (xaxis2) holds the column names, heatmap y axis (yaxis3)
    # holds the original index labels.
    assert tuple(fig.layout.xaxis2.ticktext) == tuple(df_mixed.columns)
    assert tuple(fig.layout.yaxis3.ticktext) == ("7", "7", "9", "9")
    assert fig.layout.yaxis3.title.text == "rid"


def test_duplicate_row_labels_are_distinct_rows(df_mixed):
    heatmap = px.missing_data(df_mixed).data[1]
    # Positions are integers so duplicated labels do not collapse onto one
    # another, while both labels are still displayed.
    assert heatmap.z.shape == (4, 6)
    assert list(heatmap.y) == [0, 1, 2, 3]


def test_duplicate_column_selection(df_mixed):
    fig = px.missing_data(df_mixed, columns=["a", "a", "f"])
    bar, heatmap = _traces(fig)
    assert heatmap.z.shape == (4, 3)
    assert list(bar.customdata[:, 0]) == ["a", "a", "f"]


@pytest.mark.parametrize(
    "columns",
    [
        "a",
        ["a", "f"],
        ("a", "f"),
        pd.Index(["a", "f"]),
        np.array(["a", "f"]),
    ],
)
def test_columns_argument_forms(df_mixed, columns):
    fig = px.missing_data(df_mixed, columns=columns)
    bar, heatmap = _traces(fig)
    expected = ["a"] if isinstance(columns, str) else list(columns)
    assert heatmap.z.shape == (4, len(expected))
    assert list(bar.customdata[:, 0]) == expected


def test_empty_string_is_present_by_default(df_mixed):
    fig = px.missing_data(df_mixed)
    heatmap = fig.data[1]
    # Row 1, column 'b' is "" -> present.
    assert heatmap.z[1][1] == 0


def test_empty_string_flag(df_mixed):
    fig = px.missing_data(df_mixed, empty_string_as_missing=True)
    heatmap = fig.data[1]
    assert heatmap.z[1][1] == 1
    # Whitespace-only strings stay present: only the exact "" is flagged.
    df_ws = pd.DataFrame({"s": [" ", "\t", "", None]})
    z = px.missing_data(df_ws, empty_string_as_missing=True).data[1].z
    assert np.array(z).tolist() == [[0], [0], [1], [1]]
    # And by default even exact empty strings are present.
    z_default = px.missing_data(df_ws).data[1].z
    assert np.array(z_default).tolist() == [[0], [0], [0], [1]]


def test_sorted_rows_are_stable(df_mixed):
    # Per-row missing counts are [1, 4, 2, 3], all distinct here.
    fig = px.missing_data(df_mixed, sort_rows=True)
    heatmap = fig.data[1]
    counts = (np.array(heatmap.z) == 1).sum(axis=1)
    assert counts.tolist() == [4, 3, 2, 1]
    positions = heatmap.customdata[:, 0, 3]
    assert positions.tolist() == [1, 3, 2, 0]
    # Row labels follow the new ordering, but keep the original labels.
    assert heatmap.customdata[:, 0, 0].tolist() == ["7", "9", "9", "7"]


def test_sorted_rows_ties_keep_original_order():
    # All three rows have exactly one missing value, so their relative order
    # must be preserved by the stable sort even though the patterns differ.
    df = pd.DataFrame(
        {"a": [np.nan, 1.0, 1.0], "b": [1.0, np.nan, 1.0], "c": [1.0, 1.0, np.nan]}
    )
    heatmap = px.missing_data(df, sort_rows=True).data[1]
    z = np.array(heatmap.z)
    # Each row keeps its unique missing column.
    assert z[:, :].tolist() == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    assert heatmap.customdata[:, 0, 3].tolist() == [0, 1, 2]

    # With duplicated missing counts in the middle, ties stay stable while the
    # row with more missing values moves up.
    df2 = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [1.0, 1.0, np.nan]})
    # counts: [0, 1, 2] -> descending [2(row2), 1(row1), 0(row0)]
    z2 = np.array(px.missing_data(df2, sort_rows=True).data[1].z)
    assert z2.tolist() == [[1, 1], [1, 0], [0, 0]]


def test_hover_keeps_original_identifiers(df_mixed):
    heatmap = px.missing_data(df_mixed, sort_rows=True).data[1]
    cd = heatmap.customdata
    # Every cell carries its row label, column name, status and original row
    # position, regardless of the display ordering.
    for display_row in range(4):
        original_position = int(cd[display_row, 0, 3])
        for display_col in range(6):
            assert cd[display_row, display_col, 0] == str(
                df_mixed.index[original_position]
            )
            assert cd[display_row, display_col, 1] == df_mixed.columns[display_col]
            expected_status = (
                "Missing"
                if df_mixed.isna().iloc[original_position, display_col]
                else "Present"
            )
            assert cd[display_row, display_col, 2] == expected_status


def test_zero_rows_keeps_columns():
    df = pd.DataFrame(
        {
            "a": pd.array([], dtype="Int64"),
            "b": pd.to_datetime([]),
            "c": pd.array([], dtype="string"),
        }
    )
    fig = px.missing_data(df)
    bar, heatmap = _traces(fig)
    assert np.array(heatmap.z).shape == (0, 3)
    assert np.array(bar.y).tolist() == [0, 0, 0]
    assert np.array(bar.customdata)[:, 1].tolist() == [0.0, 0.0, 0.0]
    # The selected columns are still shown.
    assert tuple(fig.layout.xaxis2.ticktext) == ("a", "b", "c")
    # Figure renders/serializes without error.
    pio.to_json(fig)


def test_zero_rows_with_column_subset(df_mixed):
    df0 = df_mixed.iloc[:0]
    fig = px.missing_data(df0, columns=["a", "e", "f"])
    bar, heatmap = _traces(fig)
    assert np.array(heatmap.z).shape == (0, 3)
    assert np.array(bar.y).tolist() == [0, 0, 0]
    assert tuple(fig.layout.xaxis2.ticktext) == ("a", "e", "f")


def test_all_null_and_complete_columns(df_mixed):
    fig = px.missing_data(df_mixed, columns=["e", "f"])
    bar, heatmap = _traces(fig)
    z = np.array(heatmap.z)
    assert z[:, 0].tolist() == [1, 1, 1, 1]  # all-null
    assert z[:, 1].tolist() == [0, 0, 0, 0]  # all-present
    assert np.array(bar.y).tolist() == [4, 0]
    # Discrete colorscale stays pinned to 0/1.
    assert heatmap.zmin == 0
    assert heatmap.zmax == 1


def test_input_is_not_modified(df_mixed):
    snapshot = df_mixed.copy(deep=True)
    snapshot_index = df_mixed.index.copy()
    px.missing_data(df_mixed, sort_rows=True, empty_string_as_missing=True)
    pd.testing.assert_frame_equal(df_mixed, snapshot)
    pd.testing.assert_index_equal(df_mixed.index, snapshot_index)
    # dtypes, column order and values are untouched
    assert list(df_mixed.columns) == list(snapshot.columns)
    assert list(df_mixed.dtypes) == list(snapshot.dtypes)


def test_non_pandas_input_raises():
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        px.missing_data([1, 2, 3])
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        px.missing_data({"a": [1, 2]})


def test_unknown_column_raises(df_mixed):
    with pytest.raises(KeyError, match="not found in data_frame"):
        px.missing_data(df_mixed, columns=["a", "missing"])
    with pytest.raises(KeyError, match="not found in data_frame"):
        px.missing_data(df_mixed, columns="missing")


def test_invalid_columns_argument(df_mixed):
    with pytest.raises(ValueError, match="'columns' must be"):
        px.missing_data(df_mixed, columns=123)


def test_title_labels_and_sizes(df_mixed):
    fig = px.missing_data(
        df_mixed,
        title="Missing overview",
        labels={"index": "Row", "column": "Variable", "status": "NA?"},
        width=700,
        height=600,
    )
    assert fig.layout.title.text == "Missing overview"
    assert fig.layout.width == 700
    assert fig.layout.height == 600
    assert fig.layout.yaxis3.title.text == "Row"
    assert fig.layout.xaxis2.title.text == "Variable"
    heatmap = fig.data[1]
    assert heatmap.colorbar.title.text == "NA?"


def test_axes_are_aligned(df_mixed):
    fig = px.missing_data(df_mixed)
    # The bar and heatmap share their x axis range, centered on the same
    # integer column positions.
    assert list(fig.layout.xaxis.range) == [-0.5, 5.5]
    assert list(fig.layout.xaxis2.range) == [-0.5, 5.5]
    # Count and percentage axes both start at 0 and stay proportional.
    assert list(fig.layout.yaxis.range) == [0, 4]
    assert list(fig.layout.yaxis2.range) == [0, 100]


def test_datetime_and_multiindex_row_labels():
    df = pd.DataFrame(
        {"a": [1.0, np.nan]},
        index=pd.to_datetime(["2021-01-01", "2021-01-02"]),
    )
    fig = px.missing_data(df)
    assert tuple(fig.layout.yaxis3.ticktext) == (
        "2021-01-01 00:00:00",
        "2021-01-02 00:00:00",
    )

    midx = pd.MultiIndex.from_tuples([("x", 1), ("x", 2)], names=["l1", "l2"])
    df_multi = pd.DataFrame({"a": [1.0, np.nan]}, index=midx)
    fig_multi = px.missing_data(df_multi)
    ticktext = tuple(fig_multi.layout.yaxis3.ticktext)
    assert ticktext == ("('x', 1)", "('x', 2)")
    assert fig_multi.layout.yaxis3.title.text == "l1, l2"
