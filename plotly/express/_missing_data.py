import numpy as np

import plotly.graph_objs as go
from plotly import optional_imports
from plotly.subplots import make_subplots

from ._core import apply_default_cascade

_PRESENT_COLOR = "#F0F0F0"
_MISSING_COLOR = "#E45756"
_MAX_TICK_LABELS = 50


def _stringify(values):
    return [str(value) for value in values]


def _sparse_tick_positions(n, max_labels=_MAX_TICK_LABELS):
    # Avoid flooding the axes when there are hundreds of rows or columns.
    if n <= max_labels:
        return list(range(n))
    step = int(np.ceil(n / max_labels))
    return list(range(0, n, step))


def missing_data(
    data_frame,
    columns=None,
    sort_rows=False,
    empty_string_as_missing=False,
    labels=None,
    color_present=_PRESENT_COLOR,
    color_missing=_MISSING_COLOR,
    title=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    Overview of missing values in a pandas DataFrame.

    The figure combines, from top to bottom:

    - A bar chart of the number (and percentage) of missing values per column.
    - A heatmap aligned with the bar chart in which every cell corresponds to
      one value of the input DataFrame, colored according to whether the value
      is missing or present.

    ``None``, ``numpy.nan``, ``pandas.NaT`` and the null values of pandas
    nullable dtypes (``pandas.NA``) are all considered missing. Empty strings
    (``""``) are considered present unless ``empty_string_as_missing`` is set
    to ``True``.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        DataFrame to inspect. The input DataFrame is never modified.
    columns : str or iterable of str, optional
        Name of a single column or iterable of column names to inspect. If
        ``None`` (the default), every column of ``data_frame`` is used. The
        columns are displayed in the order in which they are provided (or in
        their original order when ``columns`` is ``None``).
    sort_rows : bool (default ``False``)
        If ``True``, rows are displayed in decreasing order of their number of
        missing values. The sort is stable: rows with the same missing count
        keep their original relative order. If ``False`` (the default), the
        original row order is preserved. In both cases the hover label keeps
        the original row identifier and original row position.
    empty_string_as_missing : bool (default ``False``)
        If ``True``, empty strings (``""``) are considered missing in addition
        to null values. If ``False`` (the default), empty strings are
        considered present.
    labels : dict with str keys and str values (default ``{}``)
        Overrides axis titles. Supported keys are:

        - ``'index'``: title of the heatmap row axis (defaults to the
          DataFrame index name, if any).
        - ``'column'``: title of the heatmap column axis.
        - ``'status'``: title of the color bar (defaults to ``'Status'``).
    color_present : str (default ``'#F0F0F0'``)
        CSS-compatible color used for present values.
    color_missing : str (default ``'#E45756'``)
        CSS-compatible color used for missing values and for the bar chart.
    title : str
        The figure title.
    template : str or dict or plotly.graph_objects.layout.Template instance
        The figure template name or definition.
    width : number
        The figure width in pixels.
    height : number
        The figure height in pixels.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A regular ``Figure`` instance containing a bar trace and a heatmap
        trace laid out in two stacked, x-aligned subplots.

    Notes
    -----
    Rows and columns are positioned by their integer position rather than by
    their label, so duplicated row index labels or duplicated column names
    are displayed as distinct cells. Zero-row DataFrames are supported: the
    selected columns are kept and every count and percentage is zero.
    """
    args = locals()
    apply_default_cascade(args, constructor=None)
    labels = dict(labels) if labels else {}

    pd = optional_imports.get_module("pandas")
    if pd is None:
        raise ImportError(
            "px.missing_data requires pandas to be installed. You can install "
            "pandas using pip with:\n\n$ pip install pandas"
        )
    if not isinstance(data_frame, pd.DataFrame):
        raise ValueError(
            "The 'data_frame' argument of px.missing_data must be a pandas "
            "DataFrame, but an object of type %s was passed."
            % type(data_frame).__name__
        )

    # ----- Resolve the selected columns, without mutating the input --------
    if columns is None:
        selected_columns = list(data_frame.columns)
    elif isinstance(columns, str):
        selected_columns = [columns]
    else:
        try:
            selected_columns = list(columns)
        except TypeError:
            raise ValueError(
                "'columns' must be a column name or an iterable of column "
                "names, but an object of type %s was passed." % type(columns).__name__
            )
    unknown_columns = [c for c in selected_columns if c not in data_frame.columns]
    if unknown_columns:
        raise KeyError(
            "The following columns were not found in data_frame: %s"
            % ", ".join(_stringify(unknown_columns))
        )

    # Work on a copy so that none of the computations below can touch the
    # caller's DataFrame (in-place or otherwise).
    sub_frame = data_frame[selected_columns].copy()

    # ----- Boolean missing mask --------------------------------------------
    # isna() uniformly detects None, np.nan, pd.NaT and pandas nullable nulls
    # (pd.NA), while empty strings are reported as present.
    missing_mask = sub_frame.isna()
    if empty_string_as_missing:
        # eq("") returns NA on the null entries of nullable dtypes, so the
        # combined mask is filled with False before being turned into numpy.
        missing_mask = (missing_mask | sub_frame.eq("")).fillna(False)
    missing = missing_mask.to_numpy(dtype=bool)

    n_rows, n_columns = missing.shape
    column_missing_counts = missing.sum(axis=0)
    row_missing_counts = missing.sum(axis=1)

    # Stable ordering: decreasing count, ties keep their original position.
    if sort_rows:
        row_order = np.argsort(-row_missing_counts, kind="stable")
    else:
        row_order = np.arange(n_rows)
    missing_display = missing[row_order, :]

    # Integer positions (and not the labels) are used on the axes, so that
    # duplicated index labels or column names remain distinguishable.
    x_positions = list(range(n_columns))
    y_positions = list(range(n_rows))
    column_labels = _stringify(sub_frame.columns)
    original_row_labels = _stringify(sub_frame.index)
    row_labels_display = [original_row_labels[int(i)] for i in row_order]

    if n_rows:
        missing_percent = 100.0 * column_missing_counts / n_rows
    else:
        # Zero rows: percentages are defined and equal to 0 for every column.
        missing_percent = np.zeros(n_columns)

    # ----- Top bar chart: per-column missing count and percentage ---------
    bar_customdata = np.empty((n_columns, 2), dtype=object)
    bar_customdata[:, 0] = column_labels
    bar_customdata[:, 1] = missing_percent

    bar_trace = go.Bar(
        x=x_positions,
        y=column_missing_counts,
        marker=dict(color=color_missing),
        customdata=bar_customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Missing: %{y} / " + str(n_rows) + " (%{customdata[1]:.1f}%)<extra></extra>"
        ),
        name="Missing count",
        showlegend=False,
    )

    # ----- Bottom heatmap: missing status of every cell --------------------
    heatmap_customdata = np.empty(missing_display.shape + (4,), dtype=object)
    if n_columns:
        heatmap_customdata[..., 0] = np.broadcast_to(
            np.array(row_labels_display, dtype=object).reshape(-1, 1),
            missing_display.shape,
        )
        heatmap_customdata[..., 3] = np.broadcast_to(
            np.asarray(row_order, dtype=object).reshape(-1, 1),
            missing_display.shape,
        )
    if n_rows:
        heatmap_customdata[..., 1] = np.broadcast_to(
            np.array(column_labels, dtype=object).reshape(1, -1),
            missing_display.shape,
        )
        heatmap_customdata[..., 2] = np.where(missing_display, "Missing", "Present")

    heatmap_trace = go.Heatmap(
        z=missing_display.astype("int8"),
        x=x_positions,
        y=y_positions,
        colorscale=[[0.0, color_present], [1.0, color_missing]],
        zmin=0,
        zmax=1,
        xgap=2,
        ygap=2,
        showscale=True,
        colorbar=dict(
            title=dict(text=labels.get("status", "Status")),
            tickvals=[0.25, 0.75],
            ticktext=["Present", "Missing"],
            len=0.72,
            y=0,
            yanchor="bottom",
        ),
        customdata=heatmap_customdata,
        hovertemplate=(
            "Row: %{customdata[0]} (position %{customdata[3]})<br>"
            "Column: %{customdata[1]}<br>"
            "%{customdata[2]}<extra></extra>"
        ),
        name="Missing status",
        showlegend=False,
    )

    # ----- Assemble the two stacked, x-aligned subplots --------------------
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.22, 0.78],
        specs=[[{"secondary_y": True}], [{}]],
    )
    fig.add_trace(bar_trace, row=1, col=1)
    fig.add_trace(heatmap_trace, row=2, col=1)

    x_range = [-0.5, n_columns - 0.5] if n_columns else [-0.5, 0.5]
    fig.update_xaxes(range=x_range, showticklabels=False, row=1, col=1)
    x_tick_positions = _sparse_tick_positions(n_columns)
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_tick_positions,
        ticktext=[column_labels[i] for i in x_tick_positions],
        range=x_range,
        side="bottom",
        automargin=True,
        title_text=labels.get("column") or "",
        row=2,
        col=1,
    )

    # The primary y axis shows counts while the overlaying secondary axis
    # shows percentages. Both start at 0, so the two scales stay proportional
    # and the same bar can be read on either axis.
    fig.update_yaxes(
        title_text="Missing count",
        range=[0, n_rows if n_rows else 1],
        minallowed=0,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Missing %",
        range=[0, 100],
        ticksuffix="%",
        showgrid=False,
        zeroline=False,
        secondary_y=True,
        row=1,
        col=1,
    )

    y_tick_positions = _sparse_tick_positions(n_rows)
    # Reversed range puts the first displayed row at the top, including for
    # zero-row inputs where the heatmap carries no data.
    y_range = [n_rows - 0.5, -0.5] if n_rows else [0.5, -0.5]
    index_names = [name for name in sub_frame.index.names if name is not None]
    default_index_title = ", ".join(str(name) for name in index_names)
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_tick_positions,
        ticktext=[row_labels_display[i] for i in y_tick_positions],
        range=y_range,
        automargin=True,
        title_text=labels.get("index") or default_index_title,
        row=2,
        col=1,
    )

    layout_updates = dict(showlegend=False)
    if args["title"]:
        layout_updates["title_text"] = args["title"]
    elif args["template"].layout.margin.t is None:
        layout_updates["margin"] = {"t": 60}
    for attr_name in ("width", "height"):
        if args[attr_name]:
            layout_updates[attr_name] = args[attr_name]
    fig.update_layout(**layout_updates)
    fig.update_layout(template=args["template"], overwrite=True)
    return fig
