import math

import narwhals.stable.v1 as nw
import numpy as np

import plotly.graph_objs as go
from _plotly_utils.basevalidators import ColorscaleValidator

from ._core import apply_default_cascade

# Internal column names, unlikely to collide with user data.
_STATE = "__px_transition_matrix_state"
_SEQUENCE = "__px_transition_matrix_sequence"
_SORT = "__px_transition_matrix_sort"

_NO_PREVIOUS = object()
_NULL_GROUP = object()

_VALID_NORMALIZE = (None, "row", "total")
_VALID_ON_MISSING = ("break", "skip")


def _is_nan(value):
    # `math.isnan` is only defined for floats (and np.floating subclasses);
    # it lets us catch NaN values that polars does not consider "null".
    return isinstance(value, float) and math.isnan(value)


def _column_name(argument, fallback):
    """Human-readable name of the state column, for labels/category_orders."""
    if isinstance(argument, str):
        return argument
    name = getattr(argument, "name", None)
    if name is not None:
        return str(name)
    return fallback


def _as_series(argument, native_namespace):
    """Turn an array-like argument into a narwhals series of the frame's backend."""
    if nw.dependencies.is_into_series(argument):
        values = nw.from_native(argument, series_only=True).to_list()
    else:
        values = list(argument)
    return nw.new_series("__px_value__", values, native_namespace=native_namespace)


def _build_frame(data_frame, state, sequence, sort):
    """Resolve the state/sequence/sort arguments into a narwhals DataFrame.

    Raises with the same error messages as the rest of plotly express when
    possible. The input dataframe is never modified.
    """
    if state is None:
        raise ValueError(
            "The `state` argument is required: it must be the name of the "
            "column of `data_frame` (or an array-like) containing the state "
            "of each event."
        )

    if data_frame is None:
        # No dataframe: pandas is the fallback backend, like in the rest of px.
        try:
            import pandas as pd
        except ImportError:
            msg = "Pandas installation is required if no dataframe is provided."
            raise NotImplementedError(msg)
        native_namespace = pd
        frame = None
    else:
        frame = nw.from_native(data_frame, eager_only=True)
        native_namespace = nw.get_native_namespace(frame)

    def resolve(argument, internal_name):
        if argument is None:
            return None
        if isinstance(argument, str):
            if frame is None or argument not in frame.columns:
                raise ValueError(
                    "Value of '%s' is not the name of a column in 'data_frame'. "
                    "Expected one of %s but received: %s"
                    % (
                        argument,
                        str(list(frame.columns) if frame is not None else []),
                        argument,
                    )
                )
            return frame.get_column(argument).alias(internal_name)
        return _as_series(argument, native_namespace).alias(internal_name)

    columns = {_STATE: resolve(state, _STATE)}
    seq_series = resolve(sequence, _SEQUENCE)
    if seq_series is not None:
        columns[_SEQUENCE] = seq_series
    sort_series = resolve(sort, _SORT)
    if sort_series is not None:
        columns[_SORT] = sort_series

    return nw.from_dict(columns)


def _missing_mask(series):
    """Boolean list: True for null/None/NaN values of the given series."""
    nulls = series.is_null().to_numpy()
    values = series.to_numpy()
    return [bool(nulls[i]) or _is_nan(values[i]) for i in range(len(values))]


def _count_transitions(df, has_sequence, on_missing):
    """Count source -> target transitions, returns a dict {(source, target): n}."""
    n = df.shape[0]
    states = df.get_column(_STATE).to_numpy()
    state_missing = _missing_mask(df.get_column(_STATE))

    if has_sequence:
        groups_values = df.get_column(_SEQUENCE).to_numpy()
        groups_missing = _missing_mask(df.get_column(_SEQUENCE))
    else:
        groups_values = [0] * n
        groups_missing = [False] * n

    if _SORT in df.columns:
        sort_values = df.get_column(_SORT).to_numpy()
        sort_missing = _missing_mask(df.get_column(_SORT))
    else:
        sort_values = None
        sort_missing = None

    # Gather the row indices of each sequence. Null sequence values form a
    # single group of their own, but never connect to any other sequence.
    groups = {}
    for i in range(n):
        key = _NULL_GROUP if groups_missing[i] else groups_values[i]
        groups.setdefault(key, []).append(i)

    counts = {}
    for rows in groups.values():
        if sort_values is not None:
            # Stable ordering: the original row index is the tiebreaker, so
            # events sharing a sort value keep their relative input order.
            # Missing sort values are placed last.
            rows = sorted(
                rows,
                key=lambda i: (
                    sort_missing[i],
                    sort_values[i] if not sort_missing[i] else 0,
                    i,
                ),
            )

        previous = _NO_PREVIOUS
        for i in rows:
            if state_missing[i]:
                if on_missing == "break":
                    # The gap interrupts the sequence: no transition may
                    # bridge over this event.
                    previous = _NO_PREVIOUS
                # With "skip", the missing event is simply ignored and the
                # chain continues with the next valid event.
                continue
            current = states[i]
            if previous is not _NO_PREVIOUS:
                pair = (previous, current)
                counts[pair] = counts.get(pair, 0) + 1
            previous = current

    return counts


def _ordered_states(df, state_argument, category_orders):
    """States on both axes, in order.

    States explicitly requested through `category_orders` come first (even if
    they never occur in the data); the states observed in the data are then
    appended in their order of first appearance.
    """
    state_name = _column_name(state_argument, "state")
    ordered = []
    seen = set()

    def add(value):
        if value not in seen:
            seen.add(value)
            ordered.append(value)

    if category_orders:
        requested = category_orders.get(state_name)
        if requested is not None:
            for value in requested:
                add(value)

    values = df.get_column(_STATE).to_numpy()
    missing = _missing_mask(df.get_column(_STATE))
    for i, value in enumerate(values):
        if not missing[i]:
            add(value)

    return state_name, ordered


def transition_matrix(
    data_frame=None,
    state=None,
    sequence=None,
    sort=None,
    normalize=None,
    on_missing="break",
    category_orders=None,
    labels=None,
    color_continuous_scale=None,
    range_color=None,
    color_continuous_midpoint=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    Compute a state transition matrix from one or several sequences of events
    and display it as a heatmap.

    Every transition goes from the state of an event to the state of the next
    valid event *within the same sequence*. No transition is ever created
    between two different sequences, nor after the last event of a sequence.

    Parameters
    ----------

    data_frame: DataFrame or array-like or dict
        Same semantics as for the other plotly express functions: a (pandas,
        polars or pyarrow) DataFrame, or ``None`` if the other arguments are
        array-likes. This argument is never modified.

    state: str or int or Series or array-like
        Name of a column in `data_frame`, or a pandas Series or array-like,
        containing the state reached by each event. Rows with a missing
        (``None``/``NaN``) state are not valid events; see `on_missing` for
        how they interrupt the sequences.

    sequence: str or int or Series or array-like, optional
        Name of a column in `data_frame`, or a pandas Series or array-like,
        identifying the sequence (e.g. session, trajectory or subject) each
        event belongs to. Transitions are only counted between consecutive
        events of the same sequence. If omitted, all the rows form a single
        sequence.

    sort: str or int or Series or array-like, optional
        Name of a column in `data_frame`, or a pandas Series or array-like,
        giving the order of events within each sequence (e.g. a timestamp or
        a step number). If omitted, the rows are used in their input order.
        Events sharing the same sort value keep their relative input order.
        Missing sort values are placed last.

    normalize: None (default), ``'row'`` or ``'total'``
        Value represented by the heatmap colors:

        - ``None``: raw transition counts.
        - ``'row'``: transition probability conditional on the source state,
          i.e. each row is divided by the out-degree of its source state.
          Rows of states with zero out-degree are entirely zero.
        - ``'total'``: fraction of all transitions, i.e. every cell is
          divided by the total number of transitions.

        Regardless of this setting, the hover of every cell shows the raw
        count as well as the row and total probabilities.

    on_missing: ``'break'`` (default) or ``'skip'``
        How a missing state interrupts the ordering of a sequence:

        - ``'break'``: a missing state splits the sequence in two; no
          transition is counted across the gap (the events on either side
          are not connected).
        - ``'skip'``: missing states are dropped and the valid events on
          either side become consecutive, i.e. the sequence is connected
          across the gap.

    category_orders: dict with str keys and list of str/values values (default ``{}``)
        By default, the states are ordered by order of first appearance in
        `state`. This argument forces a specific ordering of the states: its
        key must be the name of the state column, and its value the ordered
        list of states. States listed here but absent from the data are still
        displayed (with zero counts/probabilities) on both axes.

    labels: dict with str keys and str values (default ``{}``)
        Names used for the axis and colorbar titles. The key should be the
        name of the state column; ``'color'`` overrides the colorbar title.

    color_continuous_scale: list of colors
        Colorscale used to represent the matrix values.

    range_color: list of two numbers
        If provided, overrides auto-scaling on the continuous color scale,
        including overriding `color_continuous_midpoint`.

    color_continuous_midpoint: number
        If set, computes the bounds of the continuous color scale to have the
        desired midpoint. Overridden by `range_color`.

    title: str
        The figure title.

    subtitle: str
        The figure subtitle.

    template: str or dict or plotly.graph_objects.layout.Template instance
        The figure template name or definition.

    width: number
        The figure width in pixels.

    height: number
        The figure height in pixels.

    Returns
    -------
    fig: graph_objects.Figure
        A figure containing a single ``heatmap`` trace. Use
        `plotly.graph_objects.Figure.update_traces` or
        `plotly.graph_objects.Figure.update_layout` to customize it further.
    """
    args = locals()
    if normalize not in _VALID_NORMALIZE:
        raise ValueError(
            "`normalize` must be one of None, 'row' or 'total', "
            "but received: %s" % str(normalize)
        )
    if on_missing not in _VALID_ON_MISSING:
        raise ValueError(
            "`on_missing` must be one of 'break' or 'skip', "
            "but received: %s" % str(on_missing)
        )
    if category_orders is not None and not isinstance(category_orders, dict):
        raise ValueError(
            "`category_orders` must be a dict mapping column names to lists "
            "of ordered values, but received: %s" % str(type(category_orders))
        )
    if labels is not None and not isinstance(labels, dict):
        raise ValueError(
            "`labels` must be a dict, but received: %s" % str(type(labels))
        )

    user_provided_colorscale = args.get("color_continuous_scale") is not None
    apply_default_cascade(args, constructor=None)
    labels = (args["labels"] or {}).copy()
    category_orders = args["category_orders"] or {}

    frame = _build_frame(
        args["data_frame"], args["state"], args["sequence"], args["sort"]
    )

    state_name, states = _ordered_states(frame, args["state"], category_orders)
    counts_dict = _count_transitions(
        frame, has_sequence=args["sequence"] is not None, on_missing=on_missing
    )

    n_states = len(states)
    index = {state: i for i, state in enumerate(states)}
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    for (source, target), value in counts_dict.items():
        # Every source/target comes from the data, so it is in `index`:
        # category_orders only adds further states.
        counts[index[source], index[target]] = value

    row_totals = counts.sum(axis=1, keepdims=True)
    row_prob = np.zeros_like(counts, dtype=float)
    np.divide(counts, row_totals, out=row_prob, where=row_totals > 0)
    total = counts.sum()
    if total > 0:
        total_prob = counts / total
    else:
        total_prob = np.zeros_like(counts, dtype=float)

    if normalize is None:
        z = counts
        color_label = "Count"
    elif normalize == "row":
        z = row_prob
        color_label = "P(target | source)"
    else:
        z = total_prob
        color_label = "Fraction of all transitions"

    customdata = np.stack([counts.astype(float), row_prob, total_prob], axis=-1)

    state_label = labels.get(state_name, state_name)
    source_label = "%s (source)" % state_label if state_label else "source"
    target_label = "%s (target)" % state_label if state_label else "target"

    hovertemplate = (
        "%s: %%{y}<br>"
        "%s: %%{x}<br>"
        "count: %%{customdata[0]}<br>"
        "P(target | source): %%{customdata[1]:.3f}<br>"
        "fraction of all transitions: %%{customdata[2]:.3f}"
        "<extra></extra>"
    ) % (source_label, target_label)

    if range_color is not None:
        zmin = range_color[0]
        zmax = range_color[1]
    else:
        zmin = None
        zmax = None

    colorscale_validator = ColorscaleValidator("colorscale", "transition_matrix")
    coloraxis = dict(
        colorscale=colorscale_validator.validate_coerce(args["color_continuous_scale"]),
        cmid=color_continuous_midpoint,
        cmin=zmin,
        cmax=zmax,
        colorbar=dict(title_text=labels.get("color", color_label)),
    )
    if user_provided_colorscale:
        coloraxis["autocolorscale"] = False

    trace = go.Heatmap(
        z=z,
        x=states,
        y=states,
        customdata=customdata,
        hovertemplate=hovertemplate,
        coloraxis="coloraxis1",
        name="",
    )
    fig = go.Figure(trace)

    layout = dict(
        xaxis=dict(
            title_text=target_label,
            categoryorder="array",
            categoryarray=list(states),
        ),
        yaxis=dict(
            title_text=source_label,
            categoryorder="array",
            categoryarray=list(states),
        ),
        coloraxis1=coloraxis,
    )
    if width is not None:
        layout["width"] = width
    if height is not None:
        layout["height"] = height
    if title:
        layout["title_text"] = title
    elif args["template"].layout.margin.t is None:
        layout["margin"] = {"t": 60}
    if subtitle:
        layout["title_subtitle_text"] = subtitle

    fig.update_layout(layout)
    fig.update_layout(template=args["template"], overwrite=True)
    return fig
