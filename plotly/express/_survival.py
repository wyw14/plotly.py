from numbers import Real

import numpy as np

import plotly.graph_objs as go

from ._core import apply_default_cascade, build_dataframe, get_label

# Name used for the (single) implicit group when no `color` argument is provided,
# and for rows whose group value is missing (null/NaN), mirroring other PX charts.
_ONE_GROUP = ""
_MISSING_GROUP = "(?)"


def _coerce_durations(values):
    """Convert the observation-time column to a finite, non-negative float array."""
    try:
        durations = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Values of `x` must be numeric observation times, but some values could "
            "not be converted to numbers."
        ) from exc
    if durations.ndim != 1:
        raise ValueError("Values of `x` must be one-dimensional.")
    if not np.all(np.isfinite(durations)):
        raise ValueError(
            "Values of `x` must be finite observation times; NaN, positive infinity "
            "or negative infinity values are not allowed."
        )
    if np.any(durations < 0):
        raise ValueError(
            "Values of `x` must be non-negative observation times; negative times "
            "are not allowed."
        )
    return durations


def _coerce_events(values):
    """Convert the event indicator to an int array of 0/1 values.

    ``True``/``False`` and ``1``/``0`` (of any integer or floating point dtype) are
    accepted; everything else, including missing values, raises.
    """
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("Values of `event` must be one-dimensional.")

    if raw.dtype == bool:
        return raw.astype(np.int64)
    if np.issubdtype(raw.dtype, np.integer):
        unique = np.unique(raw)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError(
                "Values of `event` must be event indicators, i.e. one of "
                "True/False or 1/0; invalid integer values were provided "
                "(allowed values: 0, 1)."
            )
        return raw.astype(np.int64)
    if np.issubdtype(raw.dtype, np.floating):
        unique = np.unique(raw[~np.isnan(raw)])
        if not np.all(np.isfinite(raw)) or not np.all(np.isin(unique, [0.0, 1.0])):
            raise ValueError(
                "Values of `event` must be event indicators, i.e. one of "
                "True/False or 1/0; invalid floating-point values were provided "
                "(allowed values: 0.0, 1.0)."
            )
        return raw.astype(np.int64)

    # Object arrays (e.g. pyarrow-backed columns converted with numpy): inspect each
    # value individually so that booleans, numpy scalars and Python scalars all work.
    events = np.empty(raw.shape[0], dtype=np.int64)
    for i, value in enumerate(raw.tolist()):
        if value is None:
            raise ValueError(
                "Values of `event` must not contain missing values; please encode "
                "right-censored observations with False or 0 and events with "
                "True or 1."
            )
        if isinstance(value, (bool, np.bool_)):
            events[i] = int(value)
        elif isinstance(value, Real) and not isinstance(value, bool):
            number = float(value)
            if not np.isfinite(number) or number not in (0.0, 1.0):
                raise ValueError(
                    "Values of `event` must be event indicators, i.e. one of "
                    "True/False or 1/0; invalid value %r was provided." % (value,)
                )
            events[i] = int(number)
        else:
            raise ValueError(
                "Values of `event` must be event indicators, i.e. one of "
                "True/False or 1/0; invalid value %r of type %s was provided."
                % (value, type(value).__name__)
            )
    return events


def _is_missing(value):
    return value is None or (isinstance(value, float) and np.isnan(value))


def _kaplan_meier(durations, events):
    """Compute the Kaplan-Meier estimator for one group.

    At each distinct time ``t``, the risk set contains every subject whose
    observation time is greater than or equal to ``t``: subjects censored at ``t``
    and subjects experiencing the event at ``t`` are therefore both at risk just
    before the events of ``t`` occur. Events lower the survival probability,
    censorings only shrink the risk set of later times.
    """
    order = np.argsort(durations, kind="mergesort")
    times_sorted = durations[order]
    events_sorted = events[order]

    unique_times, inverse = np.unique(times_sorted, return_inverse=True)
    n_bins = unique_times.shape[0]
    counts = np.bincount(inverse, minlength=n_bins)
    n_events = np.bincount(inverse[events_sorted == 1], minlength=n_bins)
    n_censored = counts - n_events

    n_subjects = durations.shape[0]
    # Number at risk at each distinct time: everyone observed at or after the time.
    before = np.zeros(n_bins, dtype=np.int64)
    before[1:] = np.cumsum(counts)[:-1]
    at_risk = n_subjects - before

    with np.errstate(divide="ignore", invalid="ignore"):
        survival = np.cumprod(1.0 - n_events.astype(np.float64) / at_risk)

    return dict(
        times=unique_times,
        at_risk=at_risk.astype(np.int64),
        n_events=n_events.astype(np.int64),
        n_censored=n_censored.astype(np.int64),
        survival=survival,
    )


def _ordered_groups(group_values, category_order):
    """Order group values by the user-provided order, then by first appearance."""
    present = []
    for value in group_values:
        if value not in present:
            present.append(value)

    ordered = []
    if category_order:
        for value in category_order:
            if value in present and value not in ordered:
                ordered.append(value)
    for value in present:
        if value not in ordered:
            ordered.append(value)
    return ordered


def _group_label(value):
    return _MISSING_GROUP if _is_missing(value) else str(value)


def _group_color(value, color_map, color_sequence, sequence_offset):
    if value in color_map:
        return color_map[value], sequence_offset
    color = color_sequence[sequence_offset % len(color_sequence)]
    return color, sequence_offset + 1


def _make_hover_template(prefix, x_label):
    return (
        prefix + x_label + "=%{x}<br>"
        "At risk=%{customdata[0]}<br>"
        "Events=%{customdata[1]}<br>"
        "Censored=%{customdata[2]}<br>"
        "Survival probability=%{customdata[3]:.3f}"
        "<extra></extra>"
    )


def survival(
    data_frame=None,
    x=None,
    event=None,
    color=None,
    category_orders=None,
    labels=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
):
    """
    In a survival plot, right-censored time-to-event observations of
    `data_frame` are summarized with the non-parametric Kaplan–Meier estimator
    and drawn as a step line. Each distinct observation time contributes one
    point to the line; tick marks are overlaid at times where one or more
    observations are right-censored.

    The risk set of a time ``t`` contains every subject still observed at
    ``t``, i.e. every subject whose observation time is greater than or equal
    to ``t``. Subjects censored at ``t`` and subjects experiencing the event
    at ``t`` are therefore both counted in the risk set before the events of
    ``t`` are processed: events lower the survival probability, while
    censorings only reduce the number at risk for subsequent times. Groups in
    which no event is observed stay at a survival probability of 1.

    Parameters
    ----------
    data_frame: DataFrame or array-like or dict
        This argument needs to be passed for column names (and not keyword
        names) to be used. Array-like and dict are transformed internally to a
        pandas DataFrame. Optional: if missing, the arguments `x`, `event` and
        `color` can be used as array-likes directly.
    x: str or int or Series or array-like
        Observation time (follow-up duration) of each subject, as a column
        name in `data_frame` or an array-like. Values must be finite and
        non-negative numbers.
    event: str or int or Series or array-like
        Event indicator of each subject, as a column name in `data_frame` or
        an array-like. ``True``/``1`` means the event was observed at `x`,
        ``False``/``0`` means the observation was right-censored at `x`.
        Missing values and any other marker are rejected.
    color: str or int or Series or array-like
        Optional name of a column in `data_frame`, or an array-like, whose
        distinct values are used to draw one Kaplan–Meier curve per group.
    category_orders: dict with str keys and list of str values (default `{}`)
        By default, in Python 3.7+, the order of categorical values depends on
        the order in which these values first appear in `data_frame`. This
        argument is used to force a specific ordering of the `color` groups;
        the key is the column name and the value is the desired order. Groups
        that are not listed are appended in their order of first appearance.
    labels: dict with str keys and str values (default `{}`)
        By default, column names are used in axis and legend titles. This
        argument allows overriding this. The keys are column names and the
        values are the desired labels.
    color_discrete_sequence: list of valid CSS-colors
        Strings and valid CSS-colors from which the color of each group is
        chosen in order when `color` is set. Ignored if the group value is a
        key in `color_discrete_map`.
    color_discrete_map: dict with str keys and valid CSS-color values
        Used to override the color of specific groups, keyed by group value.
    title: str
        Figure title.
    subtitle: str
        Figure subtitle.
    template: str or dict or plotly.graph_objects.layout.Template instance
        The figure template name or definition.
    width: int
        Figure width in pixels.
    height: int
        Figure height in pixels.

    Returns
    -------
    fig: plotly.graph_objects.Figure
        A plain Figure object containing, per group, a step line of the
        Kaplan–Meier survival probability and a markers-only trace of the
        right-censoring times. Hovering a point shows the time, the number of
        subjects at risk, the number of events and censorings at that time and
        the survival probability.
    """
    if x is None:
        raise ValueError("Argument `x` (observation times) is required.")
    if event is None:
        raise ValueError("Argument `event` (event indicators) is required.")

    # Reuse the standard PX argument-processing machinery by letting `event`
    # ride along as the `y` coordinate: afterwards the three arguments are all
    # plain references to columns of a narwhals DataFrame.
    args = dict(
        data_frame=data_frame,
        x=x,
        y=event,
        color=color,
        category_orders=category_orders or {},
        labels=labels or {},
        color_discrete_sequence=color_discrete_sequence,
        color_discrete_map=color_discrete_map or {},
        template=template,
        title=title,
        width=width,
        height=height,
    )
    apply_default_cascade(args, constructor=go.Scatter)
    args = build_dataframe(args, go.Scatter)
    df = args["data_frame"]

    time_col = args["x"]
    event_col = args["y"]
    group_col = args["color"]

    if not df.get_column(time_col).dtype.is_numeric():
        raise ValueError(
            "Column '%s' of the data must contain numeric observation times." % time_col
        )

    durations = _coerce_durations(df.get_column(time_col).to_numpy())
    events = _coerce_events(df.get_column(event_col).to_numpy())

    if group_col is not None:
        # Normalize missing markers (None / NaN) to a single None token so that
        # they form one group rather than one group per missing value.
        group_values = [
            None if _is_missing(value) else value
            for value in df.get_column(group_col).to_list()
        ]
    else:
        group_values = [_ONE_GROUP] * durations.shape[0]

    n_subjects = durations.shape[0]
    x_label = get_label(args, time_col)

    fig = go.Figure()

    if n_subjects > 0:
        category_order = args["category_orders"].get(group_col, []) if group_col else []
        ordered = _ordered_groups(group_values, category_order)
        color_map = args["color_discrete_map"]
        color_sequence = args["color_discrete_sequence"]

        sequence_offset = 0
        for value in ordered:
            mask = np.array([v == value for v in group_values])

            estimate = _kaplan_meier(durations[mask], events[mask])
            color, sequence_offset = _group_color(
                value, color_map, color_sequence, sequence_offset
            )
            name = _group_label(value) if group_col is not None else _ONE_GROUP
            hover_prefix = "<b>%s</b><br>" % name if group_col is not None else ""
            hover_template = _make_hover_template(hover_prefix, x_label)

            # The estimator is 1 at time 0; every distinct observation time is
            # an explicit vertex so that its risk-set statistics are hoverable.
            line_x = np.concatenate(([0.0], estimate["times"]))
            line_y = np.concatenate(([1.0], estimate["survival"]))
            group_size = n_subjects if group_col is None else int(mask.sum())
            line_custom = np.column_stack(
                (
                    np.concatenate(([group_size], estimate["at_risk"])),
                    np.concatenate(([0], estimate["n_events"])),
                    np.concatenate(([0], estimate["n_censored"])),
                    np.concatenate(([1.0], estimate["survival"])),
                )
            )

            legend_group = name
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name=name,
                    legendgroup=legend_group,
                    showlegend=group_col is not None,
                    line=dict(shape="hv", color=color, width=2),
                    customdata=line_custom,
                    hovertemplate=hover_template,
                    cliponaxis=False,
                )
            )

            censor_points = estimate["n_censored"] > 0
            if np.any(censor_points):
                fig.add_trace(
                    go.Scatter(
                        x=estimate["times"][censor_points],
                        y=estimate["survival"][censor_points],
                        mode="markers",
                        name=name,
                        legendgroup=legend_group,
                        showlegend=False,
                        marker=dict(
                            symbol="cross",
                            size=8,
                            color=color,
                            line=dict(color=color, width=1),
                        ),
                        customdata=np.column_stack(
                            (
                                estimate["at_risk"][censor_points],
                                estimate["n_events"][censor_points],
                                estimate["n_censored"][censor_points],
                                estimate["survival"][censor_points],
                            )
                        ),
                        hovertemplate=hover_template,
                        cliponaxis=False,
                    )
                )

    layout = dict(
        xaxis=dict(title_text=x_label, rangemode="tozero"),
        yaxis=dict(title_text="Survival probability", rangemode="tozero"),
        margin=dict(t=60),
    )
    if width is not None:
        layout["width"] = width
    if height is not None:
        layout["height"] = height
    if title is not None:
        layout["title_text"] = title
        layout.pop("margin", None)
    if subtitle is not None:
        layout["title_subtitle_text"] = subtitle
    if group_col is not None and n_subjects > 0:
        layout["legend"] = dict(
            title_text=get_label(args, group_col), groupclick="togglegroup"
        )

    fig.update_layout(layout)
    if args["template"] is not None:
        fig.update_layout(template=args["template"], overwrite=True)
    return fig
