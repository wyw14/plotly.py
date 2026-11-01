import numpy as np
import narwhals.stable.v1 as nw

import plotly.graph_objs as go

from ._core import apply_default_cascade
from ._doc import make_docstring


def _to_nw_dataframe(data_frame):
    """Convert any supported data_frame input into a narwhals DataFrame."""
    if data_frame is None:
        return None
    if nw.dependencies.is_pandas_like_dataframe(data_frame):
        return nw.from_native(data_frame, eager_only=True)
    if nw.dependencies.is_pandas_like_series(data_frame):
        return nw.from_native(data_frame, series_only=True).to_frame()
    frame = nw.from_native(
        data_frame, eager_or_interchange_only=True, pass_through=True
    )
    if isinstance(frame, nw.DataFrame):
        if nw.get_level(frame) == "interchange":
            frame = nw.from_native(frame.to_arrow(), eager_only=True)
        return frame
    # Other objects (dict, list, ...): fall back to a pandas DataFrame
    try:
        import pandas as pd

        return nw.from_native(pd.DataFrame(data_frame))
    except Exception:
        raise ValueError(
            "Could not parse `data_frame`: please provide a supported DataFrame "
            "type, or pass the values through the `x` argument instead."
        )


def _array_from_column_ref(frame, ref, arg_name):
    """Resolve a single column reference or array_like/Series to a numpy array."""
    is_column_ref = isinstance(ref, str) or (
        isinstance(ref, (int, np.integer)) and not isinstance(ref, bool)
    )
    if is_column_ref:
        if frame is None:
            raise ValueError(
                "Argument `%s` was given as the column reference %r but no "
                "`data_frame` was provided." % (arg_name, ref)
            )
        if isinstance(ref, str):
            column_name = ref
            if column_name not in frame.columns:
                raise ValueError(
                    "Value of '%s' used for `%s` is not the name of a column "
                    "in `data_frame`, whose columns are %s."
                    % (column_name, arg_name, list(frame.columns))
                )
        else:
            try:
                column_name = frame.columns[ref]
            except IndexError:
                raise ValueError(
                    "Column index %d used for `%s` is out of range for "
                    "`data_frame`, which has %d columns."
                    % (ref, arg_name, len(frame.columns))
                )
        return frame.get_column(column_name).to_numpy(), str(column_name)

    series = nw.from_native(ref, series_only=True, pass_through=True)
    if isinstance(series, nw.Series):
        return series.to_numpy(), (
            str(series.name) if series.name is not None else None
        )
    return np.asarray(ref), None


def _to_float(array, arg_name):
    """Convert an array of values to a 1-D float64 numpy array."""
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(
            "Argument `%s` must contain numeric values, but values that could "
            "not be converted to numbers were found." % arg_name
        )
    if array.ndim != 1:
        raise ValueError(
            "Argument `%s` must be one-dimensional, but an array with %d "
            "dimensions was provided." % (arg_name, array.ndim)
        )
    return array


def _extract_lorenz_inputs(data_frame, x, weights):
    """Return (values, weights, value_label) as validated numpy arrays.

    Missing values are skipped pairwise: a row is dropped if either its value
    or its weight is missing. The input data is never modified.
    """
    if x is None:
        raise ValueError(
            "Must pass the `x` argument: the column or array_like of "
            "non-negative values whose cumulative distribution is displayed."
        )

    frame = _to_nw_dataframe(data_frame)
    values, value_label = _array_from_column_ref(frame, x, "x")
    values = _to_float(values, "x")

    if weights is None:
        weights_array = np.ones_like(values)
    else:
        weights_array, _ = _array_from_column_ref(frame, weights, "weights")
        weights_array = _to_float(weights_array, "weights")
        if weights_array.shape != values.shape:
            raise ValueError(
                "`x` and `weights` must have the same length, but got %d and "
                "%d values respectively." % (values.shape[0], weights_array.shape[0])
            )

    # Missing values (NaN) and the corresponding row are skipped pairwise.
    valid = ~(np.isnan(values) | np.isnan(weights_array))
    values = values[valid]
    weights_array = weights_array[valid]

    if values.size == 0:
        raise ValueError(
            "No valid observations remain after dropping rows with missing "
            "values or weights."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Argument `x` must contain only finite values, but non-finite "
            "values (e.g. infinity) were found."
        )
    if np.any(values < 0):
        raise ValueError(
            "Argument `x` must contain only non-negative values, but negative "
            "values were found."
        )
    if not np.all(np.isfinite(weights_array)):
        raise ValueError(
            "Argument `weights` must contain only finite values, but "
            "non-finite weights (e.g. infinity) were found."
        )
    if np.any(weights_array <= 0):
        raise ValueError(
            "Argument `weights` must contain only strictly positive values, "
            "but zero or negative weights were found."
        )

    total_weighted_value = float(np.sum(weights_array * values))
    if total_weighted_value == 0:
        raise ValueError(
            "The total weighted value of `x` is zero, so a Lorenz curve and "
            "Gini coefficient cannot be computed."
        )

    if value_label is None:
        value_label = "value"
    return values, weights_array, value_label, total_weighted_value


def _lorenz_curve(values, weights, total_weighted_value):
    """Sort stably and build cumulative population/value shares plus Gini.

    The Gini coefficient is obtained from the area between the Lorenz curve
    and the line of perfect equality using the trapezoidal rule:
    ``Gini = 1 - 2 * integral(L(p), p)``.
    """
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]

    cumulative_population = np.concatenate(([0.0], np.cumsum(sorted_weights)))
    cumulative_value = np.concatenate(
        ([0.0], np.cumsum(sorted_weights * sorted_values))
    )

    population_share = cumulative_population / cumulative_population[-1]
    value_share = cumulative_value / total_weighted_value
    # Pin the endpoints so the curve exactly starts at (0, 0) and ends at (1, 1)
    population_share[0] = 0.0
    value_share[0] = 0.0
    population_share[-1] = 1.0
    value_share[-1] = 1.0

    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    area_under_curve = float(trapezoid(value_share, population_share))
    gini = 1.0 - 2.0 * area_under_curve
    # Clean up floating-point drift around the theoretical bounds
    if np.isclose(gini, 0.0, atol=1e-12):
        gini = 0.0
    elif np.isclose(gini, 1.0, atol=1e-12):
        gini = 1.0

    return population_share, value_share, float(gini)


def lorenz(
    data_frame=None,
    x=None,
    weights=None,
    labels=None,
    title=None,
    subtitle=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    In a Lorenz curve plot, rows of `data_frame` are sorted in ascending order
    by the non-negative values in `x` (using a stable sort) and drawn as a curve
    of the cumulative weighted share of `x` (y axis) against the cumulative
    share of the observations (x axis, using `weights` if provided). The curve
    starts at (0, 0) and ends at (1, 1) and is drawn together with the
    45-degree line of perfect equality. The figure is annotated with the Gini
    coefficient, computed from the area between the two curves using the
    trapezoidal rule. Rows with missing values or weights are dropped pairwise.
    """
    args = locals()
    apply_default_cascade(args, constructor=go.Scatter)

    (
        values,
        weights_array,
        value_label,
        total_weighted_value,
    ) = _extract_lorenz_inputs(data_frame, x, weights)
    population_share, value_share, gini = _lorenz_curve(
        values, weights_array, total_weighted_value
    )

    labels_map = args["labels"]
    x_label = labels_map.get("x") or "Cumulative share of population"
    y_label = labels_map.get("y") or "Cumulative share of %s" % value_label

    curve_hover_template = (
        "<b>%%{data.name}</b><br>%s: %%{x:.1%%}<br>%s: %%{y:.1%%}<extra></extra>"
        % (x_label, y_label)
    )
    line_hover_template = (
        "<b>%%{data.name}</b><br>%s: %%{x:.1%%}<br>%s: %%{y:.1%%}<extra></extra>"
        % (x_label, y_label)
    )

    lorenz_trace = go.Scatter(
        x=population_share,
        y=value_share,
        mode="lines",
        name="Lorenz curve",
        hovertemplate=curve_hover_template,
    )
    equality_trace = go.Scatter(
        x=[0.0, 1.0],
        y=[0.0, 1.0],
        mode="lines",
        name="Perfect equality",
        line=dict(dash="dash", color="rgba(128, 128, 128, 0.8)"),
        hovertemplate=line_hover_template,
    )

    gini_annotation = dict(
        text="Gini coefficient: %.4f" % gini,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        showarrow=False,
    )

    layout_patch = dict(
        xaxis=dict(
            title_text=x_label,
            range=[0.0, 1.0],
            tickformat=".0%",
            constrain="domain",
        ),
        yaxis=dict(
            title_text=y_label,
            range=[0.0, 1.0],
            tickformat=".0%",
            scaleanchor="x",
            constrain="domain",
        ),
        annotations=[gini_annotation],
    )
    if args["title"]:
        layout_patch["title_text"] = args["title"]
    elif args["template"].layout.margin.t is None:
        layout_patch["margin"] = {"t": 60}
    if args["subtitle"]:
        layout_patch["title_subtitle_text"] = args["subtitle"]
    for dimension in ["height", "width"]:
        if args[dimension]:
            layout_patch[dimension] = args[dimension]

    fig = go.Figure(data=[lorenz_trace, equality_trace])
    fig.update_layout(layout_patch)
    fig.update_layout(template=args["template"], overwrite=True)
    return fig


lorenz.__doc__ = make_docstring(
    lorenz,
    append_dict=dict(
        x=[
            "In a `lorenz` plot, these are the non-negative values whose "
            "cumulative distribution is displayed. Rows with missing values "
            "are dropped pairwise with `weights`."
        ],
    ),
)
