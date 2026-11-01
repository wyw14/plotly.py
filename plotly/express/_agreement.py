import numpy as np
import narwhals.stable.v1 as nw

import plotly.graph_objs as go


# Multiplier of the sample standard deviation used for the limits of agreement.
_LOA_MULTIPLIER = 1.96

# Compact number format used in hover tooltips (d3-format syntax).
_NUMBER_FORMAT = ":g"

# Internal column names used to build the temporary plotting frame. They are
# prefixed and suffixed to avoid clashing with columns of the user's data.
_X_COL = "__px_agreement_x__"
_Y_COL = "__px_agreement_y__"
_MEAN_COL = "__px_agreement_mean__"
_DIFF_COL = "__px_agreement_difference__"
_MASK_COL = "__px_agreement_mask__"

_MEAN_COLOR = "#4C78A8"
_LOA_COLOR = "#E45756"


def _wrap_data_frame(data_frame):
    """Turn any supported `data_frame` input into a narwhals DataFrame."""
    if nw.dependencies.is_pandas_like_dataframe(data_frame):
        return nw.from_native(data_frame, eager_only=True)
    if nw.dependencies.is_pandas_like_series(data_frame):
        return nw.from_native(data_frame, series_only=True).to_frame()
    wrapped = nw.from_native(
        data_frame, eager_or_interchange_only=True, pass_through=True
    )
    if isinstance(wrapped, nw.DataFrame):
        if nw.get_level(wrapped) == "interchange":
            # Materialize interchange-only objects (e.g. DuckDB) to PyArrow.
            wrapped = nw.from_native(wrapped.to_arrow(), eager_only=True)
        return wrapped
    if hasattr(data_frame, "toPandas"):
        # PySpark and friends.
        return nw.from_native(data_frame.toPandas(), eager_only=True)
    try:
        import pandas as pd
    except ImportError:
        raise NotImplementedError(
            "Pandas is required when `data_frame` is not a supported DataFrame "
            "type. Please provide a pandas, Polars or PyArrow DataFrame, or "
            "install pandas."
        )
    try:
        return nw.from_native(pd.DataFrame(data_frame), eager_only=True)
    except Exception:
        raise NotImplementedError(
            "Unable to convert `data_frame` of type %s to a pandas DataFrame. "
            "Please provide a supported dataframe type." % type(data_frame).__name__
        )


def _is_scalar(value):
    return isinstance(value, (str, bytes, int, float, bool)) or not hasattr(
        value, "__len__"
    )


def _resolve_field(df_input, value, argument_name, n, native_namespace):
    """Resolve a column reference or array-like into a length-`n` narwhals Series."""
    if value is None:
        return None
    if isinstance(value, str):
        if df_input is None:
            raise ValueError(
                "String or int arguments are only possible when a DataFrame or "
                "an array is provided in the `data_frame` argument. No "
                "DataFrame was provided, but argument '%s' is of type str."
                % argument_name
            )
        if value not in df_input.columns:
            raise ValueError(
                "Value of '%s' is not the name of a column in 'data_frame'. "
                "Expected one of %s but received: %s"
                % (argument_name, str(list(df_input.columns)), value)
            )
        return df_input.get_column(value)
    series = nw.from_native(value, series_only=True, pass_through=True)
    if not isinstance(series, nw.Series):
        if _is_scalar(value):
            if df_input is None:
                raise ValueError(
                    "String or int arguments are only possible when a DataFrame "
                    "or an array is provided in the `data_frame` argument. No "
                    "DataFrame was provided, but argument '%s' is of type %s."
                    % (argument_name, type(value).__name__)
                )
            raise ValueError(
                "Value of '%s' is not the name of a column in 'data_frame'. "
                "Expected one of %s but received: %s"
                % (argument_name, str(list(df_input.columns)), value)
            )
        try:
            series = nw.new_series(
                name=argument_name, values=value, native_namespace=native_namespace
            )
        except Exception:
            raise ValueError(
                "Argument '%s' could not be converted to a series of values."
                % argument_name
            )
    if len(series) != n:
        raise ValueError(
            "All arguments should have the same length. The length of argument "
            "`%s` is %d, whereas the length of the measurements is %d."
            % (argument_name, len(series), n)
        )
    return series


def _as_named_series(value, name, native_namespace):
    """Return a narwhals Series called `name` in `native_namespace`."""
    series = nw.from_native(value, series_only=True, pass_through=True)
    if isinstance(series, nw.Series):
        # Reuse the native series directly whenever it lives in the same backend.
        try:
            if nw.get_native_namespace(series) is native_namespace:
                return series.alias(name)
        except Exception:
            pass
        value = series.to_numpy()
    return nw.new_series(name=name, values=value, native_namespace=native_namespace)


def _to_float_array(series, column_label):
    """Cast a narwhals Series to a 1-D float64 numpy array."""
    try:
        converted = series.cast(nw.Float64())
    except Exception as err:
        raise ValueError(
            "Values of '%s' could not be converted to a numeric type. Please "
            "provide two columns of finite numeric measurements." % column_label
        ) from err
    array = np.asarray(converted.to_numpy(), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Values of '%s' must be one-dimensional." % column_label)
    return array


def _is_listlike(value):
    if isinstance(value, (str, bytes, bool)) or value is None:
        return False
    if isinstance(value, tuple) and len(value) == 2:
        return False
    return hasattr(value, "__len__")


def _normalize_hover_value(value):
    """Normalize one `hover_data` dict value into (visible, formatter, data).

    This mirrors the accepted Plotly Express forms:
    - True / False: show or hide the column;
    - a format string such as ":.2f": show the column with the format;
    - an array-like: on-board data to show, with the key as label;
    - a (format, data) tuple: on-board data with a format string.
    """
    if isinstance(value, tuple) and len(value) == 2:
        first, second = value
        visible = not (isinstance(first, bool) and not first)
        formatter = None if isinstance(first, bool) else first
        return visible, formatter, second if _is_listlike(second) else None
    if _is_listlike(value):
        return True, None, value
    if isinstance(value, bool):
        return value, None, None
    if isinstance(value, str):
        return True, value, None
    raise ValueError(
        "Values of `hover_data` must be booleans, format strings, array-likes "
        "or (format, array) tuples; received: %r" % (value,)
    )


def _format_number(value):
    return format(float(value), "g")


def agreement(
    data_frame=None,
    x=None,
    y=None,
    hover_name=None,
    hover_data=None,
    labels=None,
    title=None,
    template=None,
    width=None,
    height=None,
) -> go.Figure:
    """
    Draw a Bland–Altman agreement plot (also called a mean-difference plot)
    to compare two methods measuring the same set of objects.

    Each row with two non-missing measurements is represented by a mark whose
    horizontal position is the mean of the two measurements and whose vertical
    position is the difference between them (``x`` minus ``y``). Three
    horizontal reference lines are drawn: the mean difference (bias) and the
    limits of agreement, defined as the mean difference plus and minus 1.96
    times the sample standard deviation of the differences (computed with an
    ``n - 1`` denominator). Hovering a mark shows both original measurements,
    their mean and their difference.

    Rows for which either ``x`` or ``y`` is missing are skipped pairwise.
    Present but non-finite values (such as ``inf`` or ``-inf``), as well as
    non-numeric measurements, raise an error. At least two complete pairs are
    required. If all differences are identical the standard deviation is zero
    and the upper and lower limits of agreement coincide with the mean
    difference.

    Parameters
    ----------
    data_frame: DataFrame or array-like, optional
        A pandas, Polars or PyArrow DataFrame (or any dataframe supported by
        Narwhals), or an object convertible to a pandas DataFrame. If
        ``None``, ``x`` and ``y`` must be provided as array-likes of equal
        length.
    x: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas/polars Series
        or array_like of values. These are the first measurements; the
        difference plotted on the y axis is ``x`` minus ``y``.
    y: str or int or Series or array-like
        The second measurements, in the same format as `x`.
    hover_name: str or int or Series or array-like, optional
        Values to display in bold in the hover tooltip, in the same format as
        `x`. Rows skipped pairwise for missing measurements are also removed
        from the hover labels.
    hover_data: list of str or int, or dict with str keys, optional
        Extra columns to show in the hover tooltip. With a list, each entry is
        a column reference or an array-like of values. With a dict, keys are
        labels and values are ``True``/``False``, a d3 number format string
        (such as ``":.2f"``), an array-like of on-board values, or a
        ``(format, array)`` tuple. The original two measurements, their mean
        and their difference are always included.
    labels: dict with str keys and str values, optional
        Override the labels used in axis titles and hover tooltips, keyed by
        column name. Defaults to the column names, with ``"Mean of <x> and
        <y>"`` for the x axis and ``"<x> - <y>"`` for the y axis.
    title: str, optional
        The figure title.
    template: str or dict or plotly.graph_objects.layout.Template, optional
        The figure template to use.
    width: int, optional
        The figure width in pixels.
    height: int, optional
        The figure height in pixels.

    Returns
    -------
    go.Figure
        A standard :class:`plotly.graph_objects.Figure` containing a scatter
        trace of one mark per complete pair and three horizontal line shapes
        (mean difference and the two limits of agreement), which can be
        further customized as usual.

    Examples
    --------
    >>> import plotly.express as px
    >>> df = px.data.iris()
    >>> fig = px.agreement(df, x="sepal_length", y="petal_length")
    >>> fig.show()
    """
    from ._chart_types import scatter

    if x is None or y is None:
        raise ValueError("Both 'x' and 'y' are required to draw an agreement plot.")

    user_labels = dict(labels) if labels else {}

    df_input = _wrap_data_frame(data_frame) if data_frame is not None else None
    if df_input is not None:
        native_namespace = nw.get_native_namespace(df_input)
        n = len(df_input)
    else:
        if _is_scalar(x) or _is_scalar(y):
            raise ValueError(
                "String or int arguments are only possible when a DataFrame "
                "or an array is provided in the `data_frame` argument. No "
                "DataFrame was provided, but `x` and `y` must be array-likes "
                "of equal length."
            )
        try:
            import pandas as pd
        except ImportError:
            raise NotImplementedError(
                "Pandas is required when no `data_frame` is provided. Please "
                "pass a supported DataFrame or install pandas."
            )
        native_namespace = pd
        n = len(x)
        if len(y) != n:
            raise ValueError(
                "All arguments should have the same length. The length of "
                "argument `y` is %d, whereas the length of `x` is %d." % (len(y), n)
            )

    # Keep references to the *original* series (for hover display) and to the
    # float-converted arrays (for statistics), so that the input is never
    # mutated and original values/dtypes are preserved in the hover tooltip.
    x_series = _resolve_field(df_input, x, "x", n, native_namespace)
    y_series = _resolve_field(df_input, y, "y", n, native_namespace)

    x_label_default = x if isinstance(x, str) else (x_series.name or "x")
    y_label_default = y if isinstance(y, str) else (y_series.name or "y")

    x_array = _to_float_array(x_series, str(x_label_default))
    y_array = _to_float_array(y_series, str(y_label_default))

    # Missing values (NaN/NaT converted to NaN) are skipped pairwise.
    x_missing = np.isnan(x_array)
    y_missing = np.isnan(y_array)

    # Present-but-non-finite values (inf/-inf) must be reported explicitly,
    # even on rows where the other measurement is missing.
    for values, missing, label in [
        (x_array, x_missing, x_label_default),
        (y_array, y_missing, y_label_default),
    ]:
        present = ~missing
        if present.any() and not np.isfinite(values[present]).all():
            raise ValueError(
                "Non-finite values (such as 'inf' or '-inf') are not allowed; "
                "column '%s' contains a non-finite, non-missing value." % label
            )

    valid = ~(x_missing | y_missing)
    n_valid = int(valid.sum())
    if n_valid < 2:
        raise ValueError(
            "At least two complete pairs of non-missing measurements are "
            "required to draw an agreement plot, but only %d complete pair(s) "
            "were provided." % n_valid
        )

    x_valid = x_array[valid]
    y_valid = y_array[valid]
    differences = x_valid - y_valid

    mean_difference = float(differences.mean())
    # Sample standard deviation, using the n - 1 denominator. When every
    # difference is identical this is exactly zero (no division by zero).
    sd_difference = float(differences.std(ddof=1))
    upper_limit = mean_difference + _LOA_MULTIPLIER * sd_difference
    lower_limit = mean_difference - _LOA_MULTIPLIER * sd_difference

    # Default labels, overridable through `labels`.
    x_label = str(user_labels.get(x_label_default, x_label_default))
    y_label = str(user_labels.get(y_label_default, y_label_default))
    mean_label = "Mean of %s and %s" % (x_label, y_label)
    difference_label = "%s - %s" % (x_label, y_label)

    # Resolve the optional hover fields. Each entry is an ordered
    # (internal name, label, visible, formatter, series) tuple.
    extra_hover = []
    hover_name_name = None
    if hover_name is not None:
        hover_name_series = _resolve_field(
            df_input, hover_name, "hover_name", n, native_namespace
        )
        hover_name_name = "__px_agreement_hover_name__"
        extra_hover.append((hover_name_name, None, True, None, hover_name_series))

    hover_counter = [0]

    def _add_extra_hover(reference, label, visible, formatter):
        if not visible or reference is None:
            return
        series = _resolve_field(df_input, reference, "hover_data", n, native_namespace)
        if series is None:
            return
        internal = "__px_agreement_hover_%d__" % hover_counter[0]
        hover_counter[0] += 1
        if label is None:
            label = (
                reference
                if isinstance(reference, str)
                else (series.name or "hover_data")
            )
        label = str(user_labels.get(label, label))
        extra_hover.append((internal, label, True, formatter, series))

    if hover_data is not None:
        if isinstance(hover_data, dict):
            for key, raw_value in hover_data.items():
                visible, formatter, onboard = _normalize_hover_value(raw_value)
                if onboard is not None:
                    # On-board data keyed by `key`, possibly with a formatter.
                    _add_extra_hover(onboard, str(key), visible, formatter)
                else:
                    # Reference to a column of `data_frame`; users may hide or
                    # reformat it, except for the four mandatory fields which
                    # are always shown by this function.
                    if key in (x_label_default, y_label_default):
                        continue
                    _add_extra_hover(key, str(key), visible, formatter)
        elif isinstance(hover_data, str):
            _add_extra_hover(hover_data, None, True, None)
        else:
            try:
                entries = list(hover_data)
            except TypeError:
                raise ValueError(
                    "`hover_data` must be a list of column references or a "
                    "dict of labels to column references or values."
                )
            for entry in entries:
                if isinstance(entry, tuple) and len(entry) == 2:
                    reference, formatter = entry
                    _add_extra_hover(reference, None, True, formatter)
                else:
                    _add_extra_hover(entry, None, True, None)

    # Assemble a full-length (pre-filtering) frame in the input backend. The
    # original measurement columns are carried through untouched so that the
    # hover displays the original values and dtypes.
    series_by_name = {
        _X_COL: _as_named_series(x_series, _X_COL, native_namespace),
        _Y_COL: _as_named_series(y_series, _Y_COL, native_namespace),
        _MEAN_COL: nw.new_series(
            name=_MEAN_COL,
            values=np.where(valid, (x_array + y_array) / 2.0, np.nan),
            native_namespace=native_namespace,
        ),
        _DIFF_COL: nw.new_series(
            name=_DIFF_COL,
            values=np.where(valid, x_array - y_array, np.nan),
            native_namespace=native_namespace,
        ),
    }
    for internal, _label, _visible, _formatter, series in extra_hover:
        series_by_name[internal] = _as_named_series(series, internal, native_namespace)
    series_by_name[_MASK_COL] = nw.new_series(
        name=_MASK_COL, values=valid, native_namespace=native_namespace
    )

    full_frame = nw.from_dict(series_by_name)
    # Filtering with the boolean mask removes incomplete pairs while keeping
    # the original row order of the remaining rows.
    plot_frame = full_frame.filter(nw.col(_MASK_COL)).drop(_MASK_COL)

    # NB: for column references, Plotly Express expects a format string rather
    # than a (bool, format) tuple (whose second element would be mistaken for
    # on-board data).
    scatter_hover_data = {
        _X_COL: _NUMBER_FORMAT,
        _Y_COL: _NUMBER_FORMAT,
        _MEAN_COL: _NUMBER_FORMAT,
        _DIFF_COL: _NUMBER_FORMAT,
    }
    for internal, _label, _visible, formatter, _series in extra_hover:
        if internal == hover_name_name:
            continue
        scatter_hover_data[internal] = formatter if formatter is not None else True

    scatter_labels = {
        _X_COL: x_label,
        _Y_COL: y_label,
        _MEAN_COL: mean_label,
        _DIFF_COL: difference_label,
    }
    for internal, label, _visible, _formatter, _series in extra_hover:
        if label is not None:
            scatter_labels[internal] = label

    fig = scatter(
        plot_frame.to_native(),
        x=_MEAN_COL,
        y=_DIFF_COL,
        hover_name=hover_name_name,
        hover_data=scatter_hover_data,
        labels=scatter_labels,
        title=title,
        template=template,
        width=width,
        height=height,
    )

    # Three horizontal reference lines, drawn as full-width layout shapes so
    # they remain visible even when all pairs share the same mean.
    fig.add_hline(
        y=mean_difference,
        line_color=_MEAN_COLOR,
        line_width=2,
        annotation_text="Mean difference: %s" % _format_number(mean_difference),
        annotation_position="top left",
        annotation_font_color=_MEAN_COLOR,
    )
    fig.add_hline(
        y=upper_limit,
        line_color=_LOA_COLOR,
        line_dash="dash",
        annotation_text="Upper limit (+1.96 SD): %s" % _format_number(upper_limit),
        annotation_position="top right",
        annotation_font_color=_LOA_COLOR,
    )
    fig.add_hline(
        y=lower_limit,
        line_color=_LOA_COLOR,
        line_dash="dash",
        annotation_text="Lower limit (-1.96 SD): %s" % _format_number(lower_limit),
        annotation_position="bottom right",
        annotation_font_color=_LOA_COLOR,
    )

    return fig
