from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Time-Series Feature Analysis", page_icon="📈", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parents[0]
DEFAULT_PATH = PROJECT_ROOT/"data"/"raw"/"electricity_demand.csv"

def read_data(upload, path_text):
    if upload is not None:
        name = upload.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(upload)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(upload)
        raise ValueError("Unsupported file type. Upload CSV or Excel.")
    path = Path(path_text).expanduser() if path_text.strip() else DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}. Upload a CSV/Excel file or correct the path.")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Choose a CSV or Excel file.")


def infer_interval(ts):
    unique = pd.Series(pd.to_datetime(ts, errors="coerce", utc=True).dropna().drop_duplicates().sort_values())
    deltas = unique.diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return None, deltas
    modes = deltas.mode()
    return (modes.iloc[0] if not modes.empty else deltas.median()), deltas


def duration_label(delta):
    seconds = int(round(pd.Timedelta(delta).total_seconds()))
    for unit, name in [(604800, "week"), (86400, "day"), (3600, "hour"), (60, "minute")]:
        if seconds and seconds % unit == 0:
            n = seconds // unit
            return f"{n} {name}{'' if n == 1 else 's'}"
    return f"{seconds} seconds"


def add_calendar_features(df, time_col):
    out = df.copy()
    ts = out[time_col]
    out["hour"] = ts.dt.hour
    out["minute"] = ts.dt.minute
    out["day_of_week"] = ts.dt.dayofweek
    out["day_of_month"] = ts.dt.day
    out["month"] = ts.dt.month
    out["day_of_year"] = ts.dt.dayofyear
    out["week_of_year"] = ts.dt.isocalendar().week.astype("int64")
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    fractional_hour = ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600
    out["hour_sin"] = np.sin(2 * np.pi * fractional_hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * fractional_hour / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    out["year_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
    out["year_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
    return out


def duration_lag(df, time_col, value_col, duration, tolerance):
    """Match each row to the latest observation at or before timestamp - duration."""
    left = pd.DataFrame({"_row": np.arange(len(df)), "_target": df[time_col] - duration}).sort_values("_target")
    right = df[[time_col, value_col]].rename(columns={time_col: "_source", value_col: "_lag"}).sort_values("_source")
    joined = pd.merge_asof(
        left, right, left_on="_target", right_on="_source", direction="backward",
        tolerance=tolerance, allow_exact_matches=True
    )
    return pd.Series(joined.sort_values("_row")["_lag"].to_numpy(), index=df.index)


def rolling_feature(df, time_col, value_col, duration, statistic):
    """Trailing time-window statistic, excluding the current observation."""
    series = df.set_index(time_col)[value_col].sort_index()
    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(1, unit="ns")
    roll = shifted.rolling(duration, min_periods=1, closed="right")
    fn = {
        "Mean": roll.mean,
        "Standard deviation": roll.std,
        "Minimum": roll.min,
        "Maximum": roll.max,
    }[statistic]
    result = fn()
    result.index = result.index - pd.Timedelta(1, unit="ns")
    result = result.reindex(series.index)
    return pd.Series(result.to_numpy(), index=df.index)


st.title("📈 Time-Series Feature Analysis")
st.caption("Analyze time-series data at different sampling intervals using calendar, lag, and rolling-window features.")
with st.sidebar:
    st.header("Data source")
    upload = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    path_text = st.text_input("Or enter file path", str(DEFAULT_PATH))

try:
    raw = read_data(upload, path_text)
except Exception as exc:
    st.info("Upload a CSV/Excel file or set the path to your dataset in the sidebar.")
    st.error(str(exc))
    st.stop()
if raw.empty:
    st.error("The selected file has no rows.")
    st.stop()

with st.sidebar:
    st.header("Columns")
    cols = list(raw.columns)
    timestamp_candidates = [c for c in cols if c.lower() in {"utc_timestamp", "timestamp", "datetime", "date", "time"}]
    time_default = timestamp_candidates[0] if timestamp_candidates else cols[0]
    time_col = st.selectbox("Timestamp column", cols, index=cols.index(time_default))
    possible_values = [c for c in cols if c != time_col]
    numeric_cols = [c for c in possible_values if pd.api.types.is_numeric_dtype(raw[c])]
    # Include numeric-looking text columns, while excluding columns that cannot be parsed as numbers.
    for col in possible_values:
        if col not in numeric_cols:
            converted = pd.to_numeric(raw[col], errors="coerce")
            if converted.notna().sum() >= 2:
                numeric_cols.append(col)
    if not numeric_cols:
        st.error("No numeric or numeric-looking value column is available.")
        st.stop()
    value_candidates = [c for c in numeric_cols if c.lower() in {"de_load_actual_entsoe_transparency", "demand", "load", "value", "target"}]
    value_default = value_candidates[0] if value_candidates else numeric_cols[0]
    value_col = st.selectbox("Value column", numeric_cols, index=numeric_cols.index(value_default))

# Parse and clean selected columns. Naive timestamps are interpreted as UTC.
df = raw.copy()
df[time_col] = pd.to_datetime(df[time_col],format="%d-%m-%Y %H:%M", errors="coerce", utc=True)
df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
invalid_time_count = int(df[time_col].isna().sum())
non_numeric_count = int(df[value_col].isna().sum())
duplicate_count = int(df[time_col].duplicated().sum())
df = df.dropna(subset=[time_col, value_col]).sort_values(time_col)
df = df.drop_duplicates(subset=[time_col], keep="last").reset_index(drop=True)
if len(df) < 2:
    st.error("At least two valid timestamp/value rows are needed.")
    st.stop()
interval, all_deltas = infer_interval(df[time_col])
if interval is None:
    st.error("Unable to infer a sampling interval from the selected timestamps.")
    st.stop()

with st.sidebar:
    st.header("Time settings")
    st.write(f"**Detected interval:** {duration_label(interval)}")
    min_date, max_date = df[time_col].min().date(), df[time_col].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    aggregation = st.selectbox("Resampling", ["Keep original interval", "Mean", "Sum", "Median", "Minimum", "Maximum", "Last"])
    rule = st.selectbox("Resample interval", ["1min", "2min", "4min", "5min", "10min", "15min", "30min", "1h", "3h", "6h", "12h", "1D", "1W"], index=6, disabled=aggregation == "Keep original interval")

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range
start = pd.Timestamp(start_date, tz="UTC")
end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
view = df[(df[time_col] >= start) & (df[time_col] <= end)].copy()
if aggregation != "Keep original interval":
    indexed = view.set_index(time_col)[[value_col]].sort_index()
    methods = {
        "Mean": lambda x: x.resample(rule).mean(),
        "Sum": lambda x: x.resample(rule).sum(min_count=1),
        "Median": lambda x: x.resample(rule).median(),
        "Minimum": lambda x: x.resample(rule).min(),
        "Maximum": lambda x: x.resample(rule).max(),
        "Last": lambda x: x.resample(rule).last(),
    }
    view = methods[aggregation](indexed).dropna().reset_index()
if len(view) < 2:
    st.warning("Select a wider date range or a dataset with more observations.")
    st.stop()
view = view.sort_values(time_col).reset_index(drop=True)
view_interval, view_deltas = infer_interval(view[time_col])
view_interval = view_interval or interval
features = add_calendar_features(view, time_col)
gap_count = int((view[time_col].diff().dropna() > view_interval * 1.5).sum())

st.subheader("Overview")
a, b, c, d, e = st.columns(5)
a.metric("Observations", f"{len(view):,}")
b.metric("Typical interval", duration_label(view_interval))
c.metric("Mean", f"{view[value_col].mean():,.3f}")
d.metric("Standard deviation", f"{view[value_col].std():,.3f}")
e.metric("Detected gaps", f"{gap_count:,}")
trend_tab, patterns_tab, lag_tab, corr_tab, quality_tab = st.tabs(["Trend", "Time patterns", "Lag analysis", "Feature correlation", "Data quality"])

with trend_tab:
    st.markdown("### Time-series trend")

    # Add a NaN separator immediately before each observation following a
    # large timestamp gap. This prevents Plotly from drawing misleading
    # diagonal lines across periods with no observations.
    plot_df = view[[time_col, value_col]].copy().sort_values(time_col).reset_index(drop=True)
    gap_positions = np.flatnonzero(
        plot_df[time_col].diff().gt(view_interval * 1.5).to_numpy()
    )
    if len(gap_positions):
        separators = []
        for position in gap_positions:
            separators.append({
                time_col: plot_df.loc[position, time_col] - pd.Timedelta(nanoseconds=1),
                value_col: np.nan,
            })
        plot_df = pd.concat(
            [plot_df, pd.DataFrame(separators)], ignore_index=True
        ).sort_values(time_col).reset_index(drop=True)

    fig = px.line(
        plot_df, x=time_col, y=value_col,
        title=f"{value_col} over time",
        labels={time_col: "Timestamp", value_col: value_col},
    )
    fig.update_traces(connectgaps=False)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Value distribution")
    st.plotly_chart(
        px.histogram(view, x=value_col, nbins=50, marginal="box"),
        use_container_width=True,
    )

with patterns_tab:
    st.markdown("### Average value by hour")
    hourly = features.groupby("hour", as_index=False)[value_col].mean()
    fig = px.line(hourly, x="hour", y=value_col, markers=True)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        weekday = features.groupby("day_of_week", as_index=False)[value_col].mean()
        weekday["day"] = weekday["day_of_week"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
        st.plotly_chart(px.bar(weekday, x="day", y=value_col, category_orders={"day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]}, title="Average by weekday"), use_container_width=True)
    with right:
        monthly = features.groupby("month", as_index=False)[value_col].mean()
        fig = px.bar(monthly, x="month", y=value_col, title="Average by month")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)
    heat = features.pivot_table(index="day_of_week", columns="hour", values=value_col, aggfunc="mean").reindex(range(7))
    heat.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    st.plotly_chart(px.imshow(heat, aspect="auto", labels={"x": "Hour", "y": "Weekday", "color": "Mean"}, title="Hour × weekday heatmap"), use_container_width=True)

with lag_tab:
    st.markdown("### Duration-based lag features")
    st.caption("Only lags of 1 hour or longer are available. The correlation table compares selected lags, while the scatter plot displays one selected lag at a time. Duration lags match the latest available observation at or before the target time; unmatched lags remain blank.")
    # Only offer lags of one hour or longer.
    lag_options = {
        "1 hour": pd.Timedelta(hours=1),
        "6 hours": pd.Timedelta(hours=6),
        "12 hours": pd.Timedelta(hours=12),
        "24 hours": pd.Timedelta(hours=24),
        "7 days": pd.Timedelta(days=7),
    }
    selected_lags = st.multiselect(
        "Select lag durations (minimum 1 hour)",
        list(lag_options),
        default=["1 hour", "24 hours", "7 days"],
    )
    lag_data = pd.DataFrame({"Timestamp": view[time_col], "Actual": view[value_col]})
    summary = []
    # One full detected interval is used as tolerance, which supports intervals such as 4 minutes
    # when the requested duration (e.g. 15 minutes) does not align exactly to the sample grid.
    tolerance = max(view_interval, pd.Timedelta(seconds=1))
    for label in selected_lags:
        lag_values = duration_lag(
            view, time_col, value_col, lag_options[label], tolerance
        )
        col_name = f"Lag: {label}"
        lag_data[col_name] = lag_values
        valid = lag_data[["Actual", col_name]].dropna()
        corr = valid["Actual"].corr(valid[col_name]) if len(valid) >= 2 and valid["Actual"].nunique() > 1 and valid[col_name].nunique() > 1 else np.nan
        summary.append({"Lag": label, "Matched observations": int(len(valid)), "Correlation": corr})
    if summary:
        st.dataframe(pd.DataFrame(summary).style.format({"Correlation": "{:.4f}"}), use_container_width=True, hide_index=True)

        # Plot only one selected lag at a time to prevent all lag clouds overlapping.
        plot_lag_label = st.selectbox(
            "Lag to display in scatter plot",
            options=selected_lags,
            index=0,
            key="scatter_lag_selection",
        )
        plot_col = f"Lag: {plot_lag_label}"
        plot_df = lag_data[["Timestamp", "Actual", plot_col]].dropna().rename(
            columns={plot_col: "Lagged value"}
        )
        if len(plot_df) > 5000:
            plot_df = plot_df.sample(5000, random_state=42)
        if not plot_df.empty:
            fig = px.scatter(
                plot_df,
                x="Lagged value",
                y="Actual",
                opacity=0.4,
                render_mode="webgl",
                title=f"Actual vs lagged value — {plot_lag_label}",
                labels={"Lagged value": "Lagged value", "Actual": "Actual value"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No matching observations are available for this lag in the selected date range.")
        st.download_button("Download lag features CSV", lag_data.to_csv(index=False).encode("utf-8"), "lag_features.csv", "text/csv")
    else:
        st.info("Select one or more lags.")

    st.markdown("### Rolling-window feature")
    rolling_options = {
        "1 hour": pd.Timedelta(hours=1),
        "6 hours": pd.Timedelta(hours=6),
        "24 hours": pd.Timedelta(hours=24),
        "7 days": pd.Timedelta(days=7),
    }
    left, right = st.columns(2)
    roll_label = left.selectbox(
        "Window duration", list(rolling_options), index=2,
        key="rolling_window_duration",
    )
    statistic = right.selectbox(
        "Statistic", ["Mean", "Standard deviation", "Minimum", "Maximum"],
        key="rolling_statistic",
    )
    rolling_values = rolling_feature(
        view, time_col, value_col, rolling_options[roll_label], statistic
    )
    rolling_column = f"Rolling {statistic} ({roll_label})"
    rolling_df = pd.DataFrame({
        "Timestamp": view[time_col],
        "Actual": view[value_col],
        rolling_column: rolling_values,
    })

    st.markdown(f"#### Actual values vs. rolling {statistic.lower()}")
    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(
        x=rolling_df["Timestamp"], y=rolling_df["Actual"],
        mode="lines", name="Actual", line=dict(width=1), opacity=0.45,
    ))
    fig_rolling.add_trace(go.Scatter(
        x=rolling_df["Timestamp"], y=rolling_df[rolling_column],
        mode="lines", name=rolling_column, line=dict(width=2),
        connectgaps=False,
    ))
    fig_rolling.update_layout(
        title=f"Actual values and rolling {statistic.lower()} — {roll_label}",
        xaxis_title="Timestamp",
        yaxis_title=("Standard deviation" if statistic == "Standard deviation" else value_col),
        hovermode="x unified", legend_title="Series",
    )
    st.plotly_chart(fig_rolling, use_container_width=True)

    if statistic == "Standard deviation":
        st.markdown("#### Rolling volatility")
        fig_std = px.line(
            rolling_df, x="Timestamp", y=rolling_column,
            title=f"Rolling standard deviation — {roll_label}",
            labels={rolling_column: "Standard deviation"},
        )
        fig_std.update_layout(hovermode="x unified")
        st.plotly_chart(fig_std, use_container_width=True)

    with st.expander("View rolling feature data"):
        st.dataframe(rolling_df.tail(1000), use_container_width=True, hide_index=True)

    st.download_button(
        "Download rolling feature CSV",
        rolling_df.to_csv(index=False).encode("utf-8"),
        "rolling_features.csv", "text/csv",
    )

with corr_tab:
    # Add selected duration-based lags and rolling features to the feature-correlation view.
    corr_data = features.copy()
    for label, duration in lag_options.items():
        safe_name = label.lower().replace(" ", "_")
        corr_data[f"lag_{safe_name}"] = duration_lag(
            view, time_col, value_col, duration,
            max(view_interval, pd.Timedelta(seconds=1)),
        )
    corr_data["rolling_mean_24_hours"] = rolling_feature(view, time_col, value_col, pd.Timedelta(hours=24), "Mean")
    corr_data["rolling_std_24_hours"] = rolling_feature(view, time_col, value_col, pd.Timedelta(hours=24), "Standard deviation")
    corr_features = corr_data.select_dtypes(include=[np.number]).copy()
    corr_features = corr_features.loc[:, corr_features.nunique(dropna=True) > 1]
    if corr_features.shape[1] >= 2 and value_col in corr_features.columns:
        matrix = corr_features.corr(numeric_only=True)
        st.plotly_chart(px.imshow(matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Pearson correlation matrix"), use_container_width=True)
        target_corr = matrix[value_col].drop(value_col, errors="ignore").dropna()
        target_corr = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)
        table = target_corr.rename("Pearson correlation").reset_index()
        table.columns = ["Feature", "Pearson correlation"]
        st.dataframe(table.style.format({"Pearson correlation": "{:.4f}"}), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough varying numeric features to calculate correlations.")

with quality_tab:
    quality = pd.DataFrame([
        {"Check": "Rows in source file", "Result": f"{len(raw):,}"},
        {"Check": "Rows after cleaning", "Result": f"{len(df):,}"},
        {"Check": "Invalid timestamps", "Result": f"{invalid_time_count:,}"},
        {"Check": "Missing/non-numeric selected values", "Result": f"{non_numeric_count:,}"},
        {"Check": "Duplicate timestamps in source", "Result": f"{duplicate_count:,}"},
        {"Check": "Gaps > 1.5 × typical interval in selected range", "Result": f"{gap_count:,}"},
    ])
    st.dataframe(quality, use_container_width=True, hide_index=True)
    if not all_deltas.empty:
        counts = all_deltas.value_counts().head(20).sort_index()
        delta_df = pd.DataFrame({"Interval": [duration_label(x) for x in counts.index], "Count": counts.values})
        st.plotly_chart(px.bar(delta_df, x="Interval", y="Count", title="Most common sampling intervals"), use_container_width=True)
    st.download_button("Download cleaned selected data", view.to_csv(index=False).encode("utf-8"), "cleaned_time_series.csv", "text/csv")

st.markdown("---")
st.caption("Timestamps are interpreted as UTC. Convert to your local timezone before interpreting hour/day patterns if needed. Resampling aggregates observations; it does not recover missing data.")
# Common feature engineering for all stations.\n