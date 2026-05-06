import typing
import warnings

import lifelines
import numpy as np
import pandas as pd
import toolz as tz
from matplotlib import pyplot as plt


def get_durations_and_events_from_datetimes(
    data: pd.DataFrame,
    start_column: str,
    end_column: str,
    raise_missing_starts: bool = True,
    raise_ends_before_starts: bool = True,
    duration_column: str | None = None,
    event_column: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    # Check for missing starts
    if data[start_column].isna().any():
        message: str = "Start column `%s` is missing values" % start_column
        if raise_missing_starts:
            raise ValueError(message)
        warnings.warn(message=message)
    # Check for ends before starts
    if data[end_column].lt(other=data[start_column]).any():
        message: str = "Some values of end column `%s` come before corresponding values of start column `%s`" % (
            end_column,
            start_column,
        )
        if raise_ends_before_starts:
            raise ValueError(message)
    # Compute durations and events
    duration_column: str = duration_column or "duration"
    event_column: str = event_column or "event"
    fn: typing.Callable = lambda x: x not in ["start_times", "end_times"]
    filtered_kwargs: dict[str, typing.Any] = tz.keyfilter(predicate=fn, d=kwargs)
    durations: tuple[np.ndarray, np.ndarray] = lifelines.utils.datetimes_to_durations(
        start_times=data[start_column], end_times=data[end_column], **filtered_kwargs
    )
    return tz.pipe(
        zip([duration_column, event_column], durations),
        dict,
        tz.partial(pd.DataFrame, index=data.index),
        lambda x: data.join(other=x, how="left", validate="1:1"),
    )


def plot_low_cardinality_feature(
    data: pd.DataFrame,
    feature_column: str,
    duration_column: str,
    event_column: str,
    ax: plt.Axes | None = None,
    Fitter: typing.Any = lifelines.KaplanMeierFitter,
    init_args: dict | None = None,
    fit_args: dict | None = None,
    plot_args: dict | None = None,
    add_at_risk_counts_args: dict | None = None,
) -> tuple[plt.Axes, list]:
    ax: plt.Axes = ax or plt.subplot(1, 1, 1)
    fitters: list = []
    for value in data[feature_column].drop_duplicates().sort_values():  # type: typing.Any
        filtered_data: pd.DataFrame = data.loc[lambda x: x[feature_column].eq(other=value), :]
        durations: pd.Series = filtered_data[duration_column]
        events: pd.Series = filtered_data[event_column]
        filtered_init_args: dict = tz.keyfilter(predicate=lambda x: x != "label", d=init_args or {})
        fitter: typing.Any = Fitter(label=value, **filtered_init_args)
        filtered_fit_args: dict = tz.keyfilter(
            predicate=lambda x: x not in ["durations", "event_observed"], d=fit_args or {}
        )
        fitters.append(fitter.fit(durations=durations, event_observed=events, **filtered_fit_args))
        filtered_plot_args: dict = tz.keyfilter(predicate=lambda x: x != "ax", d=plot_args or {})
        fitters[-1].plot(ax=ax, **filtered_plot_args)
    filtered_add_at_risk_counts_args: dict = tz.keyfilter(
        predicate=lambda x: x != "ax", d=add_at_risk_counts_args or {}
    )
    lifelines.plotting.add_at_risk_counts(*fitters, ax=ax, **filtered_add_at_risk_counts_args)
    return ax, fitters
