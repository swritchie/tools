import itertools
import pathlib
import typing

import numpy as np
import pandas as pd
import toolz as tz
import tqdm
from matplotlib import pyplot as plt
from tools import utils as tsus


def describe_dataset(data: pd.DataFrame) -> pd.DataFrame:
    def get_most_frequent_values(data: pd.Series) -> dict:
        return data.value_counts(normalize=True).nlargest().round(decimals=3).to_dict()

    def describe(data: pd.DataFrame) -> pd.DataFrame:
        assign_args: dict[str, typing.Callable] = {
            "iqr": lambda x: x["75%"].sub(other=x["25%"]),
            "has_low_outlier": lambda x: x["iqr"].mul(other=1.5).rsub(other=x["25%"]).gt(other=x["min"]),
            "has_high_outlier": lambda x: x["iqr"].mul(other=1.5).add(other=x["75%"]).lt(other=x["max"]),
            "has_outlier": lambda x: x.filter(like="outlier").any(axis=1),
        }
        return data.describe().T.assign(**assign_args).drop(columns="count")

    objs: list[pd.DataFrame | pd.Series] = [
        data.dtypes.astype(dtype=str).rename(index="dtypes"),
        data.nunique().rename(index="nunique"),
        data.apply(func=get_most_frequent_values).rename(index="frequent_values"),
        data.isna().mean().rename(index="pct_missing"),
        data.isin(values=[-np.inf, np.inf]).mean().rename(index="pct_inf"),
        data.select_dtypes(include="number").lt(other=0).mean().rename(index="pct_negative"),
        data.eq(other=0).mean().rename(index="pct_zero"),
        data.pipe(func=describe),
    ]
    return pd.concat(objs=objs, axis=1).sort_values(by=["dtypes", "nunique"]).round(decimals=3)


def describe_datasets(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    filter_to_common_columns: bool = True,
    filter_to_differences: bool = True,
) -> pd.DataFrame:
    common_columns: pd.Index = reference_data.columns.intersection(other=current_data.columns)
    objs: list[pd.Series] = tz.pipe(
        [reference_data, current_data],
        tz.curried.map(lambda x: x.loc[:, common_columns] if filter_to_common_columns else x),
        tz.curried.map(lambda x: describe_dataset(data=x).select_dtypes(include="number").stack()),
        list,
    )
    assign_args: dict[str, typing.Callable] = {"difference": lambda x: x.diff(axis=1).iloc[:, -1]}
    description: pd.DataFrame = pd.concat(objs=objs, axis=1, keys=["reference", "current"]).assign(**assign_args)
    if filter_to_differences:
        eps: float = np.finfo(dtype=float).eps
        return description.loc[lambda x: x["difference"].abs().gt(other=eps), :]
    return description


def get_correlations(
    X: pd.DataFrame, y: pd.Series | None = None, is_memory_low: bool = False, method: str = "spearman"
) -> pd.Series:
    def process(data: pd.Series, method: str = method) -> pd.Series:
        return data.rename(index=method).dropna().sort_values()

    X_numeric: pd.DataFrame = X.select_dtypes(include="number")
    if y is None:
        if is_memory_low:
            column_pairs: list[tuple[str, str]] = list(itertools.combinations(iterable=X_numeric.columns, r=2))
            correlations: dict[tuple[str, str], float] = {}
            for column_pair in tqdm.tqdm(iterable=column_pairs):  # type: tuple[str, str]
                correlations[column_pair] = X_numeric[list(column_pair)].corr(method=method).iloc[0, 1]
            return pd.Series(data=correlations).pipe(func=process)
        else:
            fn: typing.Callable = tz.compose_left(tz.partial(np.ones_like, dtype=bool), tz.partial(np.triu, k=1))
            return X_numeric.corr(method=method).where(cond=fn).stack().pipe(func=process)
    return X_numeric.corrwith(other=y, method=method).pipe(func=process)


def get_differences(data: pd.DataFrame, first_column: str, second_column: str) -> pd.DataFrame:
    columns: list[str] = [first_column, second_column]
    assign_args: dict[str, typing.Callable] = {
        "abs_diff": lambda x: x.loc[:, columns].diff(axis=1).iloc[:, -1].abs(),
        "mean": lambda x: x.loc[:, columns].mean(axis=1),
        "pct_diff": lambda x: x["abs_diff"].div(other=x["mean"].where(cond=x["mean"].ne(other=0))),
    }
    return data.assign(**assign_args)


def plot_dataset_description(
    dataset_description: pd.DataFrame,
    outputs_directory: pathlib.Path | None,
    is_in_notebook: bool,
    other_columns: list | None = None,
) -> None:
    # Plot dtypes
    column: str = "dtypes"
    dataset_description[column].pipe(func=plot_value_counts)
    shared_args = dict(outputs_directory=outputs_directory, is_in_notebook=is_in_notebook)
    tsus.save_show_and_close(filename=column, **shared_args)
    # Plot other columns
    columns: list = other_columns or ["nunique", "pct_inf", "pct_missing", "pct_negative", "pct_zero", "min", "max"]
    for column in columns:  # type: str
        dataset_description[column].pipe(func=pd.to_numeric, errors="coerce").dropna().pipe(func=plot_histogram)
        tsus.save_show_and_close(filename=column, **shared_args)


def plot_histogram(data: pd.Series, bbox: list | tuple = (1.2, 0, 2e-1, 1), **kwargs) -> plt.Axes:
    filtered_kwargs: dict[str, typing.Any] = tz.keyfilter(predicate=lambda x: x not in ["kind"], d=kwargs)
    ax: plt.Axes = data.plot(kind="hist", **filtered_kwargs)
    pd.plotting.table(ax=ax, data=data.describe().round(decimals=3), bbox=bbox)
    return ax


def plot_largest_barh(
    data: pd.Series, n: int = int(1e1), return_signed: bool = True, bbox: list | tuple = (1.2, 0, 2e-1, 1), **kwargs
) -> plt.Axes:
    largest_unsigned_data: pd.Series = data.abs().nlargest(n=n).iloc[::-1]
    largest_signed_data: pd.Series = data.loc[largest_unsigned_data.index].sort_values()
    plot_data: pd.Series = largest_signed_data if return_signed else largest_unsigned_data
    filtered_kwargs: dict[str, typing.Any] = tz.keyfilter(predicate=lambda x: x not in ["kind"], d=kwargs)
    ax: plt.Axes = plot_data.plot(kind="barh", **filtered_kwargs)
    formatted_data: pd.Series = plot_data.reset_index(drop=True).round(decimals=3).iloc[::-1]
    pd.plotting.table(ax=ax, data=formatted_data, bbox=bbox)
    return ax


def plot_value_counts(data: pd.Series, n: int = int(1e1), bbox: list | tuple | None = None, **kwargs) -> plt.Axes:
    plot_data: pd.DataFrame = (
        data.value_counts(dropna=False)
        .to_frame(name="cnt")
        .assign(**{"pct": lambda x: x["cnt"].pipe(func=lambda y: y.div(other=y.sum()))})
        .nlargest(n=n, columns="cnt")
    )
    filtered_kwargs: dict[str, typing.Any] = tz.keyfilter(predicate=lambda x: x not in ["kind"], d=kwargs)
    ax: plt.Axes = plot_data["cnt"].iloc[::-1].plot(kind="barh", **filtered_kwargs)
    if bbox is None:
        bbox = (1.2, 0, 4e-1, 2e-1 * plot_data.shape[0])
    pd.plotting.table(ax=ax, data=plot_data.round(decimals=3), bbox=bbox)
    return ax
