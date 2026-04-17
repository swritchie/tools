import itertools
import typing

import numpy as np
import pandas as pd
import toolz as tz
import tqdm
from matplotlib import pyplot as plt
from sklearn import base as snbe
from sklearn import feature_selection as snfs


class QuasiConstantDropper(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self, drop_na: bool = False, threshold: float = 1e0) -> None:
        self.drop_na, self.threshold = drop_na, threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "QuasiConstantDropper":
        self.most_frequent_percentages: pd.Series = self._get_most_frequent_percentages(data=X)
        self.constant_columns: pd.Index = self._get_constant_columns()
        self.remaining_columns: pd.Index = self._get_remaining_columns(data=X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.constant_columns)

    def get_feature_names_out(self) -> list[str]:
        return self.remaining_columns.tolist()

    def _get_most_frequent_percentage(self, data: pd.Series) -> float:
        value_counts: pd.Series = data.value_counts(dropna=self.drop_na, normalize=True)
        return value_counts.iloc[0] if not value_counts.empty else 1e0

    def _get_most_frequent_percentages(self, data: pd.DataFrame) -> pd.Series:
        return data.apply(func=self._get_most_frequent_percentage).sort_values()

    def _get_percentages_above_threshold(self, data: pd.Series) -> pd.Series:
        return data[data.ge(other=self.threshold)]

    def _get_constant_columns(self) -> pd.Index:
        return self.most_frequent_percentages.pipe(func=self._get_percentages_above_threshold).index

    def _get_remaining_columns(self, data: pd.DataFrame) -> pd.Index:
        return data.columns.difference(other=self.constant_columns)


class DuplicatedDropper(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self, is_memory_low: bool = False) -> None:
        self.is_memory_low = is_memory_low

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DuplicatedDropper":
        self.duplicated_columns: pd.Index = self._get_duplicated_columns(data=X)
        self.remaining_columns: pd.Index = self._get_remaining_columns(data=X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.duplicated_columns)

    def get_feature_names_out(self) -> list[str]:
        return self.remaining_columns.tolist()

    def _get_duplicated_columns(self, data: pd.DataFrame) -> pd.Index:
        if self.is_memory_low:
            column_pairs: list[tuple[str, str]] = list(itertools.combinations(iterable=data.columns, r=2))
            duplicated_columns: list[str] = []
            for first_column, second_column in tqdm.tqdm(iterable=column_pairs):  # type: tuple[str, str]
                if data[first_column].equals(other=data[second_column]):
                    duplicated_columns.append(second_column)
            return pd.Index(data=duplicated_columns)
        else:
            return data.T.duplicated(keep="first").to_frame(name="flag").query(expr="flag").index

    def _get_remaining_columns(self, data: pd.DataFrame) -> pd.Index:
        return data.columns.difference(other=self.duplicated_columns)


def get_scores(rfecv: snfs.RFECV) -> pd.DataFrame:
    def flag_key(key: str) -> bool:
        return key.endswith("score") or key == "n_features"

    assign_args: dict[str, typing.Callable] = {
        "sem_test_score": lambda x: x.filter(regex=r"split\d+_test_score").sem(axis=1),
        "is_best": lambda x: x["mean_test_score"].pipe(func=lambda x: x.eq(other=x.max())),
        "best_mean": lambda x: x["mean_test_score"].where(cond=x["is_best"]).bfill().ffill(),
        "best_sem": lambda x: x["sem_test_score"].where(cond=x["is_best"]).bfill().ffill(),
        "best_lower": lambda x: x["best_mean"].sub(other=x["best_sem"]),
        "is_wi_1_sem": lambda x: x["mean_test_score"].gt(other=x["best_lower"]),
        "is_simplest_wi_1_sem": lambda x: x.index.__eq__(x.query(expr="is_wi_1_sem").index.min()),
    }
    return (
        tz.pipe(rfecv.cv_results_, tz.curried.keyfilter(flag_key), pd.DataFrame)
        .set_index(keys="n_features")
        .assign(**assign_args)
    )


def get_support(rfecv: snfs.RFECV) -> pd.DataFrame:
    def get_split_support(key: str, rfecv: snfs.RFECV = rfecv) -> pd.DataFrame:
        return (
            pd.DataFrame(data=rfecv.cv_results_[key], index=rfecv.cv_results_["n_features"])
            .apply(func=np.array, axis=1)
            .apply(func=lambda x: rfecv.feature_names_in_[x])
        )

    def get_frequencies(data: pd.DataFrame) -> pd.Series:
        return data.apply(func=tz.compose_left(tz.concat, tz.frequencies), axis=1)

    def get_intersections(data: pd.DataFrame) -> pd.Series:
        def flag_value(frequency: int, data: pd.DataFrame = data) -> bool:
            """Flag features selected in all splits (i.e., frequency == n_splits)"""
            return frequency == data.shape[1] - 1  # Get n_splits (support columns minus "frequencies" column)

        def get_intersection(frequencies: dict[str, int]) -> list[str]:
            return list(tz.valfilter(predicate=flag_value, d=frequencies))

        return data["frequencies"].apply(func=get_intersection)

    def get_intersection_percentage(data: pd.DataFrame) -> pd.Series:
        return data["intersection"].apply(func=len).div(other=data.index)

    support_keys: list[str] = tz.pipe(rfecv.cv_results_, tz.curried.keyfilter(lambda x: x.endswith("support")), list)
    assign_args: dict[str, typing.Callable] = {
        "frequencies": get_frequencies,
        "intersection": get_intersections,
        "intersection_percentage": get_intersection_percentage,
    }
    return (
        pd.concat(objs=map(get_split_support, support_keys), axis=1)
        .assign(**assign_args)
        .rename_axis(index="n_features")
    )


def plot_scores(scores: pd.DataFrame, **kwargs) -> plt.Axes:
    # Get scores
    best_score: dict[int, float] = scores.query(expr="is_best")["mean_test_score"].to_dict()
    simplest_wi_1_sem_score: dict[int, float] = scores.query(expr="is_simplest_wi_1_sem")["mean_test_score"].to_dict()
    # Get title
    parts: list[str] = [
        "Best: %d features / %.3f score" % next(iter(best_score.items())),
        "Simplest w/i 1 SEM: %d features / %.3f score" % next(iter(simplest_wi_1_sem_score.items())),
    ]
    title: str = "\n".join(parts)
    # Plot error bars
    keys: list[str] = ["y", "yerr", "marker"]
    filtered_kwargs: dict[str, typing.Any] = tz.keyfilter(predicate=lambda x: x not in keys, d=kwargs)
    shared_args = dict(y="mean_test_score")
    ax: plt.Axes = scores.plot(yerr="sem_test_score", marker=".", **filtered_kwargs, **shared_args)
    # Plot points
    shared_args.update(color="k", ax=ax)
    scores.query(expr="is_best").plot(marker="^", label="best", **shared_args)
    scores.query(expr="is_simplest_wi_1_sem").plot(marker="o", label="simplest", **shared_args)
    # Plot guide lines
    ax.axhline(y=scores["best_lower"].iloc[0], color="k", ls=":")
    list(map(lambda x: ax.axvline(x=next(iter(x)), c="k", ls=":"), [best_score, simplest_wi_1_sem_score]))
    # Set labels
    ax.set(ylabel="Mean +/- SEM", title=title)
    return ax


def plot_support(support: pd.DataFrame, **kwargs) -> plt.Axes:
    column: str = "intersection_percentage"
    label: str = column.replace("_", " ").title()
    args = dict(marker=".", ylabel=label, title=label)
    filtered_kwargs: dict[str, typing.Any] = tz.keyfilter(predicate=lambda x: x not in args, d=kwargs)
    ax: plt.Axes = support[column].plot(**args, **filtered_kwargs)
    return ax
