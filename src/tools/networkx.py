import typing

import networkx as nx
import numpy as np
import pandas as pd
import toolz as tz
from sklearn import base as snbe
from tools import data_analysis as tsda
from tools import utils as tsus


class CorrelatedDropper(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self, is_memory_low: bool = False, threshold: float = 1e0) -> None:
        self.is_memory_low, self.threshold = is_memory_low, threshold

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CorrelatedDropper":
        self.correlations_between_features: pd.DataFrame = tz.pipe(
            X, tz.partial(tsda.get_correlations, is_memory_low=self.is_memory_low), get_correlated_groups
        )
        self.correlations_with_target: pd.Series = tsda.get_correlations(X=X, y=y, is_memory_low=self.is_memory_low)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        first_left_inclusive_bound: float = self.correlations_between_features.index[0].left
        last_right_exclusive_bound: float = self.correlations_between_features.index[-1].right
        if self.threshold >= last_right_exclusive_bound:
            features_to_drop: list[str] = []
        else:
            self.updated_threshold: float = max(self.threshold, first_left_inclusive_bound)
            features_to_drop: list[str] = get_correlated_features_to_drop(
                correlated_feature_groups=self.correlations_between_features.loc[self.updated_threshold, "groups"],
                correlations_with_target=self.correlations_with_target,
                print_shapes=False,
            )
        return X.drop(columns=features_to_drop)

    def get_feature_names_out():
        pass


def get_correlated_features_to_drop(
    correlated_feature_groups: list[set[str]], correlations_with_target: pd.Series, print_shapes: bool = True
) -> list[str]:
    features_to_drop: list[str] = []
    for correlated_feature_group in correlated_feature_groups:  # type: set[str]
        these_correlations_with_target: pd.Series = correlations_with_target.loc[list(correlated_feature_group)]
        best_feature: str = these_correlations_with_target.abs().idxmax()
        remaining_features: set[str] = correlated_feature_group.difference([best_feature])
        features_to_drop.extend(remaining_features)
        if print_shapes:
            tsus.print_shapes(x=[correlated_feature_groups, remaining_features, features_to_drop], sep=" -> ")
    return features_to_drop


def get_correlated_groups(correlations: pd.Series, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    def set_interval_index(data: pd.DataFrame) -> pd.DataFrame:
        breaks: list[float] = data.index.tolist() + [1]
        return data.set_axis(labels=pd.IntervalIndex.from_breaks(breaks=breaks, closed="left"))

    if thresholds is None:
        thresholds: np.ndarray = np.arange(start=1e-2, stop=1, step=1e-2)
    correlated_groups: dict = {}
    for threshold in thresholds:  # type: float
        filtered_correlations: pd.DataFrame = (
            correlations.abs()
            .pipe(func=lambda x: x[x.ge(other=threshold)])
            .reset_index()
            .set_axis(labels=["source", "target", "r"], axis=1)
        )
        correlated_groups[threshold] = tz.pipe(
            filtered_correlations, tz.partial(nx.from_pandas_edgelist, edge_attr="r"), nx.connected_components, list
        )
    assign_args: dict[str, typing.Callable] = {
        "n_groups": lambda x: x["groups"].apply(func=len),
        "group_sizes": lambda x: x["groups"].apply(func=lambda x: list(map(len, x))),
        "min_group_sizes": lambda x: x["group_sizes"].apply(func=min),
        "max_group_sizes": lambda x: x["group_sizes"].apply(func=max),
        "total_group_sizes": lambda x: x["group_sizes"].apply(func=sum),
        "n_features_dropped": lambda x: x["total_group_sizes"].sub(other=x["n_groups"]),
    }
    return (
        pd.Series(data=correlated_groups)
        .drop_duplicates()
        .pipe(func=lambda x: x[x.apply(func=len).gt(other=1)])
        .to_frame(name="groups")
        .assign(**assign_args)
        .pipe(func=set_interval_index)
    )
