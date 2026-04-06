import typing

import numpy as np
import pandas as pd
import toolz as tz
from matplotlib import pyplot as plt
from sklearn import base as snbe
from sklearn import feature_selection as snfs


class InteractionEngineer(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(
        self,
        first_feature: str,
        second_feature: str,
        is_classification: bool,
        operators: list | tuple = ("add", "sub", "rsub", "mul", "div", "rdiv"),
        mutual_info_args: dict | None = None,
    ) -> None:
        self.first_feature, self.second_feature = first_feature, second_feature
        self.is_classification = is_classification
        self.operators, self.mutual_info_args = operators, mutual_info_args

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "InteractionEngineer":
        self.initial_features: pd.Index = X.columns
        self.assign_args: dict[str, typing.Callable] = self._get_assign_args()
        X_engineered: pd.DataFrame = self._engineer_features(X=X)
        self.mutual_info: pd.Series = self._get_mutual_information(X_engineered=X_engineered, y=y)
        self.best_engineered_feature: str = self._get_best_engineered_feature()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(**tz.keyfilter(predicate=lambda x: x == self.best_engineered_feature, d=self.assign_args))

    def get_feature_names_out(self) -> list[str]:
        return self.initial_features.tolist() + [self.best_engineered_feature]

    def plot(self, **kwargs) -> plt.Axes:
        filtered_kwargs: dict[str, typing.Any] = tz.keyfilter(predicate=lambda x: x not in ["kind"], d=kwargs)
        return self.mutual_info.iloc[::-1].plot(kind="barh", **filtered_kwargs)

    def _name_engineered_feature(self, operator: str) -> str:
        return "%s_%s_%s" % (self.first_feature, operator, self.second_feature)

    def _engineer_feature(self, operator: str) -> typing.Callable:
        def engineer_feature(X: pd.DataFrame) -> pd.Series:
            eps: float = np.finfo(dtype=float).eps
            first_feature: pd.Series = X[self.first_feature].pipe(
                func=lambda x: x.add(other=eps) if operator == "rdiv" else x
            )
            second_feature: pd.Series = X[self.second_feature].pipe(
                func=lambda x: x.add(other=eps) if operator == "div" else x
            )
            return getattr(first_feature, operator)(other=second_feature)

        return engineer_feature

    def _get_assign_args(self) -> dict[str, typing.Callable]:
        return tz.pipe(
            self.operators,
            tz.curried.map(lambda x: [self._name_engineered_feature(operator=x), self._engineer_feature(operator=x)]),
            dict,
        )

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, [self.first_feature, self.second_feature]].assign(**self.assign_args)

    def _get_mutual_information(self, X_engineered: pd.DataFrame, y: pd.Series) -> pd.Series:
        mutual_info: typing.Callable = (
            snfs.mutual_info_classif if self.is_classification else snfs.mutual_info_regression
        )
        mutual_info_args: dict = self.mutual_info_args.copy() if self.mutual_info_args else {}
        mutual_info_args.setdefault("random_state", 0)
        return pd.Series(
            data=mutual_info(X=X_engineered, y=y, **mutual_info_args), index=X_engineered.columns, name="mutual_info"
        ).sort_values(ascending=False)

    def _get_best_engineered_feature(self) -> str:
        return self.mutual_info.drop(labels=[self.first_feature, self.second_feature]).nlargest(n=1).index[0]
