import typing

import catboost as cb
import pandas as pd
import toolz as tz
from catboost import monoforest as cbmf
from matplotlib import pyplot as plt
from tools import typing as tstg

dict_feature_args: list[str] = ["monotone_constraints"]
list_feature_args: list[str] = ["cat_features", "embedding_features", "text_features"]


class CatBoostAnalyzer:
    def __init__(self, model: tstg.CatBoostModel, metrics: list) -> None:
        self.model, self.metrics = model, metrics

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostAnalyzer":
        pool: cb.Pool = get_pool(X=X, y=y, model=self.model)
        self.eval_metrics: pd.DataFrame = get_eval_metrics(model=self.model, pool=pool, metrics=self.metrics)
        self.feature_importances: pd.DataFrame = get_feature_importances(model=self.model, pool=pool)
        if self.model.get_all_params().get("depth") > 1:
            self.feature_interactions: pd.DataFrame = get_feature_interactions(model=self.model, pool=pool)
        return self

    def plot_eval_metrics(self, **kwargs) -> plt.Figure:
        self.eval_metrics.plot(subplots=True, **tz.keyfilter(predicate=lambda x: x not in ["subplots"], d=kwargs))
        return plt.gcf()

    def plot_feature_importances(self, bbox: list | tuple = (1.2, 0, 4e-1, 1), **kwargs) -> plt.Figure:
        filtered_kwargs: dict[str, typing.Any] = tz.keyfilter(predicate=lambda x: x not in ["subplots"], d=kwargs)
        axes: tstg.Axes = self.feature_importances.plot(subplots=True, **filtered_kwargs)
        for ax in axes.flat:  # type: plt.Axes
            ax.axhline(c="k", ls=":")
            ax.set(xticks=[])
            column: str = ax.get_legend().get_texts()[0]._text
            data: pd.Series = self.feature_importances[column].describe().round(decimals=3)
            pd.plotting.table(ax=ax, data=data, bbox=bbox)
        return plt.gcf()

    def plot_top_feature_importances(self, n: int = int(1e1), **kwargs) -> tstg.Axes:
        plot_args = dict(kind="barh", subplots=True, layout=(1, -1), sharex=False, sharey=True, legend=False)
        filtered_kwargs: dict[str, typing.Any] = tz.keyfilter(predicate=lambda x: x not in plot_args, d=kwargs)
        return (
            self.feature_importances.nlargest(n=n, columns="LossFunctionChange")
            .iloc[::-1, :]
            .plot(**plot_args, **filtered_kwargs)
        )

    def plot_feature_interactions(self, bbox: list | tuple = (1.2, 0, 2e-1, 1), **kwargs) -> plt.Axes:
        feature_interactions: pd.Series = self.feature_interactions["interaction_strength"]
        ax: plt.Axes = feature_interactions.plot(**kwargs)
        ax.axhline(c="k", ls=":")
        ax.set(xticks=[])
        data: pd.Series = feature_interactions.describe().round(decimals=3)
        pd.plotting.table(ax=ax, data=data, bbox=bbox)
        return ax

    def plot_top_feature_interactions(self, n: int = int(1e1), **kwargs) -> plt.Axes:
        filtered_kwargs: dict[str, typing.Any] = tz.keyfilter(predicate=lambda x: x not in ["kind"], d=kwargs)
        return self.feature_interactions.squeeze().nlargest(n=n).iloc[::-1].plot(kind="barh", **kwargs)


class CatBoostClassifier(cb.CatBoostClassifier):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "CatBoostClassifier":
        self._update_feature_params(X=X)
        return super().fit(X=X, y=y, **kwargs)

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, features_not_for_select: list[str] | None = None, **kwargs
    ) -> dict[str, typing.Any]:
        self._update_feature_params(X=X)
        if "features_for_select" not in kwargs:
            kwargs.update(features_for_select=X.columns.difference(other=features_not_for_select or []).tolist())
        return super().select_features(X=X, y=y, **kwargs)

    def _update_feature_params(self, X: pd.DataFrame) -> None:
        params: dict[str, typing.Any] = self.get_params()
        params = update_feature_params(params=params, X=X)
        self.set_params(**params)


class CatBoostRegressor(cb.CatBoostRegressor):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "CatBoostRegressor":
        self._update_feature_params(X=X)
        return super().fit(X=X, y=y, **kwargs)

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, features_not_for_select: list[str] | None = None, **kwargs
    ) -> dict[str, typing.Any]:
        self._update_feature_params(X=X)
        if "features_for_select" not in kwargs:
            kwargs.update(features_for_select=X.columns.difference(other=features_not_for_select or []).tolist())
        return super().select_features(X=X, y=y, **kwargs)

    def _update_feature_params(self, X: pd.DataFrame) -> None:
        params: dict[str, typing.Any] = self.get_params()
        params = update_feature_params(params=params, X=X)
        self.set_params(**params)


class MonoforestAnalyzer:
    def __init__(self, model: tstg.CatBoostModel) -> None:
        self.model = model

    def fit(self) -> "MonoforestAnalyzer":
        self.polynom_data: pd.DataFrame = self._get_polynomial_data()
        self.split_data: pd.DataFrame = self._get_split_data()
        self.joined_data: pd.DataFrame = self._join_data()
        self.aggregated_data: pd.DataFrame = self._aggregate_weights_by_feature_and_border()
        self.interaction_data: pd.DataFrame = self._filter_to_interactions()
        return self

    def get_formula_for_first_n_splits(self, n: int) -> str:
        def get_splits(data: pd.DataFrame) -> list[pd.DataFrame]:
            return tz.pipe(
                data.index.get_level_values(level=0).drop_duplicates(), tz.curried.map(lambda x: data.loc[x, :]), list
            )

        def format_split(data: pd.DataFrame) -> str:
            return tz.pipe(
                data.iterrows(),
                tz.curried.map(lambda x: "[%s %s %s]" % (x[0], x[1]["split_type"], x[1]["border"])),
                list,
                "".join,
                lambda x: "(%s) * %s" % (data["value"].iloc[0], x),
            )

        return tz.pipe(
            self.filter_to_first_n_splits(n=n),
            get_splits,
            tz.curried.map(format_split),
            list,
            lambda x: ["<Intercept>"] + x,
            "\n+ ".join,
        )

    def filter_to_first_n_splits(self, n: int) -> pd.DataFrame:
        splits: pd.Index = self.joined_data.index.get_level_values(level=0)
        first_n_splits: pd.Index = splits.to_series().drop_duplicates().head(n=n).index
        return self.joined_data.loc[splits.isin(values=first_n_splits), :]

    def filter_to_feature(self, feature: str) -> pd.DataFrame:
        return (
            self.joined_data.loc[lambda x: x.index.get_level_values(level=1).__eq__(feature), :]
            .swaplevel()
            .sort_values(by="border")
        )

    def _get_polynomial_data(self) -> pd.DataFrame:
        return (
            tz.pipe(self.model, cbmf.to_polynom, tz.curried.map(lambda x: x.__dict__), pd.DataFrame)
            .rename_axis(index="split_idx")
            .assign(**{"value": lambda x: x["value"].apply(func=lambda y: y[0])})
            .sort_values(by="weight", ascending=False)
        )

    def _get_split_data(self) -> pd.DataFrame:
        feature_names: dict[int, str] = dict(enumerate(iterable=self.model.feature_names_))
        return (
            self.polynom_data["splits"]
            .explode()
            .dropna()
            .apply(func=lambda x: x.__dict__)
            .apply(func=pd.Series)
            .assign(**{"feature": lambda x: x["feature_idx"].map(arg=feature_names)})
            .set_index(keys="feature", append=True)
        )

    def _join_data(self) -> pd.DataFrame:
        return self.split_data.drop(columns="feature_idx").join(
            other=self.polynom_data.drop(columns="splits"), how="left", validate="m:1"
        )

    def _aggregate_weights_by_feature_and_border(self) -> pd.DataFrame:
        return self.joined_data.reset_index().groupby(by=["feature", "border"])[["weight"]].sum().sort_index()

    def _filter_to_interactions(self) -> pd.DataFrame:
        def flag_interactions(data: pd.DataFrame) -> pd.Series:
            return data.groupby(level="split_idx").transform(func="size").gt(other=1)

        return self.joined_data.loc[flag_interactions, :]


class ValidationChangeCallback:
    def __init__(self, metric: str, threshold: float = 1e-4) -> None:
        self.metric, self.threshold = metric, threshold

    def after_iteration(self, info: typing.Any) -> bool:
        try:
            metric_before, metric_after = map(lambda x: info.metrics["validation"][self.metric][x], [-2, -1])  # type: tuple[float, float]
            absolute_change: float = abs(metric_after - metric_before)
            should_continue: bool = absolute_change > self.threshold
        except Exception as exception:
            should_continue: bool = True
        return should_continue


class ValidationDifferenceCallback:
    def __init__(self, metric: str, threshold: float = 1e-2) -> None:
        self.metric, self.threshold = metric, threshold

    def after_iteration(self, info: typing.Any) -> bool:
        learn_metric: float = info.metrics["learn"][self.metric][-1]
        validation_metric: float = info.metrics["validation"][self.metric][-1]
        absolute_difference: float = abs(validation_metric - learn_metric)
        should_continue: bool = absolute_difference < self.threshold
        return should_continue


def filter_select_features_response(
    parse_response: tuple[pd.Series, pd.Index, pd.Index],
) -> tuple[float, pd.Index, pd.Index]:
    loss_graph, eliminated_features, selected_features = parse_response
    min_loss: float = loss_graph.min()
    min_loss_feature: str = loss_graph.idxmin()
    optimal_eliminated_features: pd.Index = loss_graph.loc[:min_loss_feature].index[:-1]
    optimal_selected_features: pd.Index = eliminated_features.difference(other=optimal_eliminated_features).union(
        other=selected_features
    )
    return min_loss, optimal_eliminated_features, optimal_selected_features


def get_eval_metrics(model: tstg.CatBoostModel, pool: cb.Pool, metrics: list) -> pd.DataFrame:
    return pd.DataFrame(data=model.eval_metrics(data=pool, metrics=metrics))


def get_evals_result(model: tstg.CatBoostModel) -> pd.DataFrame:
    return pd.DataFrame(data=model.evals_result_).stack().apply(func=pd.Series).T


def get_feature_importances(model: tstg.CatBoostModel, pool: cb.Pool) -> pd.DataFrame:
    def get_feature_importances(importance_type: str) -> pd.Series:
        return pd.Series(
            data=model.get_feature_importance(data=pool, type=importance_type),
            index=model.feature_names_,
            name=importance_type,
        )

    return tz.pipe(
        ["PredictionValuesChange", "LossFunctionChange"],
        tz.curried.map(get_feature_importances),
        tz.partial(pd.concat, axis=1),
    ).sort_values(by="LossFunctionChange", ascending=False)


def get_feature_interactions(model: tstg.CatBoostModel, pool: cb.Pool) -> pd.DataFrame:
    columns: list[str] = ["first_feature", "second_feature", "interaction_strength"]
    return (
        pd.DataFrame(data=model.get_feature_importance(data=pool, type="Interaction"), columns=columns)
        .astype(dtype=dict(map(lambda x: [x, int], columns[:2])))
        .apply(func=lambda x: x.map(arg=dict(enumerate(iterable=model.feature_names_))) if x.name in columns[:2] else x)
        .set_index(keys=columns[:2])
    )


def get_pool(X: pd.DataFrame, y: pd.Series, model: tstg.CatBoostModel) -> cb.Pool:
    return tz.pipe(
        model.get_params(),
        tz.curried.keyfilter(lambda x: x in list_feature_args),
        lambda x: cb.Pool(data=X, label=y, **x),
    )


def parse_select_features_response(
    select_features_response: dict[str, typing.Any],
) -> tuple[pd.Series, pd.Index, pd.Index]:
    loss_graph: pd.Series = pd.Series(
        data=select_features_response["loss_graph"]["loss_values"][1:],
        index=select_features_response["eliminated_features_names"],
        name="loss",
    )
    eliminated_features: pd.Index = loss_graph.index
    selected_features: pd.Index = pd.Index(data=select_features_response["selected_features_names"])
    return loss_graph, eliminated_features, selected_features


def plot_evals_result(evals_result: pd.DataFrame, **kwargs) -> plt.Figure:
    metrics: pd.Index = evals_result.columns.get_level_values(level=0).drop_duplicates()
    fig, axes = plt.subplots(nrows=metrics.shape[0], sharex=True, squeeze=False)  # tstg.FigAxes
    for metric, ax in zip(metrics, axes.flat):  # type: tuple[str, tstg.Axes]
        plot_args = dict(title=metric, ax=ax)
        filtered_kwargs: dict[str, typing.Any] = tz.keyfilter(predicate=lambda x: x not in plot_args, d=kwargs)
        evals_result[metric].plot(**plot_args, **filtered_kwargs)
    fig.tight_layout()
    return fig


def plot_loss_graph(select_features_response: dict[str, typing.Any], **kwargs) -> plt.Axes:
    # Parse
    parse_response = parse_select_features_response(select_features_response=select_features_response)  # type: tuple[pd.Series, pd.Index, pd.Index]
    loss_graph, eliminated_features, selected_features = parse_response
    # Filter
    filter_response = filter_select_features_response(parse_response=parse_response)  # type: tuple[float, pd.Index, pd.Index]
    min_loss, optimal_eliminated_features, optimal_selected_features = filter_response
    # Count
    all_features: pd.Index = optimal_selected_features.union(optimal_eliminated_features)
    fn: typing.Callable = lambda x: x.shape[0]
    feature_counts: list[int] = list(map(fn, [optimal_eliminated_features, optimal_selected_features, all_features]))
    all_feature_count: int = feature_counts[-1]
    optimal_eliminated_feature_count: int = feature_counts[0]
    # Plot
    title: str = "Min loss: %.3f\nEliminated: %d | Selected: %d | All: %d" % (min_loss, *feature_counts)
    filtered_kwargs: dict[str, typing.Any] = tz.keyfilter(predicate=lambda x: x not in ["xticks", "title"], d=kwargs)
    ax: plt.Axes = loss_graph.plot(xticks=range(all_feature_count), title=title, **filtered_kwargs)
    ax.axvline(x=optimal_eliminated_feature_count, c="k", ls=":")
    ax.set(xticklabels=range(all_feature_count))
    return ax


def update_feature_params(params: dict[str, typing.Any], X: pd.DataFrame) -> dict[str, typing.Any]:
    for arg in dict_feature_args:
        dict_param: dict = tz.keyfilter(predicate=lambda x: x in X, d=params.get(arg, {}))
        params.update(**{arg: dict_param})
    for arg in list_feature_args:
        list_param: list = X.columns.intersection(other=params.get(arg, [])).tolist()
        params.update(**{arg: list_param})
    return params
