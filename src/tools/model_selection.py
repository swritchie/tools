import typing

import numpy as np
import pandas as pd
import toolz as tz
from matplotlib import pyplot as plt
from sklearn import model_selection as snmos


def aggregate_scores(scores: pd.DataFrame, is_simplest_smaller: bool, is_test: bool) -> pd.DataFrame:
    agg: str = "min" if is_simplest_smaller else "max"
    assign_args: dict[str, typing.Callable] = {
        "is_best": lambda x: x["mean"].pipe(func=lambda x: x.eq(other=x.max())),
        "best_mean": lambda x: x["mean"].where(cond=x["is_best"]).bfill().ffill(),
        "best_sem": lambda x: x["sem"].where(cond=x["is_best"]).bfill().ffill(),
        "best_lower": lambda x: x["best_mean"].sub(other=x["best_sem"]),
        "is_wi_1_sem": lambda x: x["mean"].ge(other=x["best_lower"]),
        "is_simplest_wi_1_sem": lambda x: x.index.__eq__(getattr(x.query(expr="is_wi_1_sem").index, agg)()),
    }
    return scores.agg(func=["mean", "sem"], axis=1).pipe(func=lambda x: x.assign(**assign_args) if is_test else x)


def get_scores(display: snmos.LearningCurveDisplay | snmos.ValidationCurveDisplay, get_test: bool) -> pd.DataFrame:
    data: np.ndarray = getattr(display, "%s_scores" % ("test" if get_test else "train"))
    if isinstance(display, snmos.LearningCurveDisplay):
        index = pd.Index(data=display.train_sizes, name="train_sizes")
    else:
        index = pd.Index(data=display.param_range, name=display.param_name)
    scores = pd.DataFrame(data=data, index=index)
    return scores


def plot_test_scores(test_scores: pd.DataFrame, **kwargs) -> plt.Axes:
    # Get scores
    best_score: dict[typing.Any, float] = test_scores.query(expr="is_best")["mean"].to_dict()
    simplest_wi_1_sem_score: dict[typing.Any, float] = test_scores.query(expr="is_simplest_wi_1_sem")["mean"].to_dict()
    # Get title
    parts: list[str] = [
        "Best: %s / %.3f score" % next(iter(best_score.items())),
        "Simplest w/i 1 SEM: %s / %.3f score" % next(iter(simplest_wi_1_sem_score.items())),
    ]
    title: str = "\n".join(parts)
    # Plot error bars
    keys: list[str] = ["y", "yerr", "marker"]
    filtered_kwargs: dict[str, typing.Any] = tz.keyfilter(predicate=lambda x: x not in keys, d=kwargs)
    shared_args = dict(y="mean")
    ax: plt.Axes = test_scores.plot(yerr="sem", marker=".", **filtered_kwargs, **shared_args)
    # Plot points
    shared_args.update(color="k", ax=ax)
    test_scores.query(expr="is_best").plot(marker="^", label="best", **shared_args)
    test_scores.query(expr="is_simplest_wi_1_sem").plot(marker="o", label="simplest", **shared_args)
    # Plot guide lines
    ax.axhline(y=test_scores["best_lower"].iloc[0], color="k", ls=":")
    list(map(lambda x: ax.axvline(x=next(iter(x)), c="k", ls=":"), [best_score, simplest_wi_1_sem_score]))
    # Set labels
    ax.set(ylabel="Mean +/- SEM", title=title)
    return ax
