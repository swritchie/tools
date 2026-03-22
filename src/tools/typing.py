import catboost as cb
import typing
from matplotlib import pyplot as plt
from numpy import typing as npt


Axes: typing.TypeAlias = plt.Axes | npt.NDArray[plt.Axes]
CatBoostModel: typing.TypeAlias = cb.CatBoostClassifier | cb.CatBoostRegressor
FigAxes: typing.TypeAlias = tuple[plt.Figure, Axes]
