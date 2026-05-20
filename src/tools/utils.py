import contextlib
import datetime
import logging
import pathlib
import shutil
import typing
import warnings

import numpy as np
import pandas as pd
import toolz as tz
from matplotlib import pyplot as plt


class DocstringParser:
    def __init__(self, x: typing.Any) -> None:
        self.docstring: str = x.__doc__ or ""

    def fit(self) -> "DocstringParser":
        self.docstring_lines: pd.Series = self._get_lines()
        self.docstring_sections: pd.DataFrame = self._get_sections()
        return self

    def print_section(self, section: str = "Overview") -> None:
        print(self._get_section(section=section), end="\n" * 2)

    def print_sections(self, sections: list[str] | tuple[str] = ("Overview",)) -> None:
        for section in sections:
            self.print_section(section=section)

    def _prefix(self) -> str:
        return "Overview\n--------\n%s" % self.docstring

    def _get_lines(self) -> pd.Series:
        return pd.Series(data=self._prefix().splitlines(), name="lines").str.strip()

    def _get_sections(self) -> pd.DataFrame:
        assign_args: dict[str, typing.Callable] = {
            "has_underscores": lambda x: x["lines"].str.contains(pat=r"^[-=]{3,}$"),
            "is_header": lambda x: x["has_underscores"].shift(periods=-1).ffill(),
            "headers": lambda x: x["lines"].where(cond=x["is_header"]).ffill().fillna(value="Overview"),
        }
        agg_args: dict[str, typing.Callable] = {
            "starts": lambda x: x.index.min(),
            "stops": lambda x: x.index.max().__add__(1),
        }
        return (
            self.docstring_lines.to_frame()
            .assign(**assign_args)
            .groupby(by="headers")["lines"]
            .agg(**agg_args)
            .sort_values(by="starts")
        )

    def _get_section(self, section: str) -> str:
        if section not in self.docstring_sections.index:
            raise ValueError("Unknown section: %s" % section)
        starts_and_stops: pd.Series = self.docstring_sections.loc[section, :]
        return self.docstring_lines.iloc[slice(*starts_and_stops)].str.cat(sep="\n")


def describe_structure(x: typing.Any, indent: int = 0, max_indent: int = 2) -> None:
    prefix: str = "  " * indent
    if indent > max_indent:
        print("%s(Depth limit reached)" % prefix)
        return
    if isinstance(x, dict):
        print("%s%s with %d keys" % (prefix, type(x), len(x)))
        for key, value in x.items():  # type: tuple[typing.Any, typing.Any]
            print("%s- key: %s" % (prefix, key))
            describe_structure(x=value, indent=indent + 1, max_indent=max_indent)
    elif isinstance(x, (list, set, tuple)):
        print("%s%s with %d elements" % (prefix, type(x), len(x)))
        for i, element in enumerate(iterable=x):  # type: tuple[int, typing.Any]
            print("%s- element: %d" % (prefix, i))
            describe_structure(x=element, indent=indent + 1, max_indent=max_indent)
    else:
        print("%s%s" % (prefix, get_type_and_shape(x=x)))


def configure_logging(path: str | None, params: dict[str, typing.Any]) -> pathlib.Path | None:
    if path is not None:
        outputs_directory = pathlib.Path(path)
        shutil.rmtree(path=outputs_directory, ignore_errors=True)
        outputs_directory.mkdir()
        logging.basicConfig(filename=outputs_directory / str(datetime.date.today()), **params["basic_config_args"])
        return outputs_directory
    else:
        logging.basicConfig(**params["basic_config_args"])


def display_data(
    data: pd.DataFrame | pd.Series, is_in_notebook: bool, name: str | None = None, info_args: dict | None = None
) -> None:
    if name is not None:
        print("=" * int(8e1), name, "-" * int(8e1), sep="\n")
    data.info(**(info_args or {}))
    if is_in_notebook:
        print("-" * int(8e1))
        display(data)


def filter_dir(x: typing.Any, include_underscores: bool = False, include_modules: bool = False) -> pd.DataFrame:
    assign_args: dict[str, typing.Callable] = {
        "has_underscore": lambda y: y["object"].str.startswith(pat="_"),
        "type": lambda y: y["object"].apply(func=lambda z: type(getattr(x, z)).__name__),
        "is_module": lambda y: y["type"].eq(other="module"),
    }
    return (
        pd.Series(data=dir(x))
        .to_frame(name="object")
        .assign(**assign_args)
        .query(expr="has_underscore.eq(other=%s)" % include_underscores)
        .query(expr="is_module.eq(other=%s)" % include_modules)
        .set_index(keys="object")
    )


def get_shape(x: typing.Any) -> typing.Any:
    if hasattr(x, "shape"):
        return x.shape
    elif hasattr(x, "size"):
        return x.size
    elif hasattr(x, "__len__"):
        return len(x)
    else:
        return np.nan


def get_type_and_shape(x: typing.Any) -> tuple[type, typing.Any]:
    return type(x), get_shape(x=x)


@contextlib.contextmanager
def ignore_exceptions() -> typing.Iterator[None]:
    try:
        yield
    except Exception as exception:
        print(type(exception), exception)


@contextlib.contextmanager
def ignore_warnings() -> typing.Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore")
        yield


def print_sequence(x: typing.Any, header: str = "Sequence") -> None:
    length: int = len(x)
    length_of_length: int = len(str(length))
    sequence: str = tz.pipe(
        enumerate(iterable=x), tz.curried.map(lambda y: f"{y[0]:0{length_of_length}d}. {y[1]}"), "\n".join
    )
    print("%s (%d):\n%s" % (header, length, sequence))


def print_shapes(x: typing.Any, include_types: bool = False, **kwargs) -> None:
    print(*map(get_type_and_shape if include_types else get_shape, x), **kwargs)


def print_type_and_return(x: typing.Any) -> typing.Any:
    print(type(x))
    return x


def save_show_and_close(outputs_directory: pathlib.Path | None, filename: str | None, is_in_notebook: bool) -> None:
    if outputs_directory is not None and filename is not None:
        plt.savefig(fname=outputs_directory / filename, bbox_inches="tight")
    if is_in_notebook:
        plt.show()
    plt.close()


def time_callable(fn: typing.Callable) -> typing.Callable:
    def wrap_callable(*args, **kwargs):
        now = datetime.datetime.now()
        result = fn(*args, **kwargs)
        print("%s - %s" % (fn.__qualname__, str(datetime.datetime.now() - now)))
        return result

    return wrap_callable


def write_readme(outputs_directory: pathlib.Path) -> None:
    readme_path: pathlib.Path = outputs_directory.joinpath("README.md")
    paths: list[pathlib.Path] = sorted(outputs_directory.glob(pattern="*.png"))
    names: list[str] = list(map(lambda x: x.name, paths))
    stems: list[str] = list(map(lambda x: x.stem, paths))
    data: str = tz.pipe(zip(stems, names), tz.curried.map(lambda x: "# %s\n![](%s)" % x), "\n".join)
    readme_path.write_text(data=data)
