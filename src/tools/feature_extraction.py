import pandas as pd
from sklearn.feature_extraction import text as snfett


class CountVectorizer(snfett.CountVectorizer):
    def fit(self, X: pd.Series, y: pd.Series | None = None) -> "CountVectorizer":
        super().fit(raw_documents=X)
        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            data=super().transform(raw_documents=X).toarray(), columns=self.get_feature_names_out(), index=X.index
        )

    def fit_transform(self, X: pd.Series, y: pd.Series | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            data=super().fit_transform(raw_documents=X).toarray(), columns=self.get_feature_names_out(), index=X.index
        )

    def set_output(self, *, transform: str | None = None) -> None:
        pass


class TfidfVectorizer(snfett.TfidfVectorizer):
    def fit(self, X: pd.Series, y: pd.Series | None = None) -> "TfidfVectorizer":
        super().fit(raw_documents=X)
        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            data=super().transform(raw_documents=X).toarray(), columns=self.get_feature_names_out(), index=X.index
        )

    def fit_transform(self, X: pd.Series, y: pd.Series | None = None) -> pd.DataFrame:
        self.fit(X=X, y=y)
        return self.transform(X=X)

    def set_output(self, *, transform: str | None = None) -> None:
        pass
