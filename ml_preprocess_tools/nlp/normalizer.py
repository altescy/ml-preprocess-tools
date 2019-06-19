from __future__ import annotations
import typing as tp

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.porter import PorterStemmer

from .token import Token


Text = tp.List[Token]
Data = tp.List[Text]


class BaseNormalizer(BaseEstimator, TransformerMixin):
    def apply(self, tokens: Text) -> Text:
        raise NotImplementedError

    def fit(self, X, y=None) -> BaseNormalizer: # pylint: disable=unused-argument
        return self

    def transform(self, X: Data) -> None:
        return [self.apply(x) for x in X]


class LowerNormalizer(BaseNormalizer):
    def apply(self, tokens: Text) -> Text:
        for t in tokens:
            t.set_surface(t.surface.lower())
        return tokens


class PorterNormalizer(BaseNormalizer):
    def __init__(self) -> None:
        self._ps = PorterStemmer()

    def apply(self, tokens: Text) -> Text:
        for t in tokens:
            t.set_surface(self._ps.stem(t.surface))
        return tokens


class BaseFormNormalizer(BaseNormalizer):
    def apply(self, tokens: Text) -> Text:
        for t in tokens:
            t.set_surface(t.base_form)
        return tokens


class NumberNormalizer(BaseNormalizer):
    def __init__(self, number_token: str = '0') -> None:
        self.number_token = number_token

    def apply(self, tokens: Text) -> Text:
        for t in tokens:
            if t.is_spacy:
                if t.pos == 'NUM':
                    t.set_surface(self.number_token)
            elif t.is_ja:
                if t.pos == '名詞' and t.tag == '数':
                    t.set_surface(self.number_token)
            else:
                if t.surface.isdigit():
                    t.set_surface(self.number_token)
        return tokens
