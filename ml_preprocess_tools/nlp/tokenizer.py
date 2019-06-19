from __future__ import annotations
import typing as tp

from collections import namedtuple
from itertools import chain
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_even_slices

from .token import Token, TokenType


Text = tp.List[Token]
Data = tp.List[Text]

SpacyToken = namedtuple(
    'SpacyToken',
    ('text', 'lemma_', 'pos_', 'tag_')
)

class BaseTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs

    def tokenize(self, x: str) -> Text:
        raise NotImplementedError

    def fit(self, X, y=None): # pylint: disable=unused-argument
        return self

    def transform(self, X: tp.List[str]) -> Data:
        n_jobs = self.n_jobs if self.n_jobs > 0 else joblib.cpu_count() + 1 + self.n_jobs
        assert n_jobs > 0
        @joblib.delayed
        @joblib.wrap_non_picklable_objects
        def task(X):
            return [self.tokenize(x) for x in list(X)]
        return list(chain.from_iterable(
            joblib.Parallel(n_jobs=n_jobs)(
                task(X[s.start:s.stop])
                for s in gen_even_slices(len(X), n_jobs)
            )
        ))


class SplitTokenizer(BaseTokenizer):
    def __init__(self, n_jobs: int = -1) -> None:
        super().__init__(n_jobs=n_jobs)

    def tokenize(self, x: str) -> tp.List[Token]:
        return [Token(w.strip(), token_type=TokenType.PLAIN) for w in x.split()]


class SpacyTokenizer(BaseTokenizer):
    def __init__(self, lang, n_jobs=-1):
        super().__init__(n_jobs)
        disables = ["textcat", "ner", "parser"]
        import spacy
        if lang in ["en", "de", "es", "pt", "fr", "it", "nl"]:
            nlp = spacy.load(lang, disable=disables)
        else:
            nlp = spacy.load("xx", disable=disables)
        self._nlp = nlp

    def tokenize(self, x: str) -> tp.List[Token]:
        tokens = self._nlp(x)
        return [
            Token(SpacyToken(t.text, t.lemma_, t.pos_, t.tag_))
            for t in tokens
        ]


class MeCabTokenizer(BaseTokenizer):
    JanomeToken = namedtuple(
        "JanomeToken",
        ("surface", "part_of_speech", "infl_type", "infl_form", "base_form", "reading", "phonetic")
    )

    def __init__(self, n_jobs=-1) -> None:
        super().__init__(n_jobs=n_jobs)
        import MeCab
        self.tagger = MeCab.Tagger("-Ochasen")

    def tokenize(self, x: str) -> Text:
        self.tagger.parse('')
        node = self.tagger.parseToNode(x)
        tokens = []
        while node:
            if node.surface:
                surface = node.surface
                features = node.feature.split(',')
                if len(features) < 9:
                    pad_size = 9 - len(features)
                    features += ["*"] * pad_size

                _token = MeCabTokenizer.JanomeToken(
                    surface, ",".join(features[:4]),
                    features[4], features[5],
                    features[6], features[7],
                    features[8]
                )
                token = Token(_token, token_type=TokenType.JA)
                tokens.append(token)
            node = node.next
        return tokens


class JanomeTokenizer(BaseTokenizer):
    def __init__(self, n_jobs=-1) -> None:
        super().__init__(n_jobs=n_jobs)
        from janome.tokenizer import Tokenizer
        self.tokenizer = Tokenizer()

    def tokenize(self, x: str) -> Text:
        tokens = self.tokenizer.tokenize(x)
        tokens = [Token(t, token_type=TokenType.JA) for t in tokens]
        return tokens
