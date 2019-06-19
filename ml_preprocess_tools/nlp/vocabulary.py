from __future__ import annotations
import typing as tp

import numbers
from sklearn.base import BaseEstimator, TransformerMixin

from .token import Token


Number = tp.Union[int, float]
Text = tp.List[Token]
Data = tp.List[Text]


class Vocabulary(BaseEstimator, TransformerMixin):
    def __init__(self, max_df: Number = 1.0, min_df: Number = 1, vocab_size: int = -1, ignore_blank: bool = True, **kwargs) -> None:
        self.max_df = max_df
        self.min_df = min_df
        self.vocab_size = vocab_size
        self.ignore_blank = ignore_blank
        self.pad = kwargs.get("pad", "<pad>")
        self.bos = kwargs.get("bos", "<bos>")
        self.eos = kwargs.get("eos", "<eos>")
        self.unk = kwargs.get("unk", "<unk>")
        self.id2word = {}  # type: tp.Dict[int, str]
        self.word2id = {}  # type: tp.Dict[str, int]
        for t in [self.pad, self.bos, self.eos, self.unk]:
            if t:
                self.word2id[t] = len(self.word2id)
                self.id2word[self.word2id[t]] = t

    def fit(self, X: Data, y: tp.Any = None) -> Vocabulary: # pylint: disable=unused-argument
        df = {}  # type: tp.Dict[str, int]
        for text in X:
            word_set = set()
            for t in text:
                w = t.surface
                if w not in self.word2id:
                    self.word2id[w] = len(self.id2word)
                    self.id2word[self.word2id[w]] = w
                word_set.add(w)
            for w in word_set:
                df[w] = df.get(w, 0) + 1
        min_limit = self.min_df if isinstance(self.min_df, numbers.Integral) \
            else self.min_df * len(X)
        max_limit = self.max_df if isinstance(self.max_df, numbers.Integral) \
            else self.max_df * len(X)
        for w, c in df.items():
            if not min_limit <= c <= max_limit:
                i = self.word2id[w]
                del self.id2word[i], self.word2id[w]
        return self

    def transform(self, X: Data) -> tp.List[tp.List[int]]:
        ret = []
        for text in X:
            if self.unk:
                ret.append([self.word2id.get(t.surface, self.index(self.unk)) for t in text])
            else:
                try:
                    ret.append([self.word2id[t.surface] for t in text])
                except KeyError:
                    raise KeyError(
                        "the input contains unknown word. "
                        "it is recommended to set `unk` attribute."
                    )
        return ret

    def inverse_transform(self, y: tp.List[tp.List[int]]) -> tp.List[tp.List[str]]:
        ret = []
        for idxs in y:
            ret.append([self.id2word[i] for i in idxs])
        return ret

    def __getitem__(self, i: int) -> str:
        if i < 0 or len(self) <= i:
            raise IndexError
        return self.id2word[i]

    def __len__(self) -> int:
        return len(self.id2word)

    def __contains__(self, w: str) -> bool:
        return w in self.word2id

    def index(self, w: str) -> int:
        return self.word2id[w]

    @property
    def vocab(self) -> tp.List[str]:
        return list(self.word2id.keys())

    @property
    def vocab_idx(self) -> tp.List[int]:
        return list(self.id2word.keys())
