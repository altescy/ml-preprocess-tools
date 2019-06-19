from __future__ import annotations
import typing as tp
import enum


class TokenType(enum.Enum):
    PLAIN = 0
    SPACY = 1
    JA = 2


class Token:
    def __init__(self, token: tp.Any, token_type: TokenType = TokenType.SPACY) -> None:
        self._token = token
        self.token_type = token_type
        self.__surface = ''

    def set_surface(self, surface: str) -> None:
        self.__surface = surface

    @property
    def is_spacy(self) -> bool:
        return self.token_type == TokenType.SPACY

    @property
    def is_ja(self) -> bool:
        return self.token_type == TokenType.JA

    @property
    def surface(self) -> str:
        if self.__surface:
            return self.__surface
        if self.is_spacy:
            return self._token.text
        if self.is_ja:
            return self._token.surface
        return self._token

    @property
    def base_form(self) -> str:
        if self.is_spacy:
            return self._token.lemma_
        if self.is_ja:
            return self._token.base_form
        return '-'

    @property
    def pos(self) -> str:
        if self.is_spacy:
            return self._token.pos_
        if self.is_ja:
            return self._token.part_of_speech.split(',')[0]
        return '-'

    @property
    def tag(self) -> str:
        if self.is_spacy:
            return self._token.tag_
        if self.is_ja:
            return self._token.part_of_speech.split(',')[1]
        return '-'

    def __repr__(self) -> str:
        return '<{}:{}>'.format(self.surface, self.pos)

    def __reduce_ex__(self, proto) -> tp.Any:
        return type(self), (self._token, self.token_type)
