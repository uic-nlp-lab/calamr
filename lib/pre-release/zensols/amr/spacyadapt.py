"""A set of adaptor classes from :class:`zensols.nlp.FeatureToken` to
:class:`spacy.tokens.Doc`.

"""
__author__ = 'Paul Landes'

from typing import Dict
from zensols.config import persisted
from zensols.nlp import TokenContainer, FeatureToken, FeatureDocument


class _Underscore(object):
    pass


class _SpacyTokenAdapter(object):
    def __init__(self, ftok: FeatureToken, sent):
        self._ftok = ftok
        self.sent = sent

    def __getattr__(self, attr, default=None):
        v = None
        ft: FeatureToken = self._ftok
        if attr == 'text' or attr == 'orth_':
            v = ft.norm
        else:
            if hasattr(ft, 'spacy_token'):
                v = getattr(ft.spacy_token, attr)
            else:
                if attr == 'ent_type_':
                    v = ft.ent_
                    if v == FeatureToken.NONE:
                        v = ''
                elif attr == 'doc':
                    v = self.sent._doc
                else:
                    v = getattr(self._ftok, attr)
        return v

    def __str__(self):
        return self._ftok.norm

    def __repr__(self):
        return self.__str__()


class _SpacySpanAdapter(object):
    def __init__(self, cont: TokenContainer, doc):
        self._cont = cont
        self._doc = doc
        self._ = _Underscore()
        self._toks = tuple(map(lambda t: _SpacyTokenAdapter(t, self),
                               self._cont.token_iter()))

    @property
    def text(self):
        # use origianl text as norm'd version might include spacing between
        # split multi-word entities
        return self._cont.norm

    @property
    def start(self):
        return self._cont[0].i

    @persisted('_i_sent_to_tok_pw')
    def _i_sent_to_tok(self) -> Dict[int, _SpacyTokenAdapter]:
        return {ta._ftok.i_sent: ta for ta in self}

    def __getitem__(self, i):
        ta: _SpacyTokenAdapter = self._i_sent_to_tok()[i]
        return ta

    def __iter__(self):
        return iter(self._toks)

    def __str___(self):
        return str(self.text)

    def __repr__(self):
        return str(self.text)


class SpacyDocAdapter(_SpacySpanAdapter):
    """Adaps a :class:`zensols.nlp.FeatureDocument` to a
    :class:`spacy.tokens.Doc`.

    """
    def __init__(self, cont: FeatureDocument):
        super().__init__(cont, cont)

    @persisted('_i_to_tok_pw')
    def _i_to_tok(self) -> Dict[int, _SpacyTokenAdapter]:
        return {ta._ftok.i: ta for ta in self}

    @property
    @persisted('_sents')
    def sents(self):
        return tuple(map(lambda s: _SpacySpanAdapter(s, self),
                         self._cont.sents))

    def __getitem__(self, i):
        ta: _SpacyTokenAdapter = self._i_to_tok()[i]
        return ta

    def __len__(self):
        return len(self._toks)
