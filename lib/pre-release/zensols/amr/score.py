"""Produces matching scores.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Iterable, Dict, Any, Type
from dataclasses import dataclass, field
import logging
import smatch
from zensols.nlp import FeatureDocumentParser
from zensols.nlp.score import (
    ErrorScore, ScoreMethod, ScoreContext, HarmonicMeanScore
)
from . import (
    AmrError, AmrSentence, AmrDocument, AmrFeatureSentence, AmrFeatureDocument
)

logger = logging.getLogger(__name__)


@dataclass
class SmatchScoreCalculator(ScoreMethod):
    """Computes the smatch scores of AMR sentences.

    """
    @classmethod
    def _get_external_modules(cls: Type) -> Tuple[str, ...]:
        return ('smatch',)

    @staticmethod
    def _match_pair(s1: str, s2: str) -> Tuple[float, float, float]:
        """Match two AMRs as single line Penman notation strings.

        Taken from :mod:`amrlib`.

        """
        smatch.match_triple_dict.clear()  # clear the matching triple dictionary
        try:
            ret = smatch.get_amr_match(s1, s2)
            return ret
        except Exception as e:
            raise AmrError(f'Could not score <{s1}>::<{s2}>: {e}', e)

    @classmethod
    def _compute_smatch(cls: Type, test_entries: Iterable[str],
                        gold_entries: Iterable[str]) -> \
            Tuple[float, float, float]:
        """Score a list of entry pairs.  The entries should be a list of single
        line strings.

        Taken from :mod:`amrlib`.

        """
        pairs = zip(test_entries, gold_entries)
        mum_match = mum_test = mum_gold = 0
        for (n1, n2, n3) in map(lambda p: cls._match_pair(*p), pairs):
            mum_match += n1
            mum_test += n2
            mum_gold += n3
        precision, recall, f_score = smatch.compute_f(
            mum_match, mum_test, mum_gold)
        return precision, recall, f_score

    def sentence_score(self, gold: AmrSentence, pred: AmrSentence) -> \
            HarmonicMeanScore:
        """Return the smatch score produced from two AMR sentences.

        :return: a score with the precision, recall and F-score

        """
        return HarmonicMeanScore(*self._compute_smatch(
            [pred.graph_single_line], [gold.graph_single_line]))

    def document_smatch(self, gold: AmrDocument, pred: AmrDocument) -> \
            HarmonicMeanScore:
        """Return the smatch score produced from the sentences as pairs from two
        documents.

        :return: a score with the precision, recall and F1

        """
        goldl: int = len(gold)
        predl: int = len(pred)
        if goldl != predl:
            raise AmrError('Expecting documents with same length, ' +
                           f'but got: {goldl} != {predl}')
        golds: Tuple[str] = tuple(map(lambda s: s.graph_single_line, gold.sents))
        preds: Tuple[str] = tuple(map(lambda s: s.graph_single_line, pred.sents))
        return HarmonicMeanScore(*self._compute_smatch(golds, preds))

    def _score(self, meth: str, context: ScoreContext) -> \
            Iterable[HarmonicMeanScore]:
        gold: AmrFeatureSentence
        pred: AmrFeatureSentence
        for gold, pred in context.pairs:
            assert isinstance(gold, AmrFeatureSentence)
            assert isinstance(pred, AmrFeatureSentence)
            try:
                yield self.sentence_score(gold.amr, pred.amr)
            except Exception as e:
                yield ErrorScore(meth, e, HarmonicMeanScore.NAN_INSTANCE)


@dataclass
class AmrScoreParser(object):
    """Parses :class:`.AmrSentence` instances from the ``snt`` metadata text
    string from a human annotated AMR.  It then returns an instance that is to
    later be scored by :class:`.ScoreMethod` such as
    :class:`.SmatchScoreCalculator`.

    """
    doc_parser: FeatureDocumentParser = field()
    """The document parser used to generate the AMR.  This should have sentence
    boundaries removed so only one :class:`.AmrSentence` is returned from the
    parse.

    """
    keep_keys: Tuple[str] = field(default=None)
    """The keys to keep/copy from the source :class:`.AmrSentence`.

    """
    def parse(self, sent: AmrSentence) -> AmrSentence:
        """Parse the ``snt`` metadata from ``sent`` and as an AMR sentence.

        """
        meta: Dict[str, Any] = sent.metadata
        text: str = meta['snt']
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'parsing: <{text}>')
        parse_doc: AmrFeatureDocument = self.doc_parser(text)
        if len(parse_doc) != 1:
            raise AmrError(
                'Expecting one sentence but got: ' +
                f'{len(parse_doc)} for <{text}>')
        feat_parse_sent: AmrFeatureSentence = parse_doc[0]
        amr_parse_sent: AmrSentence = feat_parse_sent.amr
        if self.keep_keys is not None:
            meta = {k: meta[k] for k in self.keep_keys}
            amr_parse_sent.metadata = meta
        return amr_parse_sent
