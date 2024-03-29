"""Produces CALAMR scores.

"""
__author__ = 'Paul Landes'

from typing import Iterable, Tuple, Dict, Any, List, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import sys
import logging
import itertools as it
from pathlib import Path
import pandas as pd
from zensols.persist import Stash, PersistedWork, persisted
from zensols.nlp.score import (
    ErrorScore, ScoreMethod, ScoreContext, Score, Scorer, ScoreSet
)
from zensols.amr import (
    AmrFeatureSentence, AmrFeatureDocument,
    SentenceType, AnnotatedAmrSentence, AnnotatedAmrDocument,
)
from zensols.dataset import SplitKeyContainer
from zensols.deepnlp.transformer import (
    WordPieceFeatureDocumentFactory, WordPieceFeatureDocument
)
from . import (
    DocumentGraph, DocumentGraphFactory, DocumentGraphAligner,
    FlowDocumentGraph,
)

logger = logging.getLogger(__name__)


@dataclass
class CalamrScore(Score):
    """Contains all CALAMR scores.

    """
    flow_doc_graph: FlowDocumentGraph = field(repr=False)

    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
        a_cols: List[str] = 'mean_flow aligned_portion'.split()
        stats: Dict[str, Any] = self.flow_doc_graph.stats
        agg: Dict[str, Any] = stats['agg']
        ret: Dict[str, Any] = OrderedDict({f'agg_{k}': agg[k] for k in a_cols})
        ret['bipartite_relations'] = stats['bipartite_relations']
        for cname, val in stats['components'].items():
            ret[f'{cname}_aligned_portion'] = \
                val['connected']['aligned_portion']
            vk: str
            for vk in val.keys():
                if vk == 'connected' or vk == 'counts':
                    continue
                ret[f'{cname}_{vk}'] = val[vk]
            ck: str
            dv: Dict[str, int]
            for ck, dv in val['counts'].items():
                if isinstance(dv, dict):
                    dk: str
                    cnt: int
                    for dk, cnt in dv.items():
                        ret[f'{cname}_{ck}_{dk}_count'] = cnt
                else:
                    ret[f'{cname}_{ck}_count'] = cnt
        return ret

    def __str__(self) -> str:
        return ', '.join(map(lambda s: f'{s[0]}: {s[1]:.4f}',
                             self.asflatdict().items()))


CalamrScore.NAN_INSTANCE = CalamrScore(FlowDocumentGraph())


@dataclass
class CalamrScoreMethod(ScoreMethod):
    """Computes the smatch scores of AMR sentences.  Sentence pairs are ordered
    ``(<summary>, <source>)``.

    """
    word_piece_doc_factory: WordPieceFeatureDocumentFactory = field(
        default=None)
    """The feature document factory that populates embeddings."""

    doc_graph_factory: DocumentGraphFactory = field(default=None)
    """Create document graphs."""

    doc_graph_aligner: DocumentGraphAligner = field(default=None)
    """Create document graphs."""

    def _to_annotated_sent(self, sent: AmrFeatureSentence,
                           sent_type: SentenceType) -> \
            Tuple[AmrFeatureSentence, bool]:
        mod: bool = False
        if not isinstance(sent.amr, AnnotatedAmrSentence):
            asent = sent.amr.clone(
                cls=AnnotatedAmrSentence,
                sent_type=sent_type,
                doc_sent_idx=0)
            sent = sent.clone()
            sent.amr = asent
            mod = True
        return sent, mod

    def _populate_embeddings(self, doc: AmrFeatureDocument):
        """Adds the transformer sentinel embeddings to the document."""
        wpdoc: WordPieceFeatureDocument = self.word_piece_doc_factory(doc)
        wpdoc.copy_embedding(doc)

    def score_annotated_doc(self, doc: AmrFeatureDocument) -> CalamrScore:
        """Score a document that has an
        :obj:`~zensols.amr.container.AmrFeatureDocument.amr` of type
        :class:`~zensols.amr.annotate.AnnotatedAmrDocument`.

        :raises: [zensols.amr.domain.AmrError]: if the AMR could not be parsed
                 or aligned

        """
        assert isinstance(doc, AmrFeatureDocument)
        doc_graph: DocumentGraph = self.doc_graph_factory(doc)
        prev_render_level: int = self.doc_graph_aligner.render_level
        self.doc_graph_aligner.render_level = 0
        try:
            fdg: FlowDocumentGraph = self.doc_graph_aligner.align(doc_graph)
            return CalamrScore(fdg)
        finally:
            self.doc_graph_aligner.render_level = prev_render_level

    def score_pair(self, smy: Union[AmrFeatureSentence, AmrFeatureDocument],
                   src: Union[AmrFeatureSentence, AmrFeatureDocument]) -> \
            CalamrScore:
        if isinstance(src, AmrFeatureDocument) and \
           isinstance(smy, AmrFeatureDocument):
            src_mod, smy_mod = False, False
            fsents = tuple(list(src.sents) + list(smy.sents))
            asents = tuple(map(lambda s: s.amr, fsents))
            fdoc = AmrFeatureDocument(sents=fsents)
            fdoc.amr = AnnotatedAmrDocument(sents=asents)
        else:
            assert isinstance(src, AmrFeatureSentence)
            assert isinstance(smy, AmrFeatureSentence)
            src, src_mod = self._to_annotated_sent(src, SentenceType.BODY)
            smy, smy_mod = self._to_annotated_sent(smy, SentenceType.SUMMARY)
            fdoc = AmrFeatureDocument(sents=(smy, src))
            fdoc.amr = AnnotatedAmrDocument(sents=(smy.amr, src.amr))
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'scoring <{smy}> :: <{src}>')
        if src_mod or smy_mod:
            self._populate_embeddings(fdoc)
        return self.score_annotated_doc(fdoc)

    def _score(self, meth: str, context: ScoreContext) -> Iterable[CalamrScore]:
        smy: AmrFeatureSentence
        src: AmrFeatureSentence
        for smy, src in context.pairs:
            try:
                yield self.score_pair(smy, src)
            except Exception as e:
                logger.error(f'can not score: <{src}>::<{smy}>: {e}',
                             stack_info=True, exc_info=True)
                yield ErrorScore(meth, e, CalamrScore.NAN_INSTANCE)

    def clear(self):
        self.doc_graph_aligner.clear()


@dataclass
class CalamrScorer(Scorer):
    """Scores AMR both annotated and non-annotated documents.  Annotations of
    the documents include whether each sentence is part the source or the
    summary.

    """
    score_method: CalamrScoreMethod = field(default=None)
    """The scorer to use for annotated AMR documents."""

    amr_corp_split_keys: SplitKeyContainer = field(default=None)
    """Gets the splits of an AMR corpus release."""

    anon_doc_stash: Stash = field(default=None)
    """Contains human annotated AMRs from a small toy corpus or from the AMR 3.0
    Proxy Report corpus.

    """
    split_name: str = field(default=None)
    """The name of the split to score."""

    cache_dir: Path = field(default=None)
    """Directory to store cached data."""

    doc_limit: int = field(default=sys.maxsize)
    """The limit documents to score."""

    def __post_init__(self):
        self._score_docs = PersistedWork(
            self.cache_dir / 'docs.dat', self, mkdir=True)

    @property
    def doc_keys(self) -> Tuple[str]:
        """The document keys to use for scoring, which are provided by
        :obj:`amr_corp_split_keys` for the :obj:`split_name` split.

        """
        sk: SplitKeyContainer = self.amr_corp_split_keys
        sk.prime()
        if len(sk.keys_by_split) == 0:
            # only the proxy has more than one data split
            return self.anon_doc_stash.keys()
        else:
            return sk.keys_by_split[self.split_name]

    def score_set_for_doc(self, doc: AnnotatedAmrDocument) -> ScoreSet:
        """Score an annotated document using all method configured in
        :obj:`methods`.

        :param doc: the document to score

        :return: the scores of all methods

        """
        assert isinstance(doc.amr, AnnotatedAmrDocument)
        src_sents: Tuple[AmrFeatureSentence] = tuple(
            doc.amr.get_feature_sentences(doc, doc.amr.sections))
        smy_sents: Tuple[AmrFeatureSentence] = tuple(
            doc.amr.get_feature_sentences(doc, [doc.amr.summary]))
        src = AmrFeatureDocument(sents=src_sents)
        src.amr = AnnotatedAmrDocument(
            sents=tuple(map(lambda s: s.amr, src_sents)))
        smy = AmrFeatureDocument(sents=smy_sents)
        smy.amr = AnnotatedAmrDocument(
            sents=tuple(map(lambda s: s.amr, smy_sents)))
        return self.score(ScoreContext([[src, smy]]))

    def score_doc(self, doc: AnnotatedAmrDocument) -> pd.DataFrame:
        """Score an annotated document using all method configured in
        :obj:`methods`.

        :param doc: the document to score

        :return: the scores of all methods as a dataframe

        """
        res: ScoreSet = self.score_set_for_doc(doc)
        df: pd.DataFrame = res.as_dataframe()
        cols: List[str] = df.columns.tolist()
        cname: str = 'doc_id'
        df[cname] = doc.amr.doc_id
        cols.insert(0, cname)
        df = df[cols]
        return df

    @persisted('_score_docs')
    def score_docs(self) -> pd.DataFrame:
        """Score all documents identified by :obj:`doc_keys` and return all
        methods configured in :obj:`methods`.

        """
        logger.info(f'scoring documents with limit: {self.doc_limit}')
        dfs: List[pd.DataFrame] = []
        doc_keys: List[str] = sorted(self.doc_keys)
        dk: str
        for dk in it.islice(doc_keys, self.doc_limit):
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'scoring document {dk}')
            doc: AmrFeatureDocument = self.anon_doc_stash[dk]
            dfs.append(self.score_doc(doc))
        return pd.concat(dfs)

    def score_docs_by_key(self, doc_keys: Tuple[str]) -> pd.DataFrame:
        """Score documents identified by :obj:`doc_keys` and return all methods
        configured in :obj:`methods`.

        """
        logger.info(f'scoring documents with limit: {self.doc_limit}')
        dfs: List[pd.DataFrame] = []
        dk: str
        for dk in it.islice(doc_keys, self.doc_limit):
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'scoring document {dk}')
            doc: AmrFeatureDocument = self.anon_doc_stash[dk]
            dfs.append(self.score_doc(doc))
        return pd.concat(dfs)

    def clear(self):
        """Clear the :obj:`score_docs` cache."""
        self._score_docs.clear()
