"""Parses Proxy report AMR data in to annotated AMR documents.  From the LDC
documentation:

3.10 Narrative text "Proxy Reports" from newswire data (proxy)

This data was selected and segmented from the proxy report data in
LDC's DEFT Narrative Text Source Data R1 corpus (LDC2013E19) for AMR
annotation because they are developed from and thus rich in events and
event-relations commonly found in newswire data, but also have a
templatic, report-like structure which is more difficult for machines
to process.

:link: `LDC2020T02 <https://catalog.ldc.upenn.edu/docs/LDC2020T02/README.txt>`_

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Dict
from dataclasses import dataclass, field
import logging
import re
from pathlib import Path
from zensols.persist import Stash
from zensols.install import Installer
from zensols.dataset import SplitKeyContainer
from zensols.amr import (
    AmrSentence, AmrDocument, AmrFeatureDocument,
    AnnotatedAmrDocument, AnnotatedAmrSectionDocument
)
from .. import ComponentAlignmentError

logger = logging.getLogger(__name__)


@dataclass
class ProxyReportAnnotatedAmrDocument(AnnotatedAmrDocument):
    """Overrides the sections property to skip duplicate summary sentences also
    found in the body.

    """
    @property
    def sections(self) -> Tuple[AnnotatedAmrSectionDocument]:
        """The sentences that make up the body of the document."""
        def filter_sents(s: AmrSentence) -> bool:
            return s.text not in sum_sents

        sum_sents = set(map(lambda s: s.text, self.summary))
        secs = super().sections
        for sec in secs:
            sec.sents = tuple(filter(filter_sents, sec.sents))
        return secs


@dataclass
class ProxyReportMismatchCorpusWriter(object):
    """This class switches the summary and source (body) in the proxy report
    corpus.  This is used to generate statistics used in the paper for
    comparison with the original corpus.

    """
    _KEY_REGEX = re.compile(r'^(PROXY[A-Z_]+_[0-9_]+)\.(\d+)')
    """The proxy report key regular expression used to parse out the
    non-sentence portion.

    """
    anon_doc_stash: Stash = field()
    """Contains human annotated AMRs from the AMR 3.0 Proxy Report corpus.

    """
    amr_corp_split_keys: SplitKeyContainer = field()
    """Gets the splits of an AMR corpus release."""

    installer: Installer = field()
    """The installer used for the original corpus."""

    relative_corpus_file: Path = field()
    """The path to the source human annotated corpus as a relative directory."""

    def _get_base_id(self, s: str) -> str:
        m: re.Match = self._KEY_REGEX.match(s)
        return m.group(1)

    @property
    def corpus_file(self) -> Path:
        self.installer()
        org_corpus_path: Path = self.installer.get_singleton_path()
        corpus_path: Path = org_corpus_path / self.relative_corpus_file
        corpus_path = corpus_path.resolve()
        return corpus_path

    def write(self):
        stash: Stash = self.anon_doc_stash
        kt: Tuple[str] = self.amr_corp_split_keys.keys_by_split['test']
        kd: Tuple[str] = self.amr_corp_split_keys.keys_by_split['dev']
        kl: int = min(len(kt), len(kd))
        kt = kt[:kl]
        kd = kd[:kl]
        corpus_path: Path = self.corpus_file
        if corpus_path.exists():
            raise ComponentAlignmentError(
                f'Corpus file already exists: {corpus_path}--delete first')
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_path, 'w') as f:
            for src_id, smy_id in zip(kt, kd):
                src_doc: AmrFeatureDocument = stash[src_id]
                smy_doc: AmrFeatureDocument = stash[smy_id]
                prid: str = smy_doc[0].amr.metadata['id']
                base_id: str = self._get_base_id(prid)
                mm_sents: List[AmrSentence] = list(filter(
                    lambda s: s.metadata['snt-type'] != 'body',
                    smy_doc.amr.sents))
                mm_sents.extend(filter(
                    lambda s: s.metadata['snt-type'] == 'body',
                    src_doc.amr.sents))
                sent: AmrSentence
                for six, sent in enumerate(mm_sents):
                    org_meta: Dict[str, str] = sent.metadata
                    meta: Dict[str, str] = {}
                    for k in 'snt snt-type'.split():
                        meta[k] = org_meta[k]
                    meta['org_id'] = org_meta['id']
                    meta['id'] = f'{base_id}.{six + 1}'
                    sent.metadata = meta
                mm_doc = AmrDocument(sents=mm_sents)
                mm_doc.write(writer=f)
                logger.info(f'processed {prid}')
        logger.info(f'wrote corpus to {corpus_path}')
