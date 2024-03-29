"""Add alignments to AMR sentences.

"""
__author__ = 'Paul Landes'

from typing import Union, Callable, Type, Dict, ClassVar
from dataclasses import dataclass, field
import logging
import os
from frozendict import frozendict
from pathlib import Path
from spacy.tokens import Doc, Span
from spacy.language import Language
from amrlib.alignments.faa_aligner import FAA_Aligner
from amrlib.alignments.rbw_aligner import RBWAligner
from zensols.persist import persisted
from . import AmrError, AmrSentence, AmrDocument, AmrParser, AlignmentPopulator

logger = logging.getLogger(__name__)


@dataclass
class _RuleAligner(object):
    """Uses the :mod:`amrlib` :class:`~amrlib.alignments.rbw_aligner.RBWAligner`
    aligner.

    :see: `Amrlib Docs <https://amrlib.readthedocs.io/en/latest/rbw_aligner/>`_

    """
    NAME = 'rule'

    def __call__(self, amr_sent: AmrSentence):
        # use this with an annotated penman graph object
        aligner = RBWAligner.from_penman_w_json(amr_sent.graph)
        # get the aligned graph string
        graph_string = aligner.get_graph_string()
        # udate the AMR sentence with the alignment settings
        amr_sent.graph_string = graph_string


@dataclass
class _FastAligner(object):
    """Uses Dyer et al. :class:`~amrlib.alignments.faa_aligner.FAA_Aligner`
    aligner.

    :see: `Amrlib Docs <https://amrlib.readthedocs.io/en/latest/faa_aligner/>'_

    """
    NAME: ClassVar[str] = 'faa'
    _ENVIRONMENT_VARIABLE_NAME: ClassVar[str] = 'FABIN_DIR'
    _AVAIL: ClassVar[str] = 'unk'

    @classmethod
    def is_available(cls: Type, raise_error: bool = False) -> bool:
        """Return if the FAA aligner is available based on the presense of the
        ``FABIN_DIR`` environment variable and if it points to a directory.

        :param raise_error: if ``True`` raise an error when the aligner is not
                            available

        :raises AmrError: if ``raise_error`` is ``True`` and the FAA aligner is
                          not available

        """
        if cls._AVAIL == 'unk':
            ev: str = cls._ENVIRONMENT_VARIABLE_NAME
            avail: bool = ev in os.environ
            cls._AVAIL = 'avail'
            if not avail:
                cls._AVAIL = 'noenv'
            else:
                fdir: Path = Path(os.environ[ev])
                if not fdir.is_dir():
                    cls._AVAIL = 'nodir'
        if raise_error:
            if cls._AVAIL == 'noenv':
                raise AmrError(f"Expecting environment variable: '{ev}' " +
                               "when using FAA aligner")
            elif cls._AVAIL == 'nodir':
                raise AmrError(f"Not a directory pointed to by '{ev}': {fdir}")
        return cls._AVAIL == 'avail'

    def __call__(self, amr_sent: AmrSentence):
        meta: Dict[str, str] = amr_sent.metadata
        graphs = [amr_sent.graph_string]
        inference = FAA_Aligner()
        # force white space tokenization to match the already tokenized
        # metadata ('tokens' key); examples include numbers followed by commas
        # such as dates like "April 25 , 2008"
        sent_text: str = amr_sent.tokenized_text
        #sent_text: str = amr_sent.text
        surface_aligns, aligns = inference.align_sents([sent_text], graphs)
        meta['alignments'] = aligns[0].strip()
        amr_sent.metadata = meta
        align_populator = AlignmentPopulator(amr_sent.graph)
        if len(align_populator()) > 0:
            amr_sent.invalidate_graph_string()


@dataclass
class AmrAlignmentPopulator(object):
    """Adds alignment markers to AMR graphs.

    """
    _ALIGNERS: ClassVar[Dict[str, Type]] = frozendict(
        {c.NAME: c for c in (_RuleAligner, _FastAligner)})

    aligner: Union[str, Callable] = field()
    """The aligner used to annotate sentence AMR graphs."""

    add_missing_metadata: bool = field(default=True)
    """Whether to add missing metadata to sentences when missing."""

    def __post_init__(self):
        if isinstance(self.aligner, str):
            aligner = self.aligner
            if aligner == 'best':
                aligner = 'faa' if _FastAligner.is_available() else 'rule'
            aligner_cls: Type = self._ALIGNERS.get(aligner)
            if aligner_cls is None:
                raise AmrError(f'Unknown aligner: {self.aligner}')
            self.aligner = aligner_cls()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'using aligner: {self.aligner}')

    @persisted('_assert_module_log_pw', cache_global=True)
    def _assert_module_log(self):
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'aligning using: {self.aligner.NAME}')
        return True

    def __call__(self, doc: Union[Doc, AmrDocument]) -> \
            Union[Doc, AmrDocument]:
        """Add alignment markers to sentence AMR graphs."""
        self._assert_module_log()
        spacy_sent: Span = None
        sent: Union[AmrSentence, Span]
        for sent in doc.sents:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'aligning: {sent}')
            if not isinstance(sent, AmrSentence):
                spacy_sent = sent
                sent: AmrSentence = sent._.amr
            if sent.is_failure:
                logger.warning(f'skipping alignment for failure: {sent}')
            elif sent.has_alignments:
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f'AMR already has alignments--skipping: {sent}')
            elif not sent.has_alignments:
                if self.add_missing_metadata and AmrParser.needs_metadata(sent):
                    if spacy_sent is None:
                        raise AmrError('No spacy sentence span available')
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(f'adding metadata to {sent}')
                    AmrParser.add_metadata(sent, spacy_sent)
                try:
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(f'adding alignments to {sent}')
                    self.aligner(sent)
                except Exception as e:
                    logger.error(f'could not load align <{sent}>: {e}')
                    raise e
        return doc


@Language.factory('amralign', default_config={'aligner': 'rule'})
def create_amr_align_component(nlp: Language, name: str, aligner: str):
    """Create an instance of :class:`.AmrAlignmentPopulator`.

    """
    return AmrAlignmentPopulator(aligner)
