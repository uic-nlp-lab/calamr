"""AMR parsing spaCy pipeline component and sentence generator.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple
from dataclasses import dataclass, field
import logging
import sys
import traceback
import json
from pathlib import Path
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from penman.exceptions import DecodeError
import amrlib
from amrlib.models.inference_bases import GTOSInferenceBase, STOGInferenceBase
from zensols.util import loglevel
from zensols.persist import persisted
from zensols.install import Installer
from zensols.nlp import FeatureDocumentParser, Component, ComponentInitializer
from . import AmrFailure, AmrSentence, AmrDocument

logger = logging.getLogger(__name__)


@dataclass
class ModelContainer(object):
    """Contains an installer used to download and install a model that's then used
    by the API to parse AMR graphs or generate language from AMR graphs.

    """
    name: str = field()
    """The section name."""

    installer: Installer = field(default=None)
    """"Use to install the model files.  The installer must have one and only
    resource.

    """
    alternate_path: Path = field(default=None)
    """If set, use this alternate path to find the model files."""

    @property
    def model_path(self) -> Path:
        if self.alternate_path is None:
            pkg_path = self.installer.get_singleton_path().parent
        else:
            pkg_path = self.alternate_path
        return pkg_path

    def _load_model(self) -> Path:
        self.installer.install()
        model_path: Path = self.model_path
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'resolved model path: {model_path}')
        if amrlib.defaults.data_dir != model_path:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'setting AMR model directory to {model_path}')
            amrlib.defaults.data_dir = model_path
        return model_path


@dataclass
class AmrParser(ModelContainer, ComponentInitializer):
    """Parses natural language in to AMR graphs.  It has the ability to change
    out different installed models in the same Python session.

    """
    add_missing_metadata: bool = field(default=True)
    """Whether to add missing metadata to sentences when missing.

    :see: :meth:`add_metadata`

    """
    model: str = field(default='noop')
    """The :mod:`penman` AMR model to use when creating :class:`.AmrSentence`
    instances, which is one of ``noop`` or ``amr``.  The first does not modify
    the graph but the latter normalizes out inverse relationships such as
    ``ARG*-of``.

    """
    def init_nlp_model(self, model: Language, component: Component):
        """Reset the installer to all reloads in a Python REPL with different
        installers.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'initializing ({id(model)}): {self.name}')
        doc_parser: FeatureDocumentParser = model.doc_parser
        new_parser: AmrParser = doc_parser.config_factory(self.name)
        self.installer = new_parser.installer

    @persisted('_parse_model', cache_global=True)
    def _get_parse_model(self) -> STOGInferenceBase:
        """The model that parses text in to AMR graphs.  This model is cached
        globally, as it is cached in the :mod:`amrlib` module as well.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('loading parse model')
        model_path = self._load_model()
        if model_path.name.find('gsii') > -1:
            with loglevel('transformers', logging.ERROR):
                model = amrlib.load_stog_model()
        else:
            model = amrlib.load_stog_model()
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'using parse model: {model.__class__}')
        return model, model_path

    def _clear_model(self):
        self._parse_model.clear()
        amrlib.stog_model = None

    @property
    def parse_model(self) -> STOGInferenceBase:
        model, prev_path = self._get_parse_model()
        cur_path = self.model_path
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'current path: {cur_path}, prev path: {prev_path}')
        if cur_path != prev_path:
            self._clear_model()
            model = self._get_parse_model()[0]
            amrlib.stog_model = model
        return model

    @staticmethod
    def needs_metadata(amr_sent: AmrSentence) -> bool:
        """T5 model sentences only have the ``snt`` metadata entry.

        :param amr_sent: the sentence to populate

        :see: :meth:`add_metadata`

        """
        return 'tokens' not in amr_sent.metadata

    @staticmethod
    def add_metadata(amr_sent: AmrSentence, sent: Span):
        """Add missing metadata parsed from spaCy if missing, which happens in
        the case of using the T5 AMR model.

        :param amr_sent: the sentence to populate

        :param sent: the spacCy sentence used as the source

        :see: :meth:`needs_metadata`

        """
        def map_ent(t: Token) -> str:
            et = t.ent_type_
            return 'O' if len(et) == 0 else et

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'add metadata: {amr_sent}')
        toks = tuple(map(lambda t: t.orth_, sent))
        lms = tuple(map(lambda t: t.lemma_, sent))
        amr_sent.set_metadata('tokens', json.dumps(toks))
        amr_sent.set_metadata('lemmas', json.dumps(lms))
        if hasattr(sent[0], 'ent_type_'):
            ents = tuple(map(map_ent, sent))
            amr_sent.set_metadata('ner_tags', json.dumps(ents))
        if hasattr(sent[0], 'tag_'):
            pt = tuple(map(lambda t: t.tag_, sent))
            amr_sent.set_metadata('pos_tags', json.dumps(pt))

    def __call__(self, doc: Doc) -> Doc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'parsing from {doc}')
        # force load the model in to the global amrlib module space
        stog_model: STOGInferenceBase = self.parse_model
        # add spacy underscore data holders for the amr data structures
        if not Doc.has_extension('amr'):
            Doc.set_extension('amr', default=[])
        if not Span.has_extension('amr'):
            Span.set_extension('amr', default=[])
        sent_graphs: List[AmrSentence] = []
        sent: Span
        for i, sent in enumerate(doc.sents):
            err: AmrFailure = None
            graphs: List[str] = None
            try:
                graphs = stog_model.parse_spans([sent])
                graph: str = graphs[0]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'adding graph for sent {i}: <{graph[:50]}>')
                if graph is None:
                    err = AmrFailure("Could not parse: empty graph " +
                                     f"(total={len(graphs)})", sent.text)
            except DecodeError as e:
                stack = traceback.format_exception(*sys.exc_info())
                err = AmrFailure(f'Could not decode: {e}', sent.text, stack)
            except Exception as e:
                stack = traceback.format_exception(*sys.exc_info())
                err = AmrFailure(f'Could not parse: {e}', sent.text, stack)
            if err is not None:
                sent._.amr = AmrSentence(err)
                sent_graphs.append(sent._.amr)
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'creating sentence with model: {self.model}')
                amr_sent = AmrSentence(graph, model=self.model)
                if self.add_missing_metadata and self.needs_metadata(amr_sent):
                    self.add_metadata(amr_sent, sent)
                sent._.amr = amr_sent
                sent_graphs.append(amr_sent)
        doc._.amr = AmrDocument(sent_graphs)
        return doc


@dataclass
class AmrGenerator(ModelContainer):
    """A callable that generates natural language text from an AMR graph.

    :see: :meth:`__call__`

    """
    @property
    @persisted('_generation_model', cache_global=True)
    def generation_model(self) -> GTOSInferenceBase:
        """The model that generates sentences from an AMR graph."""
        logger.debug('loading generation model')
        self._load_model()
        return amrlib.load_gtos_model()

    def __call__(self, doc: AmrDocument) -> Tuple[str]:
        """Generate a sentence from a spaCy document.

        :param doc: the spaCy document used to generate the sentence

        :return: a text sentence for each respective sentence in ``doc``

        """
        model = self.generation_model
        return tuple(map(lambda s: model.generate(s.graph_string), doc))


@Language.factory('amr_parser')
def create_amr_parser(nlp: Language, name: str, parser_name: str):
    """Create an instance of :class:`.AmrParser`.

    """
    doc_parser: FeatureDocumentParser = nlp.doc_parser
    if logger.isEnabledFor(logging.INFO):
        logger.info(f'creating AMR component {name}: doc parser: {doc_parser}')
    parser: AmrParser = doc_parser.config_factory(parser_name)
    return parser
