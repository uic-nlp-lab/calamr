#!/usr/bin/env python

from dataclasses import dataclass
import logging
from pathlib import Path
from zensols.cli import CliHarness
from zensols.install import Installer
from zensols import deepnlp
from zensols.amr import Application as AmrApplication

logger = logging.getLogger(__name__)

# initialize the NLP system
deepnlp.init()


@dataclass
class CliMismatchCorp(object):
    """Creates the corpus that swaps summaries and source text in the proxy
    report corpus.

    :see: :class:`.corpus.ProxyReportMismatchCorpusWriter`

    """
    def create(self):
        harness = CliHarness(
            root_dir=Path.cwd(),
            src_dir_name='src/python',
            app_factory_class='zensols.calamr.ApplicationFactory')
        # disable coreference resolution to speed up since this just produces an
        # AMR text output file
        oconf: str = (
            'amr_anon_feature_doc_stash.word_piece_doc_factory=None,' +
            'amr_anon_feature_doc_stash.coref_resolver=None,' +
            'amr_anon_doc_parser.coref_resolver=None')
        app: AmrApplication = harness.get_application(
            ['--config', 'etc/proxy-report.conf',
             f'--override={oconf}'],
            app_section='capp')
        harness.configure_logging(
            loggers={__name__: 'info',
                     'zensols.amr': 'info',
                     'zensols.amr.score': 'debug'})
        installer: Installer = app.config_factory('amr_anon_corpus_installer')
        installer()
        corp_path: Path = installer.get_singleton_path()
        #corp_path.parent.mkdir(parents=True, exist_ok=True)
        writer = app.config_factory('calamr_mismatch_corpus_writer')
        writer.write()


if (__name__ == '__main__'):
    CliMismatchCorp().create()
