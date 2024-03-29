#!/usr/bin/env python

from typing import List
from dataclasses import dataclass, field
import logging
from pathlib import Path
from itertools import chain
import pandas as pd
import plac
from zensols.persist import persisted
from zensols.config import ConfigFactory
from zensols.cli import CliHarness; CliHarness.add_sys_path(Path('src/python'))
from zensols import deepnlp
from zensols.install import Installer
from zensols.amr import ScorerApplication
from zensols.calamr.score import CalamrScorer

logger = logging.getLogger(__name__)

# initialize the NLP system
deepnlp.init()


@dataclass
class CliAmrScorer(object):
    """Creates the files used to score scorer output.

    """
    config: str = field()
    output_dir = Path('results/scores')
    parsed_dir: Path = field(default=Path('corpus/parsed'))

    @property
    @persisted('_config_factory')
    def config_factory(self) -> ConfigFactory:
        harness = CliHarness(
            src_dir_name='src/python',
            app_factory_class='zensols.calamr.ApplicationFactory')
        harness.configure_logging(
            loggers={__name__: 'info',
                     'zensols.amr': 'info',
                     'zensols.amr.score': 'debug'})
        ctx: ConfigFactory = harness.get_config_factory(
            ['--config', f'etc/{self.config}.conf'])
        return ctx

    @property
    def scorer(self) -> CalamrScorer:
        return self.config_factory('calamr_doc_scorer')

    def clear(self):
        self.scorer.clear()

    def score_parses(self):
        inst: Installer = self.config_factory('amr_anon_corpus_installer')
        gold: Path = inst.get_singleton_path()
        app: ScorerApplication = self.config_factory('sapp')
        parsed_dir: Path = self.parsed_dir / self.config
        inst.install()
        parse_file: Path
        for parse_file in parsed_dir.iterdir():
            fname: str = f'parse/{self.config}/{parse_file.stem}.csv'
            output_file: Path = self.output_dir / fname
            output_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f'score files: gold={gold}, ' +
                        f'parse={parse_file} -> {output_file}')
            app.score(
                input_gold=gold,
                input_parsed=parse_file,
                output_dir=output_file)

    def score_docs(self):
        df: pd.DataFrame = self.scorer.score_docs()
        output_file = self.output_dir / f'docs/{self.config}.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file)
        logger.info(f'wrote: {output_file}')


@plac.annotations(
    docs=('comma-separated list of corpus names to score as documents',
          'option', None, str),
    parses=('comma-separated list of corpus names to score as parser outputs',
            'option', None, str),
    clear=('delete previous cached score files', 'flag', 'c'))
def main(docs: str = 'proxy-report,mismatch-proxy',
         parses: str = 'lp,bio',
         clear: bool = False):
    """Score documents and then write them out to CSV files.

    """
    docs: List[str] = docs.split(',') if len(docs) > 0 else []
    parses: List[str] = parses.split(',') if len(parses) > 0 else []
    if clear:
        for config in chain.from_iterable((docs, parses)):
            logger.info(f'clearing previous score results: {config}')
            scorer = CliAmrScorer(config)
            scorer.clear()
    logger.info(f'scoring docs: {docs}')
    for config in docs:
        scorer = CliAmrScorer(config)
        scorer.score_docs()
    logger.info(f'scoring parsed: {parses}')
    for config in parses:
        scorer = CliAmrScorer(config)
        scorer.score_parses()


if (__name__ == '__main__'):
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    plac.call(main)
