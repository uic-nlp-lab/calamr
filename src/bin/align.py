#!/usr/bin/env python

from typing import Tuple
from dataclasses import dataclass, field
import logging
from pathlib import Path
import plac
from zensols.persist import persisted
from zensols.config import ConfigFactory
from zensols.cli import CliHarness; CliHarness.add_sys_path(Path('src/python'))
from zensols import deepnlp
from zensols.dataset import SplitKeyContainer
from zensols.calamr.result import AlignmentResultGenerator

logger = logging.getLogger(__name__)

# initialize the NLP system
deepnlp.init()


@dataclass
class CliAmrAligner(object):
    """Creates the files used to score scorer output.

    """
    config: str = field()
    split_name: str = field()
    name: str = field()
    only_report_reentrancies: bool = field()
    limit: int = field()

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
        oval: str = str(self.only_report_reentrancies)
        ctx: ConfigFactory = harness.get_config_factory(
            ['--config', f'etc/{self.config}.conf',
             '--override', f'calamr_default.only_report_reentrancies={oval}'])
        return ctx

    def __call__(self):
        cont: SplitKeyContainer = self.config_factory(
            'calamr_amr_corp_split_keys')
        keys: Tuple[str] = cont.keys_by_split[self.split_name]
        assert type(keys) == tuple
        gen: AlignmentResultGenerator = self.config_factory(
            'calamr_align_result_generator')
        if self.limit is not None:
            gen.limit = self.limit
        logger.info(f'aligning: {self}')
        gen(self.name, self.split_name, keys)


@plac.annotations(
    config=('the config name without the extension', 'positional', None, str),
    split=('the name of the split', 'positional', None, str,
           'training test dev'.split()),
    reentry_type=('whether to fix reentrancies', 'positional', None, str,
                  'fix only-report'.split()),
    limit=('max number of document to process', 'option', None, int))
def main(config: str, split: str, reentry_type: str, limit: int = None):
    """Align proxy report articles by split and indicate whether to fix
    reentracy flows.

    """
    aligner = CliAmrAligner(
        config=config,
        split_name=split,
        name=reentry_type,
        only_report_reentrancies=(reentry_type == 'only-report'),
        limit=limit)
    aligner()


if (__name__ == '__main__'):
    plac.call(main)
