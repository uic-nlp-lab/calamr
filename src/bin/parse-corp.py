#!/usr/bin/env python

from typing import Tuple
from dataclasses import dataclass, field
import logging
from pathlib import Path
from zensols.persist import persisted
from zensols.config import ConfigFactory
from zensols.install import Installer
from zensols.cli import CliHarness
from zensols import deepnlp
from zensols.amr import ScorerApplication

logger = logging.getLogger(__name__)


# initialize the NLP system
deepnlp.init()


@dataclass
class CliAmrParser(object):
    """Creates the files used to score parser output.

    """
    config: str = field()

    @persisted('_paths')
    def _get_paths(self) -> Tuple[Path, Path]:
        from zensols.calamr import ApplicationFactory
        harness: CliHarness = ApplicationFactory.create_harness()
        app = harness.get_application(
            f'-c etc/{self.config}.conf',
            app_section='aapp')
        fac: ConfigFactory = app.config_factory
        installer: Installer = fac('amr_anon_corpus_installer')
        installer()
        src: Path = installer.get_singleton_path()
        dst: Path = src.parent.parent / 'parsed' / self.config
        return src, dst

    def parse(self, model: str) -> Path:
        from zensols.amr import ApplicationFactory
        harness: CliHarness = ApplicationFactory.create_harness()
        src: Path
        dst: Path
        src, dst = self._get_paths()
        dst = dst / f'{model}.txt'
        dst.parent.mkdir(parents=True, exist_ok=True)
        ctx: ConfigFactory = harness.get_config_factory(
            ['--override', f'amr_default.parse_model={model}'])
        app: ScorerApplication = ctx('sapp')
        harness.configure_logging(
            loggers={__name__: 'info',
                     'zensols.amr': 'info',
                     'zensols.amr.score': 'debug'})
        logger.info(f'{src} -> {dst}')
        return app.parse_penman(src, dst, keep_keys='snt,id')[0]


if (__name__ == '__main__'):
    CliHarness.add_sys_path(Path('src/python'))
    for config in 'proxy-report lp bio'.split():
        parser = CliAmrParser(config)
        parser.parse('gsii')
        parser.parse('spring')
