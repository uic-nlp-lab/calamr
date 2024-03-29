#!/usr/bin/env python

"""Write the score method results for the paper.

"""

from pathlib import Path
from zensols.config import ConfigFactory
from zensols.cli import CliHarness ; CliHarness.add_sys_path(Path('src/python'))
from zensols.calamr import ApplicationFactory
from zensols.calamr.result import ResultAnalyzer


if (__name__ == '__main__'):
    harness: CliHarness = ApplicationFactory.create_harness()
    fac: ConfigFactory = harness.get_config_factory()
    result_analyzer: ResultAnalyzer = fac('calamr_result_analyzer')
    result_analyzer.report()
