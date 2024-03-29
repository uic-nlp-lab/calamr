#!/usr/bin/env python

from pathlib import Path
from zensols import deepnlp

# initialize the NLP system
deepnlp.init()

if 0:
    import zensols.deepnlp.transformer as tran
    tran.turn_off_huggingface_downloads()


if (__name__ == '__main__'):
    from zensols.cli import ConfigurationImporterCliHarness
    conf: str = {
        0: 'micro',
        1: 'lp',
        2: 'proxy-report',
        3: 'proxy-report-1_0',
        4: 'mismatch-proxy',
        5: 'parser-proxy',
        6: 'bio',
    }[0]
    harness = ConfigurationImporterCliHarness(
        root_dir=Path.cwd(),
        src_dir_name='src/python',
        app_factory_class='zensols.calamr.ApplicationFactory',
        config_path=f'etc/{conf}.conf',
        proto_args='proto',
        proto_factory_kwargs={'reload_pattern': r'^zensols\.calamr\.(?!flow|domain|corpus\.proxyreport)'},
    )
    r = harness.run()
