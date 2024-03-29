"""Package resources for distribution.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import logging
import shutil
from pathlib import Path
from zensols.cli import CliHarness
from zensols.hostcon import ApplicationFactory, Application

logger = logging.getLogger(__name__)


@dataclass
class Packager(object):
    """Packages the staged files in to the deployment file.

    """
    stage_dir: Path = field()
    """The directory to where the staged files to be zipped live."""

    archive_dir: Path = field()
    """The directory where the deployment file is created."""

    remote_dir: str = field()
    """The directory on the remote server where the file goes."""

    def pack(self):
        logger.info(f'packaging: {self.stage_dir} -> {self.archive_dir}')
        shutil.make_archive(self.archive_dir, 'zip', self.stage_dir)

    def deploy(self):
        deploy_dir: str = str(self.archive_dir.parent)
        remote_dir: str = self.remote_dir
        harness: CliHarness = ApplicationFactory.create_harness()
        app: Application = harness.get_instance()
        logger.info(f'deploying: {deploy_dir} -> {remote_dir}')
        app.push(remote_dir=remote_dir, local_dir=str(deploy_dir) + '/')
