"""API to download, uncompress and cache files such as binary models.

"""
__author__ = 'Paul Landes'

from typing import Union, Tuple, Dict, List
from dataclasses import dataclass, field
import logging
import sys
from pathlib import Path
from io import TextIOBase
from frozendict import frozendict
from zensols.util import PackageResource
from zensols.persist import persisted
from zensols.config import Dictable
from zensols.install import Downloader
from . import InstallError, Resource

logger = logging.getLogger(__name__)


@dataclass
class Status(Dictable):
    """Tells of what was installed and how.

    """
    resource: Resource = field()
    """The resource that might have been installed."""

    downloaded_path: Path = field()
    """The path where :obj:`resource` was downloaded, or None if it wasn't
    downloaded.

    """
    target_path: Path = field()
    """Where the resource was installed and/or downloaded on the file system.

    """
    uncompressed: bool = field()
    """Whether or not the resource was uncompressed."""


@dataclass
class Installer(Dictable):
    """Downloads files from the internet and optionally extracts them.

    The files are extracted to either :obj:`base_directory` or a path resolved
    from the home directory with name (i.e. ``~/.cache/zensols/someappname)``.
    If the ``~/.cache`` directory does not yet exist, it will base the installs
    in the home directory per the :obj:`DEFAULT_BASE_DIRECTORIES` attribute.
    Finally, the :obj:`sub_directory` is also added to the path if set.

    Instances of this class are resource path iterable and indexable by name.

    :see: :class:`.Resource`

    """
    DEFAULT_BASE_DIRECTORIES = ('~/.cache', '~/', '/tmp')
    """Contains a list of directories to look as the default base when
    :obj:`base_directory` is not given.

    :see: :obj:`base_directory`

    :see: :obj:`package_resource`

    """
    resources: Tuple[Resource] = field()
    """The list of resources to install and track."""

    package_resource: Union[str, PackageResource] = field(default=None)
    """Package resource (i.e. ``zensols.someappname``).  This field is
    converted to a package if given as a string during post initialization.
    This is used to set :obj:`base_directory` using the package name from the
    home directory if given.  Otherwise, :obj:`base_directory` is used.  One
    must be set.

    """
    base_directory: Path = field(default=None)
    """The directory to base relative resource paths.  If this is not set, then
    this attribute is set from :obj:`package_resource` on initialization.

    :see: :obj:`package_resource`

    :see: :obj:`DEFAULT_BASE_DIRECTORIES`

    """
    sub_directory: Path = field(default=None)
    """A path that is added to :obj:`base_directory` or the path referred by
    :obj:`package_resource` if set.  Setting this is useful to allow for more
    directory structure in the installation (see class docs).  This is only
    useful when using it with :obj:`package_resource`.

    """
    downloader: Downloader = field(default_factory=Downloader)
    """Used to download the file from the Internet."""

    def __post_init__(self):
        if self.package_resource is None and self.base_directory is None:
            raise InstallError(
                'Either package_resource or base_directory must be set')
        if isinstance(self.package_resource, str):
            self.package_resource = PackageResource(self.package_resource)
        if self.base_directory is None:
            self.base_directory = self._get_default_base()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'resolved base directory: {self.base_directory}')
        if self.sub_directory is not None:
            self.base_directory = self.base_directory / self.sub_directory
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'install base directory: {self.base_directory}')

    def _get_default_base(self) -> Path:
        existing = tuple(filter(lambda p: p.is_dir(),
                                map(lambda p: Path(p).expanduser(),
                                    self.DEFAULT_BASE_DIRECTORIES)))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'existing default base directories: {existing}')
        if len(existing) == 0:
            raise InstallError('No default base directories found ' +
                               f'in: {self.DEFAULT_BASE_DIRECTORIES}')
        base: Path = existing[0]
        parts: List[str] = self.package_resource.name.split('.')
        is_home: bool = (base == Path('~/').expanduser())
        if is_home:
            # make a UNIX 'hidden' file if home directory based
            parts[0] = '.' + parts[0]
        pkg_path: Path = Path(*parts)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating base path from home={base}/' +
                         f'sub={self.sub_directory}/pkg_path={pkg_path}')
        return base / pkg_path

    def get_path(self, resource: Resource, compressed: bool = False) -> Path:
        """Return the path where a resource is installed.

        :param resource: the resource to find

        :param compressed: if ``True``, return the path where its compressed
                           file (if any) lives

        :return: the path of the resource

        """
        fname = resource.get_file_name(compressed)
        return self.base_directory / fname

    def get_singleton_path(self, compressed: bool = False) -> Path:
        """Return the path of resource, which is expected to be the only one.

        :param compressed: if ``True``, return the path where its compressed
                           file (if any) lives

        :raises: InstallError if the number of :obj:`resources` length isn't 1

        :return: the resource's path

        """
        rlen = len(self.resources)
        if rlen != 1:
            raise InstallError(
                f'Expecting configured resources to be one, but got {rlen}')
        return self.get_path(self.resources[0], compressed)

    @property
    @persisted('_by_name')
    def by_name(self) -> Dict[str, Resource]:
        """All resources as a dict with keys as their respective names."""
        return frozendict({i.name: i for i in self.resources})

    @property
    @persisted('_paths_by_name')
    def paths_by_name(self) -> Dict[str, Path]:
        """All resource paths as a dict with keys as their respective names."""
        return frozendict({i.name: self.get_path(i) for i in self.resources})

    def _install(self, inst: Resource, dst_path: Path) -> Status:
        uncompressed: bool = False
        downloaded_path: Path = False
        target_path: Path = None
        if inst.is_compressed:
            comp_path = self.get_path(inst, True)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'needs decompression: {comp_path}')
            if not comp_path.is_file():
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'missing compressed file {comp_path}')
                self.downloader.download(inst.url, comp_path)
                downloaded_path = comp_path
            uncompressed = inst.uncompress(comp_path)
            target_path = comp_path
            if uncompressed:
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f'uncompressed to {comp_path}')
        else:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'downloading: {inst.url} -> {dst_path}')
            self.downloader.download(inst.url, dst_path)
            downloaded_path = dst_path
            target_path = dst_path
        return Status(inst, downloaded_path, target_path, uncompressed)

    def install(self) -> List[Status]:
        """Download and install all resources.

        :return: a list of statuses for each resource downloaded

        """
        statuses: List[Status] = []
        res: Resource
        for res in self.resources:
            local_path: Path = self.get_path(res, False)
            check_path: Path = None
            status: Status = None
            if res.check_path is not None:
                check_path = self.base_directory / res.check_path
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'local path: {local_path}, ' +
                             f'check path: {check_path}, ' +
                             f'res check path: {res.check_path}, ' +
                             f'compressed: {res.is_compressed}')
            # we can skip installation if we already find it on the file system;
            # however, we have to re-check compressed files in cases where we've
            # downloaded by not uncompressed between life-cycles (ie raised
            # exceptions)
            if (check_path is not None and check_path.exists()) or \
               (local_path.exists() and not
               (res.is_compressed and res.check_path is not None)):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'found: {local_path}--skipping')
                comp_path = self.get_path(res, True)
                status = Status(res, None, comp_path, False)
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'missing {local_path}')
                status = self._install(res, local_path)
            statuses.append(status)
        return statuses

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        dct = self.asdict()
        del dct['resources']
        self._write_dict(dct, depth, writer)
        self._write_line('resources:', depth, writer)
        res: Resource
        for res in self.resources:
            dct = res.asdict()
            del dct['name']
            self._write_line(res.name, depth + 1, writer)
            self._write_dict(dct, depth + 2, writer)

    def __call__(self) -> List[Status]:
        return self.install()

    def __getitem__(self, resource: Union[str, Resource]):
        if isinstance(resource, str):
            resource = self.by_name[resource]
        return self.get_path(resource)

    def __iter__(self):
        return map(lambda r: self.get_path(r), self.resources)

    def __len__(self):
        return len(self.resources)
