"""Installable resource.

"""

from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Sequence, Union
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
import re
import urllib
import shutil
from pathlib import Path
from urllib.parse import ParseResult
import patoolib
from zensols.config import Dictable
from . import InstallError

logger = logging.getLogger(__name__)


@dataclass
class FileSystemUpdateContext(Dictable):
    """The context given to a :class:`.FileSystemUpdate`.

    """
    resource: Resource = field()
    """The :class:`.Resource` that created this context and updating the file
    system.

    """
    check_path: Path = field()
    """The installer relative :obj:`.Resource.check_path`."""

    target: Path = field()
    """The installer relative target path from :class:`.Resource`."""


@dataclass
class FileSystemUpdate(Dictable, metaclass=ABCMeta):
    """A command (GoF pattern) to udpate the file system after a resource has
    decompressed a file.  First experiment with :class:`.ListUpdate`, then find
    the corresponding command with :obj:`dry_run` turned on, then turn it off
    once you've validated its doing the right thing.

    Path fields (i.e. :obj:`.ListUpdate.path`) are formatted with the dictionary
    version of :class:`.FileSystemUpdateContext` and also a ``target`` property
    with the uncompressed path.

    """
    dry_run: bool = field()
    """If ``True`` don't do anything, just act like it."""

    def _format_str(self, context: FileSystemUpdateContext, val: str) -> str:
        return val.format(**context.asdict())

    def _format_path(self, context: FileSystemUpdateContext,
                     attr: str, make_path: bool = True) -> \
            Union[Path, str]:
        val: str = getattr(self, attr)
        path_str: str = self._format_str(context, val)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{attr}({val}) -> {path_str}')
        return Path(path_str) if make_path else path_str

    @abstractmethod
    def invoke(self, context: FileSystemUpdateContext):
        pass


@dataclass
class ListUpdate(FileSystemUpdate):
    """Lists the contents of :obj:`path`.

    """
    path: str = field()
    """A file or directory to list."""

    def invoke(self, context: FileSystemUpdateContext):
        pdir: Path = self._format_path(context, 'path')
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'listing {pdir}')
        for path in pdir.iterdir():
            logger.info(f'list: {path.absolute()}')


@dataclass
class MoveUpdate(FileSystemUpdate):
    """Move file(s) from :obj:`source` to :obj:`target`.

    """
    source: str = field()
    """The source glob (i.e. ``{target}/../*)`` from where to move the files."""

    target: str = field()
    """The target directory where the files end up."""

    def invoke(self, context: FileSystemUpdateContext):
        source: str = self._format_path(context, 'source').resolve().absolute()
        target: Path = self._format_path(context, 'target').resolve().absolute()
        source: Path
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'moving {source} -> {target}')
        for source in source.parent.glob(source.name):
            source = source.resolve().absolute()
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'renaming {source} -> {target}')
            if not self.dry_run:
                shutil.move(source, target)


@dataclass
class RemoveUpdate(FileSystemUpdate):
    """Remove/clean up files to help keep the file system "clean".

    """
    paths: Sequence[str] = field()
    """The list of path formatted files to remove.  For example
    ``{target}/../__MACOSX``.

    """
    def _rm_path(self, path: Path):
        if path.is_dir():
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'removing clean up dir: {path}')
            if not self.dry_run:
                shutil.rmtree(path)
        elif path.is_file():
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'removing clean up file: {path}')
            if not self.dry_run:
                path.unlink()
        elif logger.isEnabledFor(logging.INFO):
            logger.info(f'skipping non-existant clean up dir: {path}')

    def invoke(self, context: FileSystemUpdateContext):
        for path_str in self.paths:
            path = Path(self._format_str(context, path_str))
            path = path.resolve().absolute()
            self._rm_path(path)


@dataclass
class Resource(Dictable):
    """A resource that is installed by downloading from the Internet and then
    optionally uncompressed.  Once the file is downloaded, it is only
    uncompressed if it is an archive file.  This is determined by the file
    extension.

    """
    _DICTABLE_ATTRIBUTES = 'remote_name is_compressed compressed_name'.split()
    _FILE_REGEX = re.compile(r'^(.+)\.(tar\.gz|tgz|tar\.bz2|gz|bz2|' +
                             '|'.join(patoolib.ArchiveFormats) + ')$')
    _NO_FILE_REGEX = re.compile(r'^(?:.+/)?(.+?)\.(.+)?$')

    url: str = field()
    """The URL that locates the file to install."""

    name: str = field(default=None)
    """Used for local file naming."""

    remote_name: str = field(default=None)
    """The name of extracted file (or root directory if a compressed file) after
    being downloaded.  If this isn't set, it is taken from the file name portion
    of the path of the URL.

    """
    is_compressed: bool = field(default=None)
    """Whether or not the file is compressed.  If this isn't set, it is derived
    from the file name.

    """
    rename: bool = field(default=True)
    """If ``True`` then rename the directory to the :obj:`name`."""

    check_path: str = field(default=None)
    """The file to check for existance before doing uncompressing."""

    sub_path: Path = field(default=None)
    """The path to a file in the compressed file after it is extracted.  This is
    only used to obtain the file name in :meth:`get_file_name` when used to
    locate the uncompressed resource file.

    """
    clean_up: bool = field(default=True)
    """Whether or not to remove the downloaded compressed after finished."""

    updates: Sequence[FileSystemUpdate] = field(default=())
    """The file system updates to apply after the file has been decompressed."""

    def __post_init__(self):
        url: ParseResult = urllib.parse.urlparse(self.url)
        remote_path: Path = Path(url.path)
        remote_name: str
        m = self._FILE_REGEX.match(remote_path.name)
        if m is None:
            m = self._NO_FILE_REGEX.match(remote_path.name)
            self._extension = None
            if m is None:
                remote_name = remote_path.name
            else:
                remote_name = m.group(1)
            if self.name is None:
                self.name = remote_path.name
        else:
            remote_name, self._extension = m.groups()
            if self.name is None:
                self.name = remote_name
        if self.remote_name is None:
            self.remote_name = remote_name
        if self.is_compressed is None:
            self.is_compressed = self._extension is not None

    def uncompress(self, path: Path = None, out_dir: Path = None) -> bool:
        """Uncompress the file.

        :param path: the file to uncompress

        :param out_dir: where the uncompressed files are extracted

        """
        uncompressed = False
        if path is None:
            src = Path(self.compressed_name)
            out_dir = Path('.')
        else:
            src = path
            if out_dir is None:
                out_dir = path.parent
        # the target is the name we want after the process completes
        target: Path = out_dir / self.name
        # this is the name of the resulting file of what we expect, or the user
        # can override it if they know what the real resulting file is
        if self.check_path is None:
            check_path = target
        else:
            check_path = out_dir / self.check_path
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'check path: {check_path}')
        # uncompress if we can't find where the output is suppose to go
        if not check_path.exists():
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'uncompressing {src} to {out_dir}')
            out_dir.mkdir(parents=True, exist_ok=True)
            patoolib.extract_archive(str(src), outdir=str(out_dir))
            uncompressed = True
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'rename: {self.rename}, ' +
                         f'path ({check_path}) exists: {check_path.exists()}')
        # the extracted data can either be a file (gz/bz2) or a directory;
        # compare to what we want to rename the target directory
        #
        # note: the check path has to be what extracts as, otherwise it it will
        # unextract it again next time it checks; if the directory extracts as
        # something other than the file name, set both the name and the check
        # path to whatever that path is
        if self.rename and not check_path.exists():
            # the source is where it was extracted
            extracted: Path = out_dir / self.remote_name
            if not extracted.exists():
                raise InstallError(f'Trying to create {check_path} but ' +
                                   f'missing extracted path: {extracted}')
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'renaming {extracted} to {target}')
            extracted.rename(target)
        if self.clean_up:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'cleaning up downloaded file: {src}')
            src.unlink()
        update_context = FileSystemUpdateContext(self, check_path, target)
        update: FileSystemUpdate
        for update in self.updates:
            logger.info(f'updating: {update}')
            update.invoke(update_context)
        return uncompressed

    @property
    def compressed_name(self) -> str:
        """The file name with the extension and used to uncompress.  If the
        resource isn't compressed, just the name is returned.

        """
        if self.is_compressed:
            name = f'{self.name}'
            if self._extension is not None:
                name = f'{name}.{self._extension}'
        else:
            name = self.name
        return name

    def get_file_name(self, compressed: bool = False) -> str:
        """Return the path where a resource is installed.

        :param compressed: if ``True``, return the path where its compressed
                             file (if any) lives

        :return: the path of the resource

        """
        fname = self.compressed_name if compressed else self.name
        if fname is None:
            fname = self.remote_name
        if not compressed and self.sub_path is not None:
            fname = str(Path(fname, self.sub_path))
        return fname
