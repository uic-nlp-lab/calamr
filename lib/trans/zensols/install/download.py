"""Downloads files with a progress bar.

Inspired by `this thread <https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads>`_.

"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging
from pathlib import Path
import urllib.request
from urllib.request import Request
from urllib import parse
from urllib.error import HTTPError
from http.client import HTTPResponse
from ssl import SSLContext
import base64
from tqdm import tqdm
from . import InstallError

logger = logging.getLogger(__name__)


class _DownloadProgressBar(tqdm):
    """Tqdm progress automoation.

    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update((b * bsize) - self.n)


@dataclass
class Downloader(object):
    """A utility class to download a file and (optionally) display a progress
    bar as it downloads.

    """
    DEFAULT_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:101.0) Gecko/20100101 Firefox/101.0'

    use_progress_bar: bool = field(default=True)
    """Whether or not to render the progress bar as the file downloads."""

    skip_if_exists: bool = field(default=True)
    """Skip download if the file exists as the target path."""

    mkdir: bool = field(default=True)
    """Recursively create directories for the target path if they do not already
    exist.

    """
    buffer_size: int = field(default=((2 ** 20) * 10))
    """The size in bytes of the read buffer with a default of 10MB."""

    tqdm_params: Dict[str, Any] = field(
        default_factory=lambda: {'unit': 'B', 'unit_scale': True})
    """Parameters given to :mod:`tqdm` for the progress bar when downloading.

    """
    user_agent: str = field(default=DEFAULT_AGENT)
    """The user agent header used for the file request."""

    def _create_context(self, request: Request) -> Optional[Any]:
        pass

    def _download_result(self, result: Request, output_path: Path,
                         tqdm: tqdm = None):
        with open(output_path, 'wb') as f:
            while True:
                data: bytearray = result.read(self.buffer_size)
                if not data:
                    break
                if tqdm is not None:
                    tqdm.update(len(data))
                f.write(data)

    def _start_download(self, url: str, output_path: Path):
        headers: Dict[str, str] = {}
        if self.user_agent is not None:
            headers['User-Agent'] = self.user_agent
        url_info: parse.Parse = parse.urlparse(url)
        request: Request
        result: HTTPResponse
        context: SSLContext
        try:
            request = Request(url, headers=headers)
            context = self._create_context(request)
        except Exception as e:
            raise InstallError(f"Could not access '{url}' in {self}: {e}", e)
        try:
            result = urllib.request.urlopen(request, context=context)
        except HTTPError as e:
            raise InstallError(f"Could not acceess '{url}: {e}'")
        if self.use_progress_bar and url_info.scheme != 'file':
            flen = result.length
            params = dict(self.tqdm_params)
            params.update({'miniters': 1, 'desc': url.split('/')[-1]})
            if flen is not None and flen > 0:
                params['total'] = flen
            with tqdm(**params) as t:
                self._download_result(result, output_path, t)
        else:
            self._download_result(result, output_path)

    def download(self, url: str, output_path: Path):
        if self.skip_if_exists and output_path.is_file():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'{output_path} is already downloaded')
        else:
            parent = output_path.parent
            if self.mkdir and not parent.is_dir():
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f'creating directory: {parent}')
                parent.mkdir(parents=True)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'downloading {url} to {output_path}')
            self._start_download(url, output_path)


@dataclass
class AuthenticationDownloader(Downloader):
    """Download by first authenticating by login with the server.

    """
    user: str = field(default=None)
    """The user for the authentication."""

    password: str = field(default=None)
    """The password for the authentication."""

    check_certificate: bool = field(default=False)
    """Whether to check the server's certification for validity."""

    def _create_context(self, request: Request) -> Optional[Any]:
        lstr = base64.b64encode(bytes(f'{self.user}:{self.password}', 'ascii'))
        decoded = lstr.decode('utf-8')
        request.add_header('Authorization', f'Basic {decoded}')
        if not self.check_certificate:
            return SSLContext()
