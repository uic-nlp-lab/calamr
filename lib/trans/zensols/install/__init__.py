from zensols.util import APIError


class InstallError(APIError):
    """Raised for issues while downloading or installing files."""


from .download import *
from .resource import *
from .installer import Status, Installer
