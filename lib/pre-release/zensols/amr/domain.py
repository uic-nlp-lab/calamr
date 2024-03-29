"""Error and exception classes.

"""
__author__ = 'Paul Landes'

from typing import Optional, Union, List
import logging
from zensols.util import APIError

logger = logging.getLogger(__name__)


class AmrFailure(object):
    """A container class that describes AMR graph creation or handling error.

    """
    def __init__(self, msg: str, sent: str,
                 stack: Optional[Union[str, List[str]]] = None):
        self.msg = msg
        self.sent = sent
        if isinstance(stack, (tuple, list)):
            self.stack = ''.join(stack)
        else:
            self.stack = stack

    def __str__(self):
        return f'{self.msg}: {self.sent}'

    def __repr__(self):
        return self.__str__()


class AmrError(APIError):
    """Raised for package API errors.

    """
    def __init__(self, msg: str, sent: str = None):
        if sent is not None:
            msg = f'{msg}: {sent}'
        super().__init__(msg)
        self.sent = sent
