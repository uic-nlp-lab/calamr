"""Hyperparameter metadata: access and documentation.  This package was designed
for the following purposes:

  * Provide a basic scaffolding to update model hyperparameters such as
    :mod:`hyperopt`.

  * Generate LaTeX tables of the hyperparamers and their descriptions for
    academic papers.


The object instance graph hierarchy is:

|  :class:`.HyperparamSet`
|  |
|  +-- :class:`.HyperparamModel`
|      |
|      +--:class:`.Hyperparam`

Access to the hyperparameters is done by calling the *set* or *model* levels
with a *dotted path notation* string.  For example, ``svm.C`` first navigates to
model ``svm``, then to the hyperparameter named ``C``.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    Union, Dict, Any, List, Tuple, ClassVar, Optional,
    Type, Set, Sequence, Iterable
)
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import sys
import re
import textwrap as tw
import copy
from itertools import chain
from pathlib import Path
from io import TextIOBase
from frozendict import frozendict
import yaml
import pandas as pd
from zensols.util import APIError
from zensols.config import Dictable, Configurable
from zensols.persist import persisted
from . import DataDescriptionError, DataFrameDescriber, DataDescriber


class HyperparamError(DataDescriptionError):
    """Raised for any error related hyperparameter access.

    """
    pass


class HyperparamValueError(HyperparamError):
    """Raised for bad values set on a hyperparameter.

    """
    pass


@dataclass(eq=True)
class Hyperparam(Dictable):
    """A hyperparameter's metadata, documentation and value.  The value is
    accessed (retrieval and setting) at runtime.  Do not use this class
    explicitly.  Instead use :class:`.HyperparamModel`.

    The index access only applies when :obj:`type` is ``list`` or ``dict``.
    Otherwise, the :obj:`value` member has the value of the hyperparameter.

    """
    _DICTABLE_WRITE_EXCLUDES: ClassVar[Set[str]] = {'name'}
    _NAME_REGEX = re.compile(r'^[a-zA-Z0-9_]+$')
    _NODE: ClassVar[str] = 'node()'

    CLASS_MAP: ClassVar[Dict[str, Type]] = frozendict({
        'str': str,
        'choice': str,
        'float': float,
        'int': int,
        'bool': bool,
        'list': list,
        'dict': dict})
    """A mapping for values set in :obj:`type` to their Python class
    equivalents.

    """
    VALID_TYPES: ClassVar[str] = frozenset(CLASS_MAP.keys())
    """Valid settings for :obj:`type`."""

    name: str = field()
    """The name of the hyperparameter (i.e. ``C`` or ``learning_rate``)."""

    type: str = field()
    """The type of :obj:`value` (i.e. ``float``, or ``int``)."""

    doc: str = field()
    """The human readable documentation for the hyperparameter.  This is used in
    documentation generation tasks.

    """
    choices: Tuple[str, ...] = field(default=None)
    """When :obj:`type` is ``choice``, the value strings used in :obj:`value`.

    """
    value: Optional[Union[str, float, int, bool, list, dict]] = \
        field(default=None)
    """The value of the hyperparamer used in the application."""

    interval: Union[Tuple[float, float], Tuple[int, int]] = field(default=None)
    """Valid intervals for :obj:`value` as an inclusive interval."""

    def __post_init__(self):
        if self.doc is None:
            raise HyperparamError(f'Missing doc in {self}')
        if self._NAME_REGEX.match(self.name) is None:
            raise HyperparamError(
                ('Illegal name (only letters, numbers and underscores ' +
                 f'allowed): {self.name}'))
        if self.type not in self.VALID_TYPES:
            raise HyperparamError(
                f'Unknown hyperparameter type: {self.type} in {self}')
        if self.choices is None:
            if self.type == 'choice':
                raise HyperparamError(
                    f'No choice values given for choice: {self}')

    @property
    def _value(self) -> Optional[Union[str, float, int, bool, list, dict]]:
        return self._value_val

    @_value.setter
    def _value(self, val: Optional[Union[str, float, int, bool, list, dict]]):
        cls: Type = type(val)
        tcls: Type = self.cls
        if val is not None:
            if cls != tcls:
                raise HyperparamValueError(
                    f"Wrong type '{cls.__name__}', expecting " +
                    f"'{tcls.__name__}' for hyperparameter '{self.name}'")
            if self.type == 'choice' and val not in self.choices:
                choices: str = ', '.join(map(lambda s: f"'{s}'", self.choices))
                raise HyperparamValueError(
                    f"Unknown choice '{val}', expecting one of: {choices}")
            if self.interval is not None:
                if val > self.interval[1] or val < self.interval[0]:
                    raise HyperparamValueError(
                        f"Out of range value '{val}' not in " +
                        f'[{self.interval[0]}, {self.interval[1]}]')
        self._value_val = val

    def _resolve(self, path: List[str], val: Any, set_val: bool = False) -> Any:
        if len(path) == 0:
            if set_val:
                self.value = val
            return self.value
        if len(path) != 1:
            raise HyperparamError(f'Wrong parameter path length: {path}')
        path_item: str = path[0]
        if path_item == self._NODE:
            if set_val:
                self.value = val
            return self
        if self.type == 'list':
            try:
                idx: int = int(path_item)
                if set_val:
                    self.value[idx] = val
                return self.value[idx]
            except ValueError as e:
                raise HyperparamError(
                    f"List indices must be integers, not '{path_item}'") from e
        elif self.type == 'dict':
            if set_val:
                self.value[path_item] = val
            return self.value[path_item]
        else:
            raise HyperparamError(
                f"Trying to index '{self.type}' with path: {path}")

    def write_sphinx(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        phead: str = f':param {self.name}: '
        pdoc: str = self.doc.lower()
        type_str: str = f':type {self.name}: {self.get_type_str(short=False)}'
        if pdoc[-1] == '.':
            pdoc = pdoc[:-1]
        if self.interval is not None:
            pdoc = f'{pdoc}, must be in the interval {self.interval_str}'
        text_wrapper = tw.TextWrapper(
            initial_indent=self._sp(depth),
            subsequent_indent=(self._sp(depth) +
                               self._get_str_space(len(phead))))
        self._write_block(text_wrapper.wrap(phead + pdoc), 0, writer)
        self._write_block(type_str, depth, writer)

    def __getitem__(self, index) -> Any:
        if self.type == 'list':
            return self.value[index]
        elif self.type == 'dict':
            return self.value[index]
        else:
            raise HyperparamError(f"'{self}' is not subscriptable")

    @property
    def cls(self) -> Type:
        """The Python equivalent class of :obj:`type`."""
        return self.CLASS_MAP[self.type]

    def get_type_str(self, short: bool = True) -> str:
        if self.type == 'choice':
            if short:
                return 'str <' + '|'.join(self.choices) + '>'
            else:
                return 'str; one of: ' + ', '.join(self.choices)
        else:
            return self.type

    @property
    def interval_str(self) -> str:
        return f'[{self.interval[0]}, {self.interval[1]}]'

    def __str__(self) -> str:
        s = f'({self.get_type_str()})'
        if self.value is not None:
            s = f'{self.value} {s}'
        if self.interval is not None:
            s = f'{s} in {self.interval_str}'
        return self.name + ': ' + s


Hyperparam.value = Hyperparam._value


@dataclass
class HyperparamContainer(Dictable, metaclass=ABCMeta):
    """A container class for :class:`.Hyperparam` instances.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    @abstractmethod
    def _resolve(self, path: List[str], val: Any, set_val: bool) -> Any:
        """Drill down to the hyperparameter navigating through
        :class:`.HyperparamModel` and :class:`.HyperparamSet` using the dotted
        path notation (see module docs).

        """
        pass

    @abstractmethod
    def flatten(self, deep: bool = False) -> Dict[str, Any]:
        """Return a flattened directory with the dotted path notation (see
        module docs).

        :param deep: if ``True`` recurse in to :class:`dict` and :class:`list`
                     hyperparameter values

        """
        pass

    @abstractmethod
    def write_sphinx(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        """Write Sphinx autodoc used in a class as
        :class:`dataclasses.dataclass` field.

        """
        pass

    def __call__(self, path: str) -> Any:
        """Return the value of the parameter.

        :param path: a dotted path notation (see module docs)

        """
        return self._resolve(path.split('.'), None, False)

    def update(self, params: Union[Dict[str, Any], HyperparamContainer]):
        """Update parameter values.

        :param params: a dict of dotted path notation keys

        """
        if isinstance(params, HyperparamContainer):
            self.update(params.flatten())
        else:
            for k, v in params.items():
                self._resolve(k.split('.'), v, True)


@dataclass(eq=True)
class HyperparamModel(HyperparamContainer):
    """The model level class that contains the parameters.  This class
    represents a machine learning model such as a SVM with hyperparameters such
    as ``C`` and ``maximum iterations``.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    name: str = field()
    """The name of the model (i.e. ``svm``).  This name can have only
    alpha-numeric and underscore charaters.

    """
    doc: str = field()
    """The human readable documentation for the model."""

    desc: str = field(default=None)
    """The description the model used in the documentation when obj:`name` is
    not sufficient.  Since :obj:`name` has naming constraints, this can be used
    as in place during documentation generation.

    """
    params: Dict[str, Hyperparam] = field(default_factory=dict)
    """The hyperparameters keyed by their names."""

    table: Optional[Dict[str, Any]] = field(default=None, repr=False)
    """Overriding data used when creating a :class:`.Table` from
    :meth:`.DataFrameDescriber.create_table`.

    """
    def __post_init__(self):
        if self.doc is None:
            raise HyperparamError(f'Missing doc in {self}')
        if self.desc is None:
            self.desc = self.name

    def _resolve(self, path: List[str], val: Any, set_val: bool) -> Any:
        if len(path) == 0:
            return self
        if len(path) == 1 and path[0] == Hyperparam._NODE:
            return self
        param: Hyperparam = self.params[path[0]]
        return param._resolve(path[1:], val, set_val)

    @classmethod
    def _flatten(cls, n: Any, p: Tuple[str], col: Dict[str, Any]):
        if isinstance(n, Hyperparam):
            np = n.name if p is None else f'{p}.{n.name}'
            cls._flatten(n.value, n.name, col)
        elif isinstance(n, Dict):
            for k, v in n.items():
                np = k if p is None else f'{p}.{k}'
                cls._flatten(v, np, col)
        elif isinstance(n, (tuple, list)):
            for i, v in enumerate(n):
                np = str(i) if p is None else f'{p}.{i}'
                col[np] = v
        else:
            col[p] = n

    def flatten(self, deep: bool = False) -> Dict[str, Any]:
        def map_param(p: Hyperparam) -> Dict[str, Any]:
            val: Any = p.value
            if p.type == 'dict' or p.type == 'list':
                val = copy.deepcopy(val)
            return (p.name, val)

        if deep:
            col: Dict[str, Any] = {}
            self._flatten(self.params, None, col)
            return col
        else:
            return dict(map(map_param, self.params.values()))

    @property
    def values_dataframe(self) -> pd.DataFrame:
        """A dataframe with parameter data.  This includes the name, type, value
        and documentation.

        """
        def map_row(p: Hyperparam) -> Dict[str, Any]:
            dct: Dict[str, Any] = p.asdict()
            dct['type'] = p.get_type_str()
            del dct['choices']
            del dct['interval']
            return dct

        return pd.DataFrame(tuple(map(map_row, self.params.values())))

    @property
    def metadata_dataframe(self) -> pd.DataFrame:
        """A dataframe describing the :obj:`values_dataframe`."""
        df = pd.DataFrame(
            ['name', 'data type', 'documentation', 'settings'],
            index='name type doc value'.split(),
            columns='description'.split())
        df.index.name = 'name'
        return df

    def clone(self) -> HyperparamModel:
        """Make a copy of this instance."""
        params = copy.deepcopy(self.__dict__)
        return self.__class__(**params)

    def create_dataframe_describer(self) -> DataFrameDescriber:
        """Return an object with metadata fully describing the hyperparameters
        of this model.

        """
        return DataFrameDescriber(
            name=self.desc,
            df=self.values_dataframe,
            desc=self.doc,
            meta=self.metadata_dataframe)

    def write_sphinx(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        doc: str = self.doc
        doc = '"""' + self.doc[0].capitalize() + self.doc[1:]
        if doc[-1] != '.':
            doc += '.'
        self._write_line(f'{self.name}: {self.__class__.__name__} = field()',
                         depth, writer)
        self._write_wrap(doc, depth, writer)
        self._write_empty(writer)
        self._write_line('Hyperparameters::', depth, writer)
        self._write_empty(writer)
        param: Hyperparam
        for i, param in enumerate(self.params.values()):
            if i > 0:
                self._write_empty(writer)
            param.write_sphinx(depth + 1, writer)
        self._write_line('"""', depth, writer)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_doc: bool = False):
        self._write_line(f'{self.desc}:', depth, writer)
        if include_doc:
            self._write_line(f'doc: {self.doc}', depth + 1, writer)
            self._write_line('params:', depth + 1, writer)
        for name, param in sorted(self.params.items(), key=lambda x: x[0]):
            if include_doc:
                self._write_line(f'{name}:', depth + 2, writer)
                self._write_object(param, depth + 3, writer)
            else:
                if isinstance(param.value, dict):
                    self._write_line(param.name, depth + 1, writer)
                    self._write_object(param.value, depth + 2, writer)
                else:
                    self._write_line(str(param), depth + 1, writer)

    def __getitem__(self, name: str):
        return self.params[name]

    def get(self, name: str) -> HyperparamModel:
        return self.params.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self.params

    def __getattr__(self, attr: str, default: Any = None) -> Any:
        val: Optional[Hyperparam] = self.params.get(attr)
        if val is not None:
            return val.value
        return super().__getattribute__(attr)

    def __setattr__(self, attr: str, value: Any = None):
        if attr in {'params', 'name', 'doc', 'desc', 'table'}:
            super().__setattr__(attr, value)
        else:
            val: Optional[Hyperparam] = self.params.get(attr)
            if val is not None:
                val.value = value
            else:
                super().__setattr__(attr, value)

    def __len__(self) -> int:
        return len(self.params)

    def __str__(self) -> str:
        s: str = self.name
        if s != self.desc:
            s += f' ({self.desc})'
        return s


@dataclass(eq=True)
class HyperparamSet(HyperparamContainer):
    """The top level in the object graph hierarchy (see module docs).  This
    contains a set of models and typically where calls by packages such as
    :mod:`hyperopt` are used to update the hyperparameters of the model(s).

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    models: Dict[str, HyperparamModel] = field(default_factory=dict)
    """The models containing hyperparameters for this set."""

    name: Optional[str] = field(default=None)
    """The name fo the hyperparameter set."""

    def _resolve(self, path: List[str], val: Any, set_val: bool) -> Any:
        if len(path) == 0:
            raise HyperparamError(f'Missing model in path: {path}')
        model: HyperparamModel = self.models[path[0]]
        return model._resolve(path[1:], val, set_val)

    def flatten(self, deep: bool = False) -> Dict[str, Any]:
        def map_model(m: HyperparamModel):
            return map(lambda mt: (f'{m.name}.{mt[0]}', mt[1]),
                       m.flatten(deep).items())

        return dict(chain.from_iterable(map(map_model, self.models.values())))

    def create_describer(self, meta_path: Path = None) -> DataDescriber:
        """Return an object with metadata fully describing the hyperparameters
        of this model.

        :param meta_path: if provided, set the path on the returned instance

        """
        def map_model(m: HyperparamModel) -> DataFrameDescriber:
            dd = m.create_dataframe_describer()
            if meta_path is not None:
                dd.meta_path = meta_path / f'{self.name}.csv'
            return dd

        return DataDescriber(
            describers=tuple(map(map_model, self.models.values())))

    def write_sphinx(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        if self.name is not None:
            cname: str = self.name.capitalize() + 'Hyperparams'
            models: str = ', '.join(
                map(lambda m: f'``{m.desc}``', self.models.values()))
            doc: str = f'"""Hyperparaeters for models {models}.'
            mod: str = re.sub(r'^(.+)\..+$', '\\1', __name__)
            self._write_line('from dataclasses import dataclass, field',
                             depth, writer)
            self._write_line(f'from {mod} import HyperparamModel',
                             depth, writer)
            self._write_empty(writer, 2)
            self._write_line('@dataclass', depth, writer)
            self._write_line(f'class {cname}(object):', depth, writer)
            self._write_line(doc, depth + 1, writer)
            self._write_empty(writer)
            self._write_line('"""', depth + 1, writer)
        model: HyperparamModel
        for i, model in enumerate(self.models.values()):
            if i > 0:
                self._write_empty(writer)
            model.write_sphinx(depth + 1, writer)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_doc: bool = False):
        self._write_line('models:', depth, writer)
        for name, model in self.models.items():
            model.write(depth + 1, writer, include_doc=include_doc)

    def __getitem__(self, name: str):
        return self.models[name]

    def get(self, name: str) -> HyperparamModel:
        return self.models.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self.models

    def __getattr__(self, attr: str, default: Any = None) -> Any:
        val: Any = self.get(attr)
        if val is not None:
            return val
        return super().__getattribute__(attr)

    def __len__(self) -> int:
        return len(self.models)


@dataclass
class HyperparamSetLoader(object):
    """Loads a set of hyperparameters from a YAML :class:`pathlib.Path`,
    :class:`dict` or stream :class:`io.TextIOBase`.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    _FILE_NAME_REGEX: ClassVar[re.Match] = re.compile(r'^(.+)-hyperparam$')
    """The regular expression used to match hyperparameter YAML definition
    files.

    """
    data: Union[Dict[str, Any], Path, TextIOBase] = field()
    """The source of data to load, which is a YAML :class:`pathlib.Path`,
    :class:`dict` or stream :class:`io.TextIOBase`.

    :see: :obj:`updates`

    """
    config: Configurable = field(default=None)
    """The application configuration used to update the hyperparameters from
    other sections.

    """
    updates: Sequence[Dict[str, Any]] = field(default=())
    """A sequence of dictionaries with keys as :class:`.HyperparamModel` names
    and values as sections with values to set after loading using :obj:`data`.

    """
    def _from_param(self, name: str, param: Dict[str, Any]):
        if 'name' not in param:
            param['name'] = name
        return Hyperparam(**param)

    def _from_model(self, name: str, model: Dict[str, Any]):
        params: Dict[str, Any] = model['params']
        return HyperparamModel(
            name=name,
            desc=model.get('desc'),
            doc=model.get('doc'),
            params=dict(map(lambda p: (p[0], self._from_param(*p)),
                            params.items())),
            table=model.get('table'))

    def _from_dict(self, data: Dict[str, Any],
                   name: str = None) -> HyperparamSet:
        return HyperparamSet(
            name=name,
            models=dict(map(lambda m: (m[0], self._from_model(*m)),
                            data.items())))

    def _from_stream(self, stream: TextIOBase,
                     name: str = None) -> HyperparamSet:
        return self._from_dict(yaml.load(stream, yaml.FullLoader), name=name)

    def _from_path(self, path: Path) -> HyperparamSet:
        name: str = path.stem
        m: re.Match = self._FILE_NAME_REGEX.match(name)
        if m is not None:
            name = m.group(1)
        with open(path) as f:
            return self._from_stream(f, name=name)

    def _get_updates(self) -> Iterable[Dict[str, Any]]:
        param_update: Dict[str, Any]
        for param_update in self.updates:
            for k, v in param_update.items():
                settings: Dict[str, Any] = dict(self.config[v])
                yield dict(map(lambda t: (f'{k}.{t[0]}', t[1]),
                               settings.items()))

    @persisted('_load')
    def load(self) -> HyperparamSet:
        """Load and return the hyperparameter object graph from :obj:`data`.

        """
        hs: HyperparamSet
        if isinstance(self.data, Path):
            hs = self._from_path(self.data)
        elif isinstance(self.data, Dict):
            hs = self._from_dict(self.data)
        elif isinstance(self.data, TextIOBase):
            hs = self._from_stream(self.data)
        else:
            raise APIError(f'Unknown input type: {type(self.data)}')
        if self.config is not None:
            update: Dict[str, Any]
            for update in self._get_updates():
                hs.update(update)
        return hs

    def __call__(self, path: str = None) -> \
            Union[HyperparamSet, HyperparamSet, Hyperparam]:
        """Calls "meth:`load`.

        :param path: if provided, use as the dot separated path in to
                     :class:`.HyperparamSet`; the first level will be an
                     instance of :class:`.HyperparmModel`

        """
        hset: HyperparamSet = self.load()
        if path is not None:
            return hset(path)
        else:
            return hset

    def __getitem__(self, path: str) -> \
            Union[HyperparamSet, HyperparamSet, Hyperparam]:
        return self(path)
