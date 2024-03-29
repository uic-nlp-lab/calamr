"""Populate embeddings generated from :class:`.EmbeddingGenerator`.

:see: :mod:`embedgen`

"""
__author__ = 'Paul Landes'

from typing import ClassVar, Dict, Any, Sequence
from dataclasses import dataclass, field
import logging
from pathlib import Path
import pandas as pd
from torch import Tensor
from zensols.config import Configurable, IniConfig
from zensols.persist import persisted
from zensols.install import Installer
from zensols.deeplearn import TorchConfig
from zensols.deepnlp.embed import WordEmbedModel
from . import (
    BankObjectPopulator, BankObject, Function, Relation,
    Roleset, Role, FunctionPersister,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingPopulator(object):
    """Adds embeddings to certains :class:`.BankObject` instances (see module
    docs).

    """
    META_FILE: ClassVar[str] = 'meta.conf'
    """The name of the metadata file in the distribution file."""

    SENT_TEXT_FILE: ClassVar[str] = 'sentence.csv'
    """The name of the CSV file that has the sentence output with keys."""

    DEFAULT_SECTION: ClassVar[str] = 'propbankdb_default'
    """The default application context section that has the distribution file
    version.

    """
    VERSION_OP: ClassVar[str] = 'version'
    """The model version configuration option and metadata section."""

    EMBEDDING_SECTION: ClassVar[str] = 'embedding'
    """The embedding information section in the distribution metadata config
    file.

    """
    FILE_NAME_OPTION: ClassVar[str] = 'file_name'
    """The option key in the distribution metadata config file."""

    config: Configurable = field()
    """Used to check API with model version."""

    embed_model: WordEmbedModel = field()
    """The embedding model that was created from :class:`.EmbeddingGenerator`
    and used to populate data in :class:`.BankObject` instances.

    """
    function_persister: FunctionPersister = field()
    """The persister used to populate embeddings for :class:`.Function`s."""

    torch_config: TorchConfig = field()
    """Used to copy the embedding matrix to the GPU."""

    roleset_key_pattern: str = field(default=r's~{rs.id.label}')
    """The pattern used format/parse :class:`.Roleset` keys."""

    role_key_pattern: str = field(default=r'r~{rs.id}~{r.index}')
    """The pattern used format/parse :class:`.Role` keys."""

    function_key_pattern: str = field(default=r'f-{f.label}')
    """The pattern used format/parse :class:`.Function` keys."""

    relation_key_pattern: str = field(default=r'e-{r.label}')
    """The pattern used format/parse :class:`.Relation` keys."""

    def __post_init__(self):
        installer: Installer = self.embed_model.installer
        self._meta_file: Path = installer.get_singleton_path() / self.META_FILE
        exists: bool = self._meta_file.is_file()
        if exists:
            logger.info(f'meta file: {self._meta_file}')
            self._load_metadata()
        self._function_embeddings_added = False

    def get_metadata_config(self, add_api: bool = False) -> Configurable:
        """Get the distribution metadata."""
        config = IniConfig(self._meta_file)
        if add_api:
            av: str = self.config.get_option(
                self.VERSION_OP, self.DEFAULT_SECTION)
            mv: str = config.get_option(
                self.VERSION_OP, self.DEFAULT_SECTION)
            sec: str = self.VERSION_OP
            config.set_option('api', av, sec)
            config.set_option('model', mv, sec)
            config.set_option('valid', av == mv, sec)
        return config

    @property
    def is_valid_model_version(self) -> bool:
        """Return whether the API and model versions are in sync."""
        return self.get_metadata_config(True).get_option(
            'valid', self.VERSION_OP)

    def _load_metadata(self):
        """Load the metadata used to configure the :obj:`embed_model`, which
        were the parameters created by the :class:`.EmbeddingGenerator` when the
        distribution file was created.

        """
        config: Configurable = self.get_metadata_config()
        sec: Dict[str, Any] = config.populate({}, self.EMBEDDING_SECTION)
        del sec[self.FILE_NAME_OPTION]
        self.embed_model.__dict__.update(sec)

    def roleset_to_key(self, roleset: Roleset) -> str:
        """Format :class:`.Roleset` keys."""
        return self.roleset_key_pattern.format(rs=roleset)

    def role_to_key(self, roleset: Roleset, role: Role) -> str:
        """Format :class:`.Role` keys."""
        return self.role_key_pattern.format(rs=roleset, r=role)

    def function_to_key(self, func: Function) -> str:
        """Format :class:`.Function` keys."""
        return self.function_key_pattern.format(f=func)

    def relation_to_key(self, rel: Relation) -> str:
        """Format :class:`.Relation` keys."""
        return self.relation_key_pattern.format(r=rel)

    def object_to_key(self, *objs: Sequence[BankObject]) -> str:
        """Generate a key from ``objs`` by calling one of the ``*_to_key``
        methods.

        """
        if len(objs) == 2:
            meth = self.role_to_key
        else:
            meth = {
                Roleset: self.roleset_to_key,
                Function: self.function_to_key,
                Relation: self.relation_to_key,
            }[type(objs[0])]
        return meth(*objs)

    def get_sentence(self, *objs: Sequence[BankObject]) -> str:
        """Get the sentence used to produce the embedding in ``objs`` bank
        objects.

        :see: :meth:`object_to_key`

        """
        installer: Installer = self.embed_model.installer
        path: Path = installer.get_singleton_path() / self.SENT_TEXT_FILE
        key: str = self.object_to_key(*objs)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'looking up sentence {key} in {path}')
        df: pd.DataFrame = pd.read_csv(path)
        return df[df['key'] == key]['sent'].item()

    @persisted('_to_matrix_pw')
    def _get_embedding_matrix(self) -> Tensor:
        return self.embed_model.to_matrix(self.torch_config)

    def _get_embedding(self, key: str) -> Tensor:
        emb: Tensor = self._get_embedding_matrix()
        model: WordEmbedModel = self.embed_model
        eix: int = model.word2idx_or_unk(key)
        return emb[eix]

    def _assert_function_embeddings(self):
        """Populate :class:`.Function` embeddings if not already.  This only has
        to be done once since it uses a GoF flyweight pattern.

        """
        model: WordEmbedModel = self.embed_model
        if not self._function_embeddings_added:
            func: Function
            for func in self.function_persister.get():
                fk: str = self.function_to_key(func)
                # there are no embeddings for descriptions with `none`
                if fk not in model:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"missing embedding '{fk}' for: {func}")
                else:
                    func.embedding: Tensor = self._get_embedding(fk)
            self._function_embeddings_added = True

    def populate_roleset(self, roleset: Roleset):
        """Populate embeddings for all of a role set's object graph.

        :param roleset: the role set with respective
                         :obj:`BaseObject.embedding` fields to be populated

        """
        model: WordEmbedModel = self.embed_model
        rs_key: str = self.roleset_to_key(roleset)
        self._assert_function_embeddings()
        if rs_key in model:
            roleset.embedding: Tensor = self._get_embedding(rs_key)
            role: Role
            for role in roleset.roles:
                r_key: str = self.role_to_key(roleset, role)
                if r_key in model:
                    role.embedding: Tensor = self._get_embedding(r_key)
                else:
                    logger.warning(
                        f'role set in embeddings, but not role: {role}')

    def populate_relation(self, rel: Relation):
        model: WordEmbedModel = self.embed_model
        e_key: str = self.relation_to_key(rel)
        if e_key in model:
            rel.embedding: Tensor = self._get_embedding(e_key)
        else:
            logger.warning(f'missing relation embedding: {rel}')


@dataclass
class EmbeddingBankObjectPopulator(BankObjectPopulator):
    embed_populator: EmbeddingPopulator = field()

    def populate(self, bank_object: BankObject):
        if isinstance(bank_object, Roleset):
            self.embed_populator.populate_roleset(bank_object)
        elif isinstance(bank_object, Relation):
            self.embed_populator.populate_relation(bank_object)
