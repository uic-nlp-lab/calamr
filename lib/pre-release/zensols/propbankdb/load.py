"""Parses frameset XML files from the file system.

"""
__author__ = 'Paul Landes'

from typing import List, Iterable, Dict, Tuple, Set, Any, Optional
from dataclasses import dataclass, field
import dataclasses as dc
import logging
import sys
from pathlib import Path
import itertools as it
from itertools import chain
import re
import pandas as pd
from lxml import etree
from lxml.etree import _ElementTree as ElementRoot
from lxml.etree import _Element as Element
from penman.models.amr import model as amr_model
from zensols.util import time
from zensols.persist import persisted
from zensols.install import Installer
from . import (
    BankParseError, RoleResource, PartOfSpeech, Function, Reification, Relation,
    PropBankArgumentSpan, PropBank, Example, RoleLink, Role, Alias, RolesetId,
    Roleset, Predicate, Frameset, Database,
)

logger = logging.getLogger(__name__)


@dataclass
class _XmlResource(object):
    """Parse the XML and provides utility xpath look up methods.

    """
    path: Path = field()
    """The XML file to parse."""

    @property
    @persisted('_root')
    def root(self) -> ElementRoot:
        """The root element of the XML file."""
        if self.path.suffix.endswith('html'):
            parser = etree.HTMLParser()
            root = etree.parse(self.path, parser)
        else:
            root = etree.parse(self.path)
        return root

    def nodes(self, xpath: str, root: Element = None,
              expect: bool = True) -> List[Element]:
        """Return nodes found at an XPath.

        :param xpath: the location of the dnoees to return

        :param root the root element, or the parsed XML root if not provided

        :param expect: if ``True`` raise a ``BankParseError`` when no nodes are
                       found at ``xpath``

        """
        root = self.root if root is None else root
        nodes: List[Element] = root.xpath(xpath)
        if len(nodes) == 0:
            if expect:
                if 0:
                    print(etree.tostring(root))
                raise BankParseError(
                    f'Expecting more than one element at {xpath}: {self.path}')
        return nodes

    def node(self, xpath: str, root: Element = None,
             expect: bool = True) -> Element:
        """Get a single node like in meth:`nodes` but expect a singleton."""
        nodes: List[Element] = self.nodes(xpath, root, expect)
        if len(nodes) == 0 and not expect:
            return
        elif len(nodes) != 1:
            raise BankParseError(
                f'Expecting exactly one element at {xpath}, ' +
                f'but got: {len(nodes)}: {self.path}')
        return nodes[0]

    def text(self, xpath: str, root: Element = None,
             expect: bool = True) -> str:
        """Like :meth:`node` but expect a text node and return its text."""
        elem: Element = self.node(xpath, root, expect)
        if elem is not None:
            # strip added white space, otherwise mulitple DOM functions added
            return str(elem).strip()

    def asdict(self, root: Element) -> Dict[str, Any]:
        """Convert an element to nested dictionary structure.

        :link: `Attrib: <https://gist.github.com/ErDmKo/010849e53fac8e0071a5>`_

        """
        def child_as_json(children: List[Element]) -> Dict[str, Any]:
            out = {}
            for child in list(children):
                if len(list(child)):
                    if child.tag not in out:
                        out[child.tag] = []
                    out[child.tag].append(child_as_json(child))
                else:
                    out[child.tag] = child.text
            return out

        return child_as_json(root)


@dataclass
class _BankResource(_XmlResource):
    """Parses a Frameset XML file.

    """
    functions: Dict[str, Function] = field()
    """Role functions taken from in the frameset DTD from the AMR 3.0 corpus."""

    pos_by_value: Dict[str, PartOfSpeech] = field(
        default_factory=lambda: {p.value: p for p in PartOfSpeech})

    def __post_init__(self):
        self._func_count = max(map(lambda f: f.uid, self.functions.values()))

    def _role_link(self, elem: Element) -> RoleLink:
        return RoleLink(
            cls=self.text('@class', elem),
            resource=RoleResource[self.text('@resource', elem).lower()],
            version=self.text('@version', elem),
            name=self.text('./text()', elem))

    def _role(self, elem: Element) -> Role:
        func_key: str = self.text('@f', elem).upper()
        func_key = '-' if len(func_key) == 0 else func_key
        func: Function = self.functions.get(func_key)
        if func is None:
            logger.warning(f"No function '{func_key}' in {self.path}")
            self._func_count += 1
            func = Function(uid=self._func_count, label=func_key)
            self.functions[func_key] = func
        return Role(
            description=self.text('@descr', elem),
            function=func,
            index=self.text('@n', elem),
            role_links=tuple(map(
                self._role_link,
                self.nodes('rolelinks/rolelink', elem, expect=False))))

    def _alias(self, elem: Element) -> Alias:
        return Alias(
            part_of_speech=self.pos_by_value.get(
                self.text('@pos', elem),
                PartOfSpeech.unknown),
            word=self.text('./text()', elem))

    @staticmethod
    def _map_index(s: str) -> int:
        return PropBank.UNKONWN_INDEX if s == '?' else int(s)

    def _propbank_args(self, elem: Element) -> PropBankArgumentSpan:
        return PropBankArgumentSpan(
            type=self.text('@type', elem),
            span=(self._map_index(self.text('@start', elem)),
                  self._map_index(self.text('@end', elem))),
            token=self.text('text()', elem, expect=False))

    def _propbank(self, elem: Element):
        if len(elem.xpath('*[count(child::*) = 0]')) == 0:
            logger.warning(f'empty example propbank in {self.path}')
        else:
            arg_elems: List[Element] = elem.xpath('arg')
            rel_str: str = self.text('rel/text()', elem, False)
            args: Tuple[PropBankArgumentSpan]
            rels: List[str]
            if len(arg_elems) == 0:
                logger.info(f'no arguments found in example in {self.path}')
                args = ()
            else:
                args = tuple(map(self._propbank_args, arg_elems))
            if rel_str is None:
                logger.info(f'no relative tokens in example in {self.path}')
                rel_tokens = ()
            else:
                rel_tokens = rel_str.split()
            return PropBank(
                relative_indicies=tuple(
                    map(self._map_index,
                        self.text('rel/@relloc', elem).split())),
                relative_tokens=rel_tokens,
                argument_spans=args)

    def _example(self, elem: Element) -> Example:
        pbs: List[Element] = elem.xpath('propbank')
        if len(pbs) == 0:
            logger.info(f'no examples found in {self.path}')
        else:
            assert len(pbs) == 1
            pb: Element = pbs[0]
            return Example(
                name=self.text('@name', elem),
                source=self.text('@src', elem),
                text=self.text('text/text()', elem),
                propbank=self._propbank(pb))

    def _roleset(self, elem: Element) -> Roleset:
        examples: List[Element] = elem.xpath('example')
        if len(examples) > 0:
            examples = map(self._example, self.nodes('example', elem))
            examples = tuple(filter(lambda x: x is not None, examples))
        else:
            examples = ()
        return Roleset(
            id=RolesetId(self.text('@id', elem)),
            name=self.text('@name', elem),
            aliases=tuple(map(self._alias, self.nodes('aliases/alias', elem))),
            roles=tuple(map(self._role, self.node('roles', elem))),
            examples=examples)

    def _predicate(self, elem: Element) -> Predicate:
        return Predicate(
            lemma=self.text('@lemma', elem),
            rolesets=tuple(map(self._roleset, self.nodes('roleset', elem))))

    def _frameset(self, elem: Element) -> Frameset:
        return Frameset(
            path=self.path,
            predicates=tuple(map(self._predicate,
                                 self.nodes('predicate', elem))))

    def parse(self) -> Frameset:
        return self._frameset(self.node('/frameset'))


@dataclass
class FramesetParser(object):
    """Parses all frameset XML files on the file system.

    """
    installer: Installer = field()
    """An installer that points to the GitHub repo where the probank framesets
    are kept.  This points to the source zip file on GitHub.

    """
    function_path: Path = field()
    """A CSV file that has all role functions and their descriptions."""

    frames_dir: Path = field()
    """The relative path from where the GitHub propbank files are downloaded and
    uncompressed.

    """
    @property
    def xml_dir(self) -> Path:
        """The directory that has the frameset XML files."""
        self.installer()
        return self.installer.get_singleton_path() / self.frames_dir

    @property
    @persisted('_function_df')
    def function_df(self) -> Dict[str, Function]:
        """Keys are function labels/names, values are the function instances."""
        df: pd.DataFrame = pd.read_csv(self.function_path)
        return {f.label: f for f in map(lambda r: Function(*r),
                                        df.itertuples(name=None))}

    def _parse_frameset(self, xml_file: Path) -> Frameset:
        """Parse a frameset XML file in to an instance."""
        func_df: Dict[str, Function] = self.function_df
        try:
            res = _BankResource(xml_file, func_df)
            return res.parse()
        except BankParseError as e:
            raise e
        except Exception as e:
            raise BankParseError(f'Could not parse {xml_file}: {e}') from e

    def _parse_framesets(self) -> List[Frameset]:
        role_ids: Dict[RolesetId, Path] = {}
        xml_files: Iterable[Path] = filter(
            lambda p: p.suffix == '.xml', self.xml_dir.iterdir())
        xml_files: Path
        for xml_file in xml_files:
            frameset: Frameset = self._parse_frameset(xml_file)
            p: Predicate
            for p in frameset.predicates:
                rs: Roleset
                for rs in p.rolesets:
                    if rs.id in role_ids:
                        logger.warning(
                            f'duplicate role set ID {rs.id} ' +
                            f'found in {role_ids[rs.id]} and {xml_file}')
                    role_ids[rs.id] = xml_file
            yield frameset

    def __call__(self, limit: int = sys.maxsize) -> Tuple[Frameset]:
        """Parse all frameset XML files in to an instances."""
        with time('parsed {cnt} frame sets'):
            framesets: Tuple[Frameset] = tuple(it.islice(
                self._parse_framesets(), limit))
            cnt: int = len(framesets)
            return framesets


@dataclass
class RelationLoader(object):
    """Loads relations from the :mod:`penman` package.

    """
    description_mods: Tuple[Tuple[re.Pattern, str]] = field()
    """A tuple of regular expressions and replacement strings for relation
    descriptions.

    """
    def __post_init__(self):
        self.description_mods = tuple(
            map(lambda x: (re.compile(x[0]), x[1]), self.description_mods))

    def _create_regexes(self) -> Tuple[str, re.Pattern]:
        """Create tuples of role regular expressions and their type.

        """
        def order_role_regex(tups) -> int:
            top: Set[str] = {':ARG[0-9]', ':op[0-9]+', ':mod',
                             ':name', ':time', ':location'}
            return chain.from_iterable(
                [filter(lambda t: t[0] in top, tups),
                 filter(lambda t: t[0] not in top, tups)])

        return tuple(map(lambda i: (i[0], f'^{i[0]}', i[1]['type']),
                         order_role_regex(amr_model.roles.items())))

    def _get_description(self, s: str) -> str:
        regex: re.Pattern
        repl: str
        prev: str = None
        for regex, repl in self.description_mods:
            s = regex.sub(repl, s)
            if prev is not None and s != prev:
                break
            prev = s
        s = s.replace('-', ' ')
        return s

    def __iter__(self) -> Iterable[Relation]:
        reifications: Dict[str, Tuple[str, str, str]] = amr_model.reifications
        literal: str
        regex: re.Pattern
        type: str
        for label, regex, type in self._create_regexes():
            reif_tup: Tuple = reifications.get(label)
            reif: Reification = None
            name: str = label[1:]
            descr: str = self._get_description(name)
            lab_trunc: int = label.find('[')
            if lab_trunc > -1:
                label = label[:lab_trunc]
            label = label[1:]
            if type != 'general' and descr.find(type) == -1:
                descr += f' {type}'
            if reif_tup is not None:
                args = []
                for a in reif_tup[0][1:]:
                    m: re.Pattern = Relation.REGEX.match(a)
                    assert m.group(1) == 'ARG'
                    args.append(int(m.group(2)))
                reif = Reification(reif_tup[0][0], *args)
            yield Relation(
                label=label,
                type=type,
                description=descr,
                regex=regex,
                reification=reif)


@dataclass
class DatabaseLoader(object):
    """Loads the parsed frameset XML files in to an SQLite database.

    """
    parser: FramesetParser = field()
    """Parses all frameset XML files on the file system, which is then used to
    load the database.

    """
    relation_loader: RelationLoader = field()
    """Loads AMR role relations."""

    db: Database = field()
    """A data access object to the frameset database."""

    frameset_limit: int = field(default=sys.maxsize)
    """The max number of framesets to load."""

    @persisted('_frames')
    def _get_frames(self) -> Tuple[Frameset]:
        return self.parser(self.frameset_limit)

    def _enums(self):
        """Load :class:`.RoleResource`s in in to the database.

        """
        for name, en in (('insert_roleres', RoleResource),
                         ('insert_pos', PartOfSpeech)):
            for i in en:
                self.db.frameset_persister.execute_no_read(
                    name, params=(i.value, i.name))

    def _functions(self):
        """Load :class:`.Function`s in in to the database."""
        funcs: Set[Function] = set()
        fs: Frameset
        for fs in self._get_frames():
            p: Predicate
            for p in fs.predicates:
                rs: Roleset
                for rs in p.rolesets:
                    role: Role
                    for role in rs.roles:
                        funcs.add(role.function)
        for func in funcs:
            self.db.function_persister.insert_row(*dc.astuple(func))

    def _relations(self):
        rel: Relation
        for rel in self.relation_loader:
            params: List = list(dc.astuple(rel)[1:-1])
            if rel.reification is not None:
                params.append(str(dc.astuple(rel.reification)))
            else:
                params.append(None)
            self.db.relation_persister.insert_row(*params)

    def _framesets(self):
        """Load framesets in to the database."""
        fs: Frameset
        for fs in self._get_frames():
            fs_id: int = self.db.frameset_persister.insert_row(fs.path.name)
            p: Predicate
            for p in fs.predicates:
                p_id: int = self.db.predicate_persister.insert_row(
                    fs_id, p.lemma)
                rs: Roleset
                for rs in p.rolesets:
                    rs_id: int = self.db.roleset_persister.insert_row(
                        p_id, str(rs.id), rs.name)
                    al: Alias
                    for al in rs.aliases:
                        self.db.alias_persister.insert_row(
                            rs_id, al.part_of_speech.value, al.word)
                    ex: Example
                    for ex in rs.examples:
                        pb: Optional[PropBank] = ex.propbank
                        self.db.example_persister.insert_row(
                            rs_id, ex.name, ex.source, ex.text,
                            None if pb is None else pb.asjson())
                    role: Role
                    for role in rs.roles:
                        r_id: int = self.db.role_persister.insert_row(
                            rs_id, role.description, role.function.uid,
                            role.index)
                        rl: RoleLink
                        for rl in role.role_links:
                            self.db.role_link_persister.insert_row(
                                r_id, rl.cls, rl.resource.value,
                                rl.version, rl.name)

    def clear(self):
        """Drop the database, which deletes the SQLite file."""
        self.db.frameset_persister.conn_manager.drop()

    def __call__(self):
        """Load the database from the the XML files on disk.

        """
        framesets = self._get_frames()
        with time(f'loaded {len(framesets)} frame sets in to database'):
            self.clear()
            self._enums()
            self._functions()
            self._relations()
            self._framesets()
