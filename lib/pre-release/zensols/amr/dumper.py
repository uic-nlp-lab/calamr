"""Plot AMR graphs.

"""
__author__ = 'Paul Landes'

from typing import List, Union, Tuple, Dict
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
import textwrap as tw
import shutil
from pathlib import Path
from amrlib.graph_processing.amr_plot import AMRPlot
from zensols.config import Dictable
from . import AmrError, AmrSentence, AmrDocument

logger = logging.getLogger(__name__)


class _AMRPlot(AMRPlot):
    # Instances are nodes (circles with info) ie.. concept relations
    def _add_instance(self, t):
        label = str(t.source) + '/' + str(t.target)
        # change to oval shape
        self.graph.node(t.source, label=label, shape='oval')

    # Edges are lines connecting nodes
    def _add_edge(self, t):
        # add space so text doesn't overlap edges
        self.graph.edge(t.source, t.target, label='  ' + t.role)


@dataclass
class Dumper(Dictable, metaclass=ABCMeta):
    """Plots and writes AMR content in human readable formats.

    """
    target_dir: Path = field()
    """The path where the file ends up; this defaults to the text of the
    sentence with an extension (i.e. ``.pdf``).

    """
    add_doc_dir: bool = field(default=True)
    """Whether to write files to a directory for the document."""

    write_text: bool = field(default=True)
    """Whether to write the sentence text in to the generated diagram."""

    sent_file_format: str = field(default='{sent.short_name}')
    """The file format for sentence files when not supplied."""

    add_description: bool = field(default=True)
    """Whether to the description."""

    front_text: str = field(default=None)
    """Text to add in the description."""

    width: int = field(default=79)
    """The width of the text when rendering the graph."""

    overwrite_dir: bool = field(default=False)
    """Whether to remove the output directories."""

    overwrite_sent_file: bool = field(default=False)
    """Whether to remove the output sentence files."""

    @abstractmethod
    def _create_paths(self, target_name: str) -> Tuple[Path, Path]:
        """Return paths based on a target name.

        :param target_name: what will become the stem portion of the output file

        :return: two paths:

            * the temporary file generated to later be deleted
            * the output file name with the plot

        """
        pass

    @abstractmethod
    def _render_sent(self, sent: AmrSentence, path: Path) -> Path:
        """Render a sentence as a plot.

        :param sent: the sentence graph to render

        :param path: where to output the graph as a file (i.e. a PDF)

        """
        pass

    def clean(self) -> bool:
        """Remove the output directory if it exists.

        :return: whether the output directory existed

        """
        if self.target_dir.is_dir():
            if self.target_dir.absolute() == Path('.').absolute():
                raise AmrError('Refusing to remove the current directory')
            logger.info(f'removing {self.target_dir}')
            shutil.rmtree(self.target_dir)
            return True
        return False

    def _maybe_remove_target(self):
        if self.overwrite_dir:
            self.clean()

    def plot_sent(self, sent: AmrSentence,
                  target_name: str = None) -> List[Path]:
        """Create a plot of the AMR graph visually.  The file is generated with
        graphviz in a temporary space, then moved to the target path.

        :param target_name: the file name added to :object:`target_dir`, or if
                            ``None``, computed from the sentence text

        :return: the path(s) where the file(s) were generated

        """
        if target_name is None:
            target_name = self.sent_file_format.format(sent=sent)
        else:
            target_name = str(Path(target_name).stem)
        written_paths: List[Path] = []
        temp: Path
        out_file: Path
        temp_file, out_file = self._create_paths(target_name)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating plot in {out_file}')
        self._maybe_remove_target()
        if out_file.exists() and not self.overwrite_sent_file:
            raise AmrError(f'Plot already exists: {out_file}')
        src = self._render_sent(sent, temp_file)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'removing digraph tmp file: {temp_file}')
        temp_file.unlink()
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'wrote: {src}')
        written_paths.append(src)
        if self.write_text:
            text_graph_path: Path = self.target_dir / f'{target_name}.txt'
            with open(text_graph_path, 'w') as f:
                if self.front_text is not None:
                    f.write(f'# ::note {self.front_text}\n')
                sent.write(writer=f)
            written_paths.append(text_graph_path)
        return written_paths

    def plot_doc(self, doc: AmrDocument,
                 target_name: str = None) -> List[Path]:
        """Create a plot for each AMR sentence as a graph.  The file is
        generated with graphviz in a temporary space, then moved to the target
        directory.

        If the directory doesn't exist, it is created.

        :return: the path(s) where the file(s) were generated

        """
        gen_paths: List[Path] = []
        prev_overwrite_dir: bool = self.overwrite_dir
        self._maybe_remove_target()
        try:
            self.overwrite_dir = False
            sent: AmrSentence
            for sent in doc.sents:
                gen_paths.extend(self.plot_sent(sent, target_name))
        finally:
            self.overwrite_dir = prev_overwrite_dir
        return gen_paths

    def dump_sent(self, sent: AmrSentence) -> List[Path]:
        """Dump the contents of the sentence."""
        return self.plot_sent(sent)

    def dump_doc(self, doc: AmrDocument) -> List[Path]:
        """Dump the contents of the document to a directory.  This includes the
        plots and graph strings of all sentence.  This also includes a
        ``doc.txt`` file that has the graph strings and their sentence index.

        :param doc: the document to plot

        :return: the paths to each file that was generated

        """
        paths: List[Path] = self.plot_doc(doc)
        gen_paths: List[Path] = []
        if len(paths) > 0:
            first: Path = paths[0]
            doc_path: Path
            if self.add_doc_dir:
                doc_path = Path(first.parent / first.stem)
                doc_path.mkdir(parents=True, exist_ok=True)
            else:
                doc_path = Path('.')
            for path in paths:
                dst = doc_path / path.name
                shutil.move(path, dst)
                gen_paths.append(path)
            if self.write_text:
                doc_seq_path: Path = doc_path / 'doc.txt'
                with open(doc_seq_path, 'w') as f:
                    doc.write(writer=f, add_sent_id=True)
                logger.info(f'wrote: {doc_seq_path}')
                gen_paths.append(doc_seq_path)
        return gen_paths

    def render(self, cont: Union[AmrDocument, AmrSentence],
               target_name: str = None) -> List[Path]:
        """Create a PDF for an AMR document or sentence as a graph.  The file is
        generated with graphviz in a temporary space, then moved to the target
        directory.

        :see: :meth:`dump_sent`

        :see: :meth:`dump_doc`

        :return: the path(s) where the file(s) were generated

        """
        if isinstance(cont, AmrSentence):
            return self.dump_sent(cont)
        elif isinstance(cont, AmrDocument):
            return self.dump_doc(cont)
        else:
            raise AmrError(f'Unknown AMR type: {type(cont)}')

    def __call__(self, cont: Union[AmrDocument, AmrSentence],
                 target_name: str = None) -> List[Path]:
        """See :meth:`render`."""
        return self.render(cont, target_name)


@dataclass
class GraphvizDumper(Dumper):
    """Dumps plots created by graphviz using the ``dot`` program.

    """
    extension: str = field(default='pdf')
    attribs: Dict[str, str] = field(default_factory=lambda: dict(rankdir='TB'))

    def _create_paths(self, target_name: str) -> Tuple[Path, Path]:
        temp_digraph_path: Path = self.target_dir / target_name
        out_file = Path(str(temp_digraph_path) + '.' + self.extension)
        return temp_digraph_path, out_file

    def _render_sent(self, sent: AmrSentence, path: Path) -> Path:
        graph_str: str = sent.graph_string
        plot = _AMRPlot(render_fn=str(path), format=self.extension)
        if self.add_description:
            text: str = sent.text
            if self.front_text is not None:
                text = self.front_text + ' ' + text
            text = '\n'.join(tw.wrap(text, width=self.width))
            plot.graph.attr(label=text, labelloc='top', labeljust='left')
        plot.graph.attr(**self.attribs)
        plot.build_from_graph(graph_str, debug=False)
        return Path(plot.render())
