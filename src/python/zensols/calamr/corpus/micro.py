"""Micro summarization corpus.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
from penman.surface import Alignment, RoleAlignment
from zensols.amr import AmrSentence, AmrDocument, CorpusWriter


@dataclass
class MicroSummaryCorpusWriter(CorpusWriter):
    input_file: Path = field(default=None)

    def _remove_alignments(self, sent: AmrSentence):
        epis: Dict[Tuple[str, str, str], List] = sent.graph.epidata
        epi: List
        for epi in epis.values():
            rms = []
            for i, x in enumerate(epi):
                if isinstance(x, (Alignment, RoleAlignment)):
                    rms.append(i)
            rms.reverse()
            for i in rms:
                del epi[i]
        metadata = sent.metadata
        del metadata['alignments']
        sent.metadata = metadata

    def add_from_file(self, input_file: Path):
        """Read raw document text and write the micro-summary corpus file.

        :param input_file: the JSON file to read the doc text
        """
        with open(input_file) as f:
            doc_strs: List[Dict[str, str]] = json.load(f)
        for ds in doc_strs:
            body: AmrDocument = self._parse(ds['body'])
            summary: AmrDocument = self._parse(ds['summary'])
            for sent in body:
                sent.set_metadata('snt-type', 'body')
            for sent in summary:
                sent.set_metadata('snt-type', 'summary')
            sents: List[AmrSentence] = list(summary.sents)
            sents.extend(body.sents)
            for sent in sents:
                self._remove_alignments(sent)
            doc: AmrDocument = AmrDocument(sents=sents)
            self.docs.append(doc)

    def __call__(self):
        self.add_from_file(self.input_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        super().__call__()
