# CALAMR: Component ALignment for Abstract Meaning Representation

This repository contains code and data the paper [CALAMR: Component ALignment
for Abstract Meaning Representation].  This code is used to align the
components of a bipartite source and summary AMR graph.  The results are useful
as a semantic graph similarity score (like SMATCH) or to find the summarized
portion (as AMR nodes, edges and subgraphs) of a document or the portion of the
source that represents the summary.


## Inclusion in Your Projects

The purpose of this repository is to reproduce the results in the paper.  If
want to align AMR graphs for your own work, please refer to the
[zensols.calamr] repository, which has reusable code and examples.  If you use
this library or the [PropBank API] or PropBank curated database, please
[cite](#citation) our paper.


## Reproducing the Results

To reproduce the results from the paper, first process the corpus.  These next
steps create the document summarization and parser metrics.

Preprocessing the corpus with the following steps:

1. Install a Python 3.10.8 virtual environment on Linux.  Note this version of
   the code assumes Linux, but new version does not.
1. Clone this repository: `git clone https://github.com/uic-nlp-lab/calamr`
1. Enter the repository and create release directory that corpora to be
   installed: `cd calamr && mkdir download`
1. Download the [AMR Release 3.0]:
   `cp .../path/to/download/amr_annotation_3.0_LDC2020T02.tgz download`
1. For reproducing the results that compare with earlier work on the AMR
   Release 1.0 corpus, place that corpus file in `download` directory as well.
1. Install the environment: `./bin/install.sh <path to Python home directory>`.
   If you use conda, create a new conda 3.10.8 environment and set it to the
   Python home directory it creates (not including the `bin/python3` directory)
1. Check the previous step to make sure it successfully creates new Python
   environment in directory `pyenv`.  Also make sure it clones the `amr_coref`
   repository, and applies the patch successfully.
1. Create the sentence type/align merged corpus file:
   `./bin/prep.sh mergeanons`
1. Create the mismatch corpus (please contact the authors for the original
   corpus file used in the experiments as the random seed was not set):
   `./bin/prep.sh mismatchcorp`
1. Create the parser output of the corpora: `./bin/prep.sh parsecorp`
1. Create the JAMR output for the corpora.  This is a manual process, which
   includes downloading and installing the JAMR parser.  We created this file
   manually, but will provide it for requests that include proof of purchase of
   the [AMR Release 3.0] corpus.
1. Score documents and pairs (document table): `./bin/prep.sh score`
1. Align documents: `./bin/prep.sh align`
1. Output alignment statistics: `./bin/prep.sh alignstats`


## To recreate the example diagrams from the paper

The micro corpus are short examples for illustrating the alignment algorithm
via component diagram.  You can add your own sentences to the [AMR parser
input](corpus/amr-micro-summary.json) and rerun the micro corpus create an
align steps below.

Follow the steps to creating the virtual environment in [result
reproduction](#reproducing-the-results) section, and then:
1. Create the AMR micro corpus: `./bin/micro.sh createcorp`
1. Align the micro corpus graphs: `./bin/micro.sh align`


## Attribution

This project, or reference model code, uses:

* Python 3.10
* [amrlib] for AMR parsing.
* [amr_coref] for AMR co-reference
* [zensols.amr] for AMR features and summarization data structures.
* [Sentence-BERT] embeddings
* [zensols.propbankdb] and [zensols.deepnlp] for PropBank embeddings
* [zensols.nlparse] for natural language features and [NLP scoring]
* [Smatch] (Cai and Knight. 2013) and [WLK] (Opitz et. al. 2021) for scoring.


## Citation

If you use this project in your research please use the following BibTeX entry:

```bibtex
@inproceedings{landes-di-eugenio-2024-calamr-component,
    title = "{CALAMR}: Component {AL}ignment for {A}bstract {M}eaning {R}epresentation",
    author = "Landes, Paul  and
      Di Eugenio, Barbara",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italy",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.236",
    pages = "2622--2637"
}
```


## License

[MIT License](LICENSE.md)

Copyright (c) 2023 - 2024 Paul Landes


<!-- links -->
[AMR Release 3.0]: https://catalog.ldc.upenn.edu/LDC2020T02
[zensols.calamr]: https://github.com/plandes/calamr
[zensols.propbankdb]: https://github.com/plandes/propbankdb
[PropBank API]: https://github.com/plandes/propbankdb

[CALAMR: Component ALignment for Abstract Meaning Representation]: https://example.com

[amrlib]: https://github.com/bjascob/amrlib
[amr_coref]: https://github.com/bjascob/amr_coref
[spaCy]: https://spacy.io
[Smatch]: https://github.com/snowblink14/smatch
[WLK]: https://github.com/flipz357/weisfeiler-leman-amr-metrics
[zensols.nlparse]: https://github.com/plandes/nlparse
[NLP scoring]: https://plandes.github.io/nlparse/api/zensols.nlp.html#zensols-nlp-score
[Sentence-BERT]: https://www.sbert.net
[zensols.amr]: https://github.com/plandes/amr
[zensols.deepnlp]: https://github.com/plandes/deepnlp
[zensols.propbankdb]: https://github.com/plandes/propbankdb
