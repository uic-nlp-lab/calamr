"""Weisfeiler-Leman Graph Kernels for AMR Graph Similarity

Juri Opitz, Angel Daza, and Anette Frank. 2021. Weisfeiler-Leman in the Bamboo:
Novel AMR Graph Metrics and a Benchmark for AMR Graph Similarity. Transactions
of the Association for Computational Linguistics, 9:1425–1441.

:link: `GitHub <https://github.com/flipz357/weisfeiler-leman-amr-metrics>`_

"""
from typing import Iterable, List, Dict, Any
from dataclasses import dataclass, field
import logging
import numpy as np
from zensols.nlp.score import ScoreMethod, ScoreContext, FloatScore
from .graph_helpers import GraphParser
from .amr_similarity import WLK
from zensols.nlp.score import ErrorScore
from zensols.amr import AmrFeatureSentence

logger = logging.getLogger(__name__)


@dataclass
class WeisfeilerLemanKernelScoreCalculator(ScoreMethod):
    """Computes the Weisfeiler-Leman Kernel scores of AMR sentences (see module
    docs).

    """
    params: Dict[str, Any] = field(
        default_factory=lambda: dict(iters=2, communication_direction='both'))
    """Parameters given to the implementation of the Weisfeiler-Leman Kernel
    scoring method :class:`.WLK` class.

    """
    def _score(self, meth: str, context: ScoreContext) -> Iterable[FloatScore]:
        def map_sent(s: AmrFeatureSentence) -> str:
            assert isinstance(s, AmrFeatureSentence)
            return s.amr.graph_single_line

        grapa = GraphParser(input_format='penman',
                            edge_to_node_transform=False)
        predictor = WLK(**self.params)
        for s1, s2 in context.pairs:
            try:
                gs1: str = map_sent(s1)
                gs2: str = map_sent(s2)
                g1s, _ = grapa.parse([gs1])
                g2s, _ = grapa.parse([gs2])
                scores: List[np.float64] = predictor.predict(g1s, g2s)
                yield FloatScore(float(scores[0]))
            except Exception as e:
                logger.error('could not score <{g1}>::<{g2}>: {e}', e)
                yield ErrorScore(meth, e, FloatScore.NAN_INSTANCE)
