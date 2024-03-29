"""An optimizer that uses a :class:`~zensols.nlp.score.Scorer` as an objective.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass
from abc import abstractmethod
import pandas as pd
from zensols.nlp.score import ScoreContext, ScoreResult, ScoreSet, Scorer
from .opt import HyperparameterOptimizer


@dataclass
class ScoringHyperparameterOptimizer(HyperparameterOptimizer):
    """An optimizer that uses a :class:`~zensols.nlp.score.Scorer` as the
    objective and a means to determine the loss.  The default loss function is
    defined as 1 - F1 using the ``f1_score``
    :meth:`~zensols.nlp.score.ScoreResult.as_dataframe` column.

    """
    @abstractmethod
    def _get_next_score_context(self) -> ScoreContext:
        pass

    def _get_scorer(self) -> Scorer:
        return self.config_factory('nlp_scorer')

    @property
    def scorer(self) -> Scorer:
        return self._get_scorer()

    def _get_loss(self, df: pd.DataFrame) -> float:
        return float(1 - df['f_score'].mean())

    def _get_score_dataframe(self, score_set: ScoreSet) -> pd.DataFrame:
        return score_set.as_dataframe()

    def _objective(self) -> Tuple[float, pd.DataFrame]:
        sctx: ScoreContext = self._get_next_score_context()
        res: ScoreResult = self.scorer.score(sctx)
        df: pd.DataFrame = self._get_score_dataframe(res)
        loss: float = self._get_loss(df)
        return loss, df
