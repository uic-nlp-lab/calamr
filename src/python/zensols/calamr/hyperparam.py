"""Optimized hyperparameter updates.

"""
__author__ = 'Paul Landes'

from typing import Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from zensols.config import Dictable
from zensols.datdesc.hyperparam import HyperparamModel


@dataclass
class HyperparameterResource(Dictable):
    hyp: HyperparamModel = field()
    """Initial hyperparameters set by the configuration.

    :see: :obj:`.summary.CapacityCalculator.hyp`

    """
    optimize_file: Path = field()
    """The CSV file with the optimal hyperparameters."""

    criteria: Dict[str, str] = field(default_factory=dict)
    """A dictionary of column/value fields used to reduce in
    :obj:`optimal_hyperparameters`.

    """
    @property
    def optimal_scores(self) -> pd.DataFrame:
        """A dataframe of the results of the performance and hyperparameter
        settings obtained while tuning for combinations of corpora, parser and
        scoring method.

        """
        return pd.read_csv(self.optimize_file)

    @property
    def optimal_hyperparameters(self) -> pd.DataFrame:
        """A dataframe of statistics on the best hyperparameter values obtained
        while tuning for combinations of corpora, parser and scoring method.

        :see: :obj:`criteria`

        """
        init: Dict[str, float] = self.hyp.flatten(deep=True)
        df: pd.DataFrame = pd.read_csv(self.optimize_file)
        for k, v in self.criteria.items():
            df = df[df[k] == v]
        df = df[sorted(set(init.keys()) & set(df.columns))]
        dfs = df.describe().T
        dfs.insert(0, 'name', dfs.index)
        dfs = dfs.reset_index(drop=True)
        dfs = dfs['name min max mean std'.split()]
        dfs['spread'] = dfs['max'] - dfs['min']
        dfs['init'] = dfs['name'].map(lambda n: init[n])
        dfs = dfs.sort_values('std')
        return dfs

    def update(self):
        """Update :obj:`hyp` using the mean selected scores created by
        :obj:`optimal_hyperparameters`.

        """
        df: pd.DataFrame = self.optimal_hyperparameters
        nvs: Dict[str, float] = dict(
            df['name mean'.split()].itertuples(index=False))
        self.hyp.update(nvs)

    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
        return self.hyp._from_dictable(*args, **kwargs)
