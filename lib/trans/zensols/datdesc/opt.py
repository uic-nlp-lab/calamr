"""Contains container and utility classes for hyperparameter optimization.
These classes find optimial hyperparamters for a model and save the results as
JSON files.  This module is meant to be used by command line applications
configured as Zensols `Resource libraries`_.

:see: `Resource libraries <https://plandes.github.io/util/doc/config.html#resource-libraries>`_

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, Dict, Any, ClassVar, Type, List, Iterable
from dataclasses import dataclass, field
from abc import abstractmethod, ABCMeta
import logging
import sys
import itertools as it
import json
from json import JSONDecodeError
from pathlib import Path
from io import TextIOBase
import shutil
import pandas as pd
import hyperopt as ho
from hyperopt import hp
from zensols.persist import persisted, PersistedWork
from zensols.config import ConfigFactory, Dictable
from . import HyperparamModel, HyperparamSet, HyperparamSetLoader
from .hyperparam import HyperparamError, Hyperparam

logger = logging.getLogger(__name__)


@dataclass
class HyperparamResult(Dictable):
    """Results of an optimization and optionally the best fit.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    name: str = field()
    """The name of the of :class:`.HyperparameterOptimizer`, which is the
    directory name.

    """
    hyp: HyperparamModel = field()
    """The updated hyperparameters."""

    scores: pd.DataFrame = field()
    """The last score results computed during the optimization."""

    loss: float = field()
    """The last loss."""

    eval_ix: int = field()
    """The index of the optimiation."""

    @classmethod
    def from_file(cls: Type, path: Path) -> HyperparamResult:
        """Restore a result from a file name.

        :param path: the path from which to restore

        """
        with open(path) as f:
            data: Dict[str, Any] = json.load(f)
        model_name: str = data['hyp']['name']
        hyp_set: HyperparamSet = HyperparamSetLoader(
            {model_name: data['hyp']}).load()
        hyp_model: HyperparamModel = hyp_set[model_name]
        return cls(
            name=model_name,
            hyp=hyp_model,
            scores=pd.read_json(json.dumps(data['scores'])),
            loss=data['loss'],
            eval_ix=data['eval_ix'])

    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
        dct: Dict[str, Any] = super()._from_dictable(*args, **kwargs)
        dct['scores'] = self.scores.to_dict()
        return dct

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        dct: Dict[str, Any] = self.asdict()
        del dct['scores']
        del dct['hyp']
        self._write_dict(dct, depth, writer)
        self._write_line('hyp:', depth, writer)
        self._write_object(self.hyp, depth + 1, writer)
        self._write_line('scores:', depth, writer)
        df = self.scores
        if len(df) == 1:
            df = df.T
        with pd.option_context('display.max_colwidth', self.WRITABLE_MAX_COL):
            self._write_block(repr(df), depth + 1, writer)


@dataclass
class HyperparamRun(Dictable):
    """A container for the entire optimization run.  The best run contains the
    best fit (:obj:`.HyperparamResult`) as predicted by the hyperparameter
    optimization algorithm.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True
    _DICTABLE_ATTRIBUTES: ClassVar[List[str]] = [
        'initial_loss', 'loss_stats', 'best']

    runs: Tuple[Tuple[Path, HyperparamResult]] = field(repr=False)
    """The results from previous runs."""

    @property
    def final_path(self) -> Path:
        """The path of the final run."""
        return self.runs[-1][0]

    @property
    def final(self) -> HyperparamResult:
        """The results of the final run, which as the best fit (see class
        docs).

        """
        return self.runs[-1][1]

    @property
    def initial_loss(self) -> float:
        """The loss from the first run."""
        return self.runs[0][1].loss

    @property
    def losses(self) -> Tuple[float]:
        """The loss value for all runs"""
        return tuple(map(lambda r: r[1].loss, self.runs))

    @property
    def loss_stats(self) -> Dict[str, float]:
        """The loss statistics (min, max, ave, etc)."""
        df = pd.DataFrame(self.losses, columns=['loss'])
        # skip initial row
        df = df.iloc[1:]
        stats = df.describe().to_dict()
        return stats['loss']

    @property
    def best_result(self) -> HyperparamResult:
        """The result that had the lowest loss."""
        runs: List[HyperparamRun] = list(map(lambda r: r[1], self.runs))
        runs.sort(key=lambda r: r.loss)
        return runs[0]

    @classmethod
    def from_dir(cls: Type, path: Path) -> HyperparamRun:
        """Return an instance with the runs stored in directory ``path``.

        """
        def read_result(path: Path):
            try:
                return HyperparamResult.from_file(path)
            except JSONDecodeError as e:
                raise HyperparamError(f'Could not parse {path}: {e}') from e

        files: List[Path] = sorted(path.iterdir(), key=lambda p: p.stem)
        return cls(runs=tuple(map(lambda p: (p, read_result(p)), files)))


@dataclass
class CompareResult(Dictable):
    """Contains the loss and scores of an initial run and a run found on the
    optimal hyperparameters.

    """
    initial_param: Dict[str, Any] = field()
    """The initial hyperparameters."""

    initial_loss: float = field()
    """The initial loss."""

    initial_scores: pd.DataFrame = field()
    """The initial scores."""

    best_eval_ix: int = field()
    """The optimized hyperparameters."""

    best_param: Dict[str, Any] = field()
    """The optimized hyperparameters."""

    best_loss: float = field()
    """The optimized loss."""

    best_scores: pd.DataFrame = field()
    """The optimized scores."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('initial:', depth, writer)
        self._write_line('parameters:', depth + 1, writer)
        self._write_dict(self.initial_param, depth + 2, writer)
        self._write_line(f'loss: {self.initial_loss}', depth + 1, writer)
        self._write_line('scores:', depth + 1, writer)
        df = self.initial_scores
        if len(df) == 1:
            df = df.T
        self._write_block(df.to_string(), depth + 2, writer)

        self._write_line('best:', depth, writer)
        self._write_line(f'eval_ix: {self.best_eval_ix}', depth + 1, writer)
        self._write_line('parameters:', depth + 1, writer)
        self._write_dict(self.best_param, depth + 2, writer)
        self._write_line(f'loss: {self.best_loss}', depth + 1, writer)
        self._write_line('scores:', depth + 1, writer)
        df = self.best_scores
        if len(df) == 1:
            df = df.T
        self._write_block(df.to_string(), depth + 2, writer)


@dataclass
class HyperparameterOptimizer(object, metaclass=ABCMeta):
    """Creates the files used to score optimizer output.

    """
    name: str = field(default='default')
    """The name of the optimization experiment set.  This has a bearing on where
    files are stored (see :obj:`opt_intermediate_dir`).

    """
    hyperparam_names: Tuple[str, ...] = field(default=())
    """The name of the hyperparameters to use to create the space.

    :see: :meth:`_create_space`

    """
    max_evals: int = field(default=1)
    """The maximum number of evaluations of the hyperparmater optimization
    algorithm to execute.

    """
    show_progressbar: bool = field(default=True)
    """Whether or not to show the progress bar while running the optimization.

    """
    intermediate_dir: Path = field(default=Path('opthyper'))
    """The directory where the intermediate results are saved while the
    algorithm works.

    """
    baseline_path: Path = field(default=None)
    """A JSON file with hyperparameter settings to set on start.  This file
    contains the output portion of the ``final.json`` results, (which are the
    results parsed and set in :obj:`HyperparamResult`).

    """
    def __post_init__(self):
        self._last_result: HyperparamResult = None
        self._eval_ix = 0
        self._config_factory = PersistedWork(
            '_config_factory', self, cache_global=True)

    @abstractmethod
    def _create_config_factory(self) -> ConfigFactory:
        """Create the harness for the Zensols application."""
        pass

    @abstractmethod
    def _get_hyperparams(self) -> HyperparamModel:
        """Return the hyperparameter instance for the application."""
        pass

    @abstractmethod
    def _objective(self) -> Tuple[float, pd.DataFrame]:
        """The objective implementation used by this class.

        :return: a tuple of the (``loss``, ``scores``), where the scores are any
                 dataframe of the scores of the evaulation

        """
        pass

    def _create_space(self) -> Dict[str, float]:
        """Create the hyperparamter spacy used by the :mod:`hyperopt` optimizer.

        :see: :obj:`hyperparam_names`

        """
        if len(self.hyperparam_names) == 0:
            raise HyperparamError(
                'No given hyperparamter names to optimizer create space')
        model: HyperparamModel = self.hyperparams
        space: Dict[str, Any] = {}
        name: str
        for name in self.hyperparam_names:
            param: Hyperparam = model[name]
            if param.type == 'float':
                if param.interval is None:
                    raise HyperparamError(f'No interval for parameter {param}')
                space[name] = hp.uniform(name, *param.interval)
            elif param.type == 'int':
                if param.interval is None:
                    raise HyperparamError(f'No interval for parameter {param}')
                space[name] = hp.uniformint(name, *param.interval)
            elif param.type == 'choice':
                space[name] = hp.choice(name, param.choices)
            else:
                raise HyperparamError(
                    f'Unsupported parameter type: {param.type}')
        return space

    def _compare(self) -> Tuple[float, pd.DataFrame]:
        """Like :meth:`_objective` but used when comparing the initial
        hyperparameters with the optimized.

        """
        return self._objective()

    def _get_score_iterations(self) -> int:
        """Return the number of scored items (times to call :meth:`_optimize`)
        called by :meth:`get_score_dataframe`.

        """
        return 1

    def _get_result_file_name(self, name: str = None) -> str:
        """Return a file name based ``name`` used for storing results."""
        if name is None:
            name = f'{self._eval_ix}.json' if name is None else f'{name}.json'
        else:
            name = f'{name}.json'
        return name

    @property
    @persisted('_config_factory')
    def config_factory(self) -> ConfigFactory:
        """The app config factory."""
        return self._create_config_factory()

    @property
    def hyperparams(self) -> HyperparamModel:
        """The model hyperparameters to be updated by the optimizer."""
        return self._get_hyperparams()

    @property
    def results_intermediate_dir(self) -> Path:
        """The directory that has all intermediate results by subdirectory
        name.

        """
        return self.intermediate_dir / 'tmp'

    @property
    def opt_intermediate_dir(self) -> Path:
        """The optimization result directory for the config/parser.

        """
        return self.results_intermediate_dir / self.name

    def remove_result(self):
        """Remove an entire run's previous optimization results."""
        to_del: Path = self.opt_intermediate_dir
        if to_del.is_dir():
            logger.warning(f'deleting: {to_del}')
            shutil.rmtree(to_del)

    def _persist_result(self, name: str = None):
        """Write the last result to the file system in JSON format."""
        name: str = self._get_result_file_name(name)
        res_path: Path = self.opt_intermediate_dir / name
        res_path.parent.mkdir(parents=True, exist_ok=True)
        with open(res_path, 'w') as f:
            self._last_result.asjson(writer=f, indent=4)

    def _run_objective(self, space: Dict[str, Any] = None) -> float:
        hp: HyperparamModel = self.hyperparams
        if space is not None:
            hp.update(space)
        loss: float
        scores: pd.DataFrame
        loss, scores = self._objective()
        self._last_result = HyperparamResult(
            name=self.name,
            hyp=hp,
            scores=scores,
            loss=loss,
            eval_ix=self._eval_ix)
        self._persist_result()
        self._eval_ix += 1
        return loss

    def _create_uniform_space(self, params: Tuple[str, float, float],
                              integer: bool = False) -> \
            Dict[str, float]:
        """Create a uniform space used by the optimizer.

        :param params: a tuple of tuples with the form
                       ``(<param name>, <start range>, <end range>)``

        :param integer: whether the uniform range are of type integer

        """
        def map_param(name: str, start: float, end: float) -> Tuple[Any, ...]:
            if integer:
                return (name, hp.uniformint(name, start, end))
            else:
                return (name, hp.uniform(name, start, end))

        return dict(map(lambda x: map_param(*x), params))

    def _create_choice(self, params: Tuple[str, Tuple[str, ...]]):
        """Create a choice space.

        :param params: a tuple of tuples with the form
                       ``(<param name>, (<choice 1>, <choice 2>...))``
        """
        def map_param(name: str, choices: Tuple[Any, ...]) -> Tuple[Any, ...]:
            return (name, hp.choice(name, choices))

        return dict(map(lambda x: map_param(*x), params))

    def _minimize_objective(self) -> Dict[str, float]:
        """Run the hyperparameter optimization process and return the results as
        a dict of the optimized parameters.

        """
        search_space: Dict[str, float] = self._create_space()
        if logger.isEnabledFor(logging.INFO):
            logger.info('starting hyperparameter minimization objective')
        return ho.fmin(
            fn=self._run_objective,
            show_progressbar=self.show_progressbar,
            space=search_space,
            algo=ho.tpe.suggest,
            max_evals=self.max_evals)

    def _finish_optimize(self, best: Dict[str, float]):
        """Called by :meth:`optimize` when complete.  Command line programs will
        probably want to report the hyperparameters and last score values
        computed during the hyperparameter optimization using
        :meth:`write_best_result`.

        """
        pass

    def optimize(self):
        """Run the optimization algorithm.

        """
        self.remove_result()
        self._objective()
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f'initial loss: {self._last_result.loss}')
        best: Dict[str, float] = self._minimize_objective()
        self._finish_optimize(best)

    def get_run(self, result_dir: Path = None) -> HyperparamRun:
        """Get the best run from the file system.

        :param result_dir: the result directory, which defaults to
                           :obj:`opt_intermediate_dir`

        """
        if result_dir is None:
            result_dir = self.opt_intermediate_dir
        return HyperparamRun.from_dir(result_dir)

    def get_best_result(self) -> HyperparamResult:
        return self.get_run().best_result

    def get_best_results(self) -> Dict[str, HyperparamResult]:
        """Return the best results across all hyperparameter optimization runs
        with keys as run names.

        """
        res_dirs: Iterable[Path] = self.results_intermediate_dir.iterdir()
        res_dirs = filter(lambda p: p.is_dir(), res_dirs)
        best_results: Dict[str, HyperparamResult] = {}
        res_dir: Path
        for res_dir in res_dirs:
            run: HyperparamRun = self.get_run(res_dir)
            res: HyperparamResult = run.best_result
            best_results[res_dir.name] = res
        return best_results

    def _get_baseline(self, set_hyp: bool = True) -> HyperparamResult:
        """Get the baseline hyperparamters from a file if specified in
        :obj:`baseline_path` if set, otherwise get it from the file system.

        """
        res: HyperparamResult
        if self.baseline_path is None:
            logger.info('restoring from best run')
            res: HyperparamResult = self.get_run().best_result
        else:
            logger.info(f'restoring from {self.baseline_path}')
            if self.baseline_path.is_dir():
                # assume a previous run when a directorty
                run = HyperparamRun.from_dir(self.baseline_path)
                res = run.best_result
            else:
                try:
                    res = HyperparamResult.from_file(self.baseline_path)
                except KeyError:
                    # when a flat parameter file, there will be no `hyp` key
                    logger.info(
                        f'baseline file {self.baseline_path} does not ' +
                        'look like previous results--trying as parameter file')
                    with open(self.baseline_path) as f:
                        params: Dict[str, Any] = json.load(f)
                        hyp = self.hyperparams.clone()
                        hyp.update(params)
                        # return a bogus result, which is alright since used
                        # only to copy parameters
                        res = HyperparamResult(
                            hyp=hyp,
                            scores=None,
                            loss=-1,
                            eval_ix=-1)
        if set_hyp:
            self.hyperparams.update(res.hyp)
        return res

    def get_comparison(self) -> CompareResult:
        """Compare the scores of the default parameters with those predicted by
        the optimizer of the best run.

        """
        hyp: HyperparamModel = self.hyperparams
        prev: Dict[str, Any] = hyp.flatten()
        cmp_res: CompareResult = None
        try:
            initial_loss: float
            initial_scores: pd.DataFrame
            initial_loss, initial_scores = self._compare()
            best_res: HyperparamResult = self._get_baseline()
            best: Dict[str, Any] = best_res.hyp.flatten()
            best_loss: float
            best_scores: pd.DataFrame
            hyp.update(best)
            best_loss, best_scores = self._compare()
            cmp_res = CompareResult(
                initial_param=prev,
                initial_loss=initial_loss,
                initial_scores=initial_scores,
                best_eval_ix=best_res.eval_ix,
                best_param=best,
                best_loss=best_loss,
                best_scores=best_scores)
        finally:
            hyp.update(prev)
        return cmp_res

    def _write_result(self, res: HyperparamResult, writer: TextIOBase):
        print(f'{self.name}:', file=writer)
        res.write(writer=writer)

    def write_best_result(self, writer: TextIOBase = sys.stdout,
                          include_param_json: bool = False):
        """Print the results from the best run.

        :param include_param_json: whether to output the JSON formatted
                                   hyperparameters

        """
        best_res: HyperparamResult = self.get_best_result()
        self._write_result(best_res, writer)
        if include_param_json:
            print('parameters:', file=writer)
            print(json.dumps(best_res.hyp.flatten(), indent=4), file=writer)

    def write_compare(self, writer: TextIOBase = sys.stdout):
        """Write the results of a compare of the initial hyperparameters against
        the optimized.

        """
        cmp_res: CompareResult = self.get_comparison()
        if cmp_res is None:
            print('no results or error', file=writer)
        else:
            cmp_res.write(writer=writer)

    def get_score_dataframe(self, iterations: int = None) -> pd.DataFrame:
        """Create a dataframe from the results scored from the best
        hyperparameters.

        :param iterations: the number times the objective is called to produce
                           the results (the objective space is not altered)

        """
        hyp: HyperparamModel = self.hyperparams
        prev: Dict[str, Any] = hyp.flatten()
        dfs: List[pd.DataFrame] = []
        if iterations is None:
            iterations = self._get_score_iterations()
        logger.info(f'scoring {iterations} iterations using best settings')
        try:
            self._get_baseline()
            for i in range(iterations):
                self._run_objective(None)
                df: pd.DataFrame = self._last_result.scores
                df.insert(0, 'iteration',
                          tuple(it.islice(it.repeat(i), len(df))))
                dfs.append(df)
        finally:
            hyp.update(prev)
        return pd.concat(dfs)

    def write_score(self, writer: TextIOBase = sys.stdout) -> HyperparamResult:
        """Restore the hyperparameter state, score the data and print the
        results.  Use the :obj:`baseline` parameters if available, otherwise use
        the parameters from the best best run.

        """
        hyp: HyperparamModel = self.hyperparams
        prev: Dict[str, Any] = hyp.flatten()
        try:
            self._get_baseline()
            print('using hyperparameters:')
            self.hyperparams.write(1)
            self._run_objective(None)
            self._write_result(self._last_result, writer)
            return self._last_result
        finally:
            hyp.update(prev)

    def write_scores(self, output_file: Path = None, iterations: int = None):
        """Write a file of the results scored from the best hyperparameters.

        :param output_file: where to write the CSV file; defaults to a file in
                            :obj:`opt_intermediate_dir`

        :param iterations: the number times the objective is called to produce
                           the results (the objective space is not altered)

        """
        if output_file is None:
            output_file = self.intermediate_dir / 'scores' / f'{self.name}.csv'
        df: pd.DataFrame = self.get_score_dataframe(iterations)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file)
        logger.info(f'wrote scores to {output_file}')

    @property
    def aggregate_score_dir(self) -> Path:
        """The output directory containing runs with the best parameters of the
        top N results (see :meth:`aggregate_scores`).

        """
        return self.intermediate_dir / 'agg'

    def aggregate_scores(self):
        """Aggregate best score results as a separate CSV file for each data
        point with :meth:`get_score_dataframe`.  This is saved as a separate
        file for each optmiziation run since this method can take a long time as
        it will re-score the dataset.  These results are then "stiched" together
        with :meth:`gather_aggregate_scores`.

        """
        results: Dict[str, HyperparamResult] = self.get_best_results()
        res_tups: Tuple[str, HyperparamResult] = sorted(
            results.items(), key=lambda t: t[1].loss)
        logger.info('scoring top best results')
        self.aggregate_score_dir.mkdir(parents=True, exist_ok=True)
        logger.setLevel(logging.INFO)
        name: str
        best: HyperparamResult
        for name, best in res_tups:
            output_file: Path = self.aggregate_score_dir / f'{name}.csv'
            self.name = name
            logger.info(f'scoring {name} on {self.match_sample_size} ' +
                        f'samples with loss {best.loss}')
            df: pd.DataFrame = self.get_score_dataframe(1)
            df.insert(0, 'name', name)
            df.to_csv(output_file, index=False)
            logger.info(f'wrote: {output_file}')

    def gather_aggregate_scores(self) -> pd.DataFrame:
        """Return a dataframe of all the aggregate scores written by
        :meth:`aggregate_scores`.

        """
        dfs: List[pd.DataFrame] = []
        agg_score_file: Path
        for agg_score_file in self.aggregate_score_dir.iterdir():
            dfs.append(pd.read_csv(agg_score_file))
        return pd.concat(dfs)
