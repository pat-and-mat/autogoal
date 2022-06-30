import random
import logging
import pickle
from typing import Dict, List, Optional
import enlighten
import time
import datetime
import statistics
import math
import termcolor
import json

import autogoal.logging
from autogoal.search._base import Logger, ConsoleLogger, MultiLogger

from autogoal.utils import RestrictedWorkerByJoin, Min, Gb, Sec
from autogoal.sampling import ModelSampler, ReplaySampler, merge_updates, update_model
from rich.progress import Progress
from rich.panel import Panel


class NSSearch:
    def __init__(
        self,
        generator_fn=None,
        fitness_fn=None,
        pop_size=20,
        maximize=True,
        errors="raise",
        early_stop=0.5,
        evaluation_timeout: int = 10 * Sec,
        memory_limit: int = 4 * Gb,
        search_timeout: int = 5 * Min,
        target_fn=None,
        allow_duplicates=True,
    ):
        if generator_fn is None and fitness_fn is None:
            raise ValueError("You must provide either `generator_fn` or `fitness_fn`")

        self._generator_fn = generator_fn
        self._fitness_fn = fitness_fn or (lambda x: x)
        self._pop_size = pop_size
        self._maximize = maximize
        self._errors = errors
        self._evaluation_timeout = evaluation_timeout
        self._memory_limit = memory_limit
        self._early_stop = early_stop
        self._search_timeout = search_timeout
        self._target_fn = target_fn
        self._allow_duplicates = allow_duplicates

        if self._evaluation_timeout > 0 or self._memory_limit > 0:
            self._fitness_fn = RestrictedWorkerByJoin(
                self._fitness_fn, self._evaluation_timeout, self._memory_limit
            )

    def run(self, generations=None, logger=None):
        """Runs the search performing at most `generations` of `fitness_fn`.

        Returns:
            Tuple `(best, fn)` of the best found solution and its corresponding fitness.
        """
        if logger is None:
            logger = Logger()

        if generations is None:
            generations = math.inf

        if isinstance(logger, list):
            logger = MultiLogger(*logger)

        if isinstance(self._early_stop, float):
            early_stop = int(self._early_stop * generations)
        else:
            early_stop = self._early_stop

        best_solution = None
        best_fn = None
        no_improvement = 0
        start_time = time.time()
        seen = set()

        logger.begin(generations, self._pop_size)

        try:
            while generations > 0:
                stop = False

                logger.start_generation(generations, best_fn)
                self._start_generation()

                solutions, fns = [], []

                improvement = False

                for _ in range(self._pop_size):
                    solution = None

                    try:
                        solution = self._generate()
                    except Exception as e:
                        logger.error(
                            "Error while generating solution: %s" % e, solution
                        )
                        continue

                    if not self._allow_duplicates and repr(solution) in seen:
                        continue

                    try:
                        logger.sample_solution(solution)
                        fn = self._fitness_fn(solution)
                    except Exception as e:
                        fn = tuple(
                            (-math.inf if m else math.inf) for m in self._maximize
                        )
                        logger.error(e, solution)

                        if self._errors == "raise":
                            logger.end(best_solution, best_fn)
                            raise e from None

                    if not self._allow_duplicates:
                        seen.add(repr(solution))

                    logger.eval_solution(solution, fn)
                    solutions.append(solution)
                    fns.append(fn)

                    # if (
                    #     best_fn is None
                    #     or (fn > best_fn and self._maximize)
                    #     or (fn < best_fn and not self._maximize)
                    # ):
                    #     logger.update_best(solution, fn, best_solution, best_fn)
                    #     best_solution = solution
                    #     best_fn = fn
                    #     improvement = True

                    #     if self._target_fn is not None and (
                    #         (best_fn >= self._target_fn and self._maximize)
                    #         or (best_fn <= self._target_fn and not self._maximize)
                    #     ):
                    #         stop = True
                    #         break

                    spent_time = time.time() - start_time

                    if self._search_timeout and spent_time > self._search_timeout:
                        autogoal.logging.logger().info(
                            "(!) Stopping since time spent is %.2f." % (spent_time)
                        )
                        stop = True
                        break

                fronts = self.non_dominated_fronts(fns)
                solution, fn = (solutions[fronts[0][0]], fns[fronts[0][0]])

                if best_fn is None or self._dominates(fn, best_fn):
                    logger.update_best(solution, fn, best_solution, best_fn)
                    best_solution = solution
                    best_fn = fn
                    improvement = True

                    if self._target_fn is not None and self._dominates(
                        best_fn, self._target_fn
                    ):
                        stop = True

                if not improvement:
                    no_improvement += 1
                else:
                    no_improvement = 0

                generations -= 1

                if generations <= 0:
                    autogoal.logging.logger().info(
                        "(!) Stopping since all generations are done."
                    )
                    stop = True
                    break

                if early_stop and no_improvement >= early_stop:
                    autogoal.logging.logger().info(
                        "(!) Stopping since no improvement for %i generations."
                        % no_improvement
                    )
                    stop = True
                    break

                logger.finish_generation(fns)
                self._finish_generation(fns, fronts)

                if stop:
                    break

        except KeyboardInterrupt:
            pass

        logger.end(best_solution, best_fn)
        return best_solution, best_fn

    def _generate(self):
        # BUG: When multiprocessing is used for evaluation and no generation
        #      function is defined, the actual sampling occurs during fitness
        #      evaluation, and since that process has a copy of the solution
        #      we don't get the history in the `ReplaySampler`.

        sampler = ReplaySampler(self._build_sampler())

        if self._generator_fn is not None:
            solution = self._generator_fn(sampler)
        else:
            solution = sampler

        solution.sampler_ = sampler
        return solution

    def non_dominated_fronts(self, fns) -> List[int]:
        fronts = [[]]
        domination_counts = [0] * len(fns)
        dominated_fns = [[] for _ in fns]

        for i, fn_i in enumerate(fns):
            for j, fn_j in enumerate(fns):
                if self._dominates(fn_i, fn_j):
                    dominated_fns[i].append(j)
                elif self._dominates(fn_j, fn_i):
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                fronts[0].append(i)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for idx in fronts[i]:
                for dominated_idx in dominated_fns[idx]:
                    domination_counts[dominated_idx] -= 1
                    if domination_counts[dominated_idx] == 0:
                        next_front.append(dominated_idx)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def _dominates(self, a, b) -> bool:
        is_non_dominated = all(
            (a[i] >= b[i] if m else a[i] <= b[i]) for i, m in enumerate(self._maximize)
        )
        dominates = any(
            (a[i] > b[i] if m else a[i] < b[i]) for i, m in enumerate(self._maximize)
        )
        return is_non_dominated and dominates

    def _build_sampler(self):
        raise NotImplementedError()

    def _start_generation(self):
        pass

    def _finish_generation(self, fns, fronts):
        pass


class NSPESearch(NSSearch):
    def __init__(
        self,
        *args,
        learning_factor: float = 0.05,
        selection: float = 0.2,
        epsilon_greed: float = 0.1,
        random_state: Optional[int] = None,
        name: str = None,
        save: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._learning_factor = learning_factor
        self._selection = selection
        self._epsilon_greed = epsilon_greed
        self._model: Dict = {}
        self._random_states = random.Random(random_state)
        self._name = name or str(time.time())
        self._save = save

    def _start_generation(self):
        self._samplers = []

    def _build_sampler(self):
        if len(self._samplers) < self._epsilon_greed * self._pop_size:
            sampler = ModelSampler(random_state=self._random_states.getrandbits(32))
        else:
            sampler = ModelSampler(
                self._model, random_state=self._random_states.getrandbits(32)
            )

        self._samplers.append(sampler)
        return sampler

    def _finish_generation(self, fns, fronts):
        # Compute the marginal model of the best pipelines
        indices = self.best_indices_by_ranking_and_crowding(
            fns, fronts, k=int(self._selection * len(fns))
        )
        samplers: List[ModelSampler] = [self._samplers[i] for i in indices]
        updates: Dict = merge_updates(*[sampler.updates for sampler in samplers])

        # Update the probabilistic model with the marginal model from the best pipelines
        self._model = update_model(self._model, updates, self._learning_factor)

        # save an internal state of metaheuristic for other executions
        if self._save == True:
            with open("model-" + self._name + ".pickle", "wb") as f:
                pickle.dump(self._model, f)

    def best_indices_by_ranking_and_crowding(self, fns, fronts, k) -> List[int]:
        indices = []
        for front in fronts:
            if len(indices) + len(front) <= k:
                indices.extend(front)
            else:
                indices.extend(
                    sorted(
                        front,
                        key=lambda i: (fns[i], self.crowding_distance(fns, front, i)),
                        reverse=True,
                    )[: k - len(indices)]
                )
                break
        return indices

    def crowding_distance(self, fns, front, i):
        if len(front) == 0:
            raise ValueError("Front is empty.")
        elif len(front) < 0:
            raise ValueError("Front is negative.")

        crowding_distances = [0] * len(front)
        for m in range(len(self._maximize)):
            front = list(sorted(front, key=lambda i: fns[i][m]))
            crowding_distances[front[0]] = math.inf
            crowding_distances[front[len(front) - 1]] = math.inf
            m_values = [fns[i][m] for i in front]
            scale = max(m_values) - min(m_values)
            if scale == 0:
                scale = 1
            for i in range(1, len(front) - 1):
                crowding_distances[i] += (
                    fns[front[i + 1]][m] - fns[front[i - 1]][m]
                ) / scale
        return crowding_distances[i]

    def load(self, name_pickle_file):
        """Rewrites the probabilistic distribution of metaheuristic with the value of the name model."""

        with open(name_pickle_file) as f:
            loaded_obj = pickle.load(f)
        self._model = loaded_obj


class ConsoleLoggerWithMOSupport(ConsoleLogger):
    def start_generation(self, generations, best_fn):
        if not isinstance(best_fn, (list, tuple)):
            return super().start_generation(generations, best_fn)

        current_time = time.time()
        elapsed = int(current_time - self.start_time)
        avg_time = elapsed / (self.start_generations - generations + 1)
        remaining = int(avg_time * generations)
        elapsed = datetime.timedelta(seconds=elapsed)
        remaining = datetime.timedelta(seconds=remaining)

        best_fn_str_tuple = tuple(f"{float(fn or 0.0):0.3}" for fn in best_fn)

        print(
            self.emph("New generation started"),
            self.success(f"best_fn={repr(best_fn_str_tuple)}"),
            self.primary(f"generations={generations}"),
            self.primary(f"elapsed={elapsed}"),
            self.primary(f"remaining={remaining}"),
        )

    def end(self, best, best_fn):
        if not isinstance(best_fn, (list, tuple)):
            return super().end(best, best_fn)

        best_fn_str_tuple = tuple(f"{float(fn or 0.0):0.3}" for fn in best_fn)
        print(
            "Ending:",
            self.emph(
                "Search completed: best_fn=%s, best=\n%r"
                % (repr(best_fn_str_tuple), best)
            ),
        )

    def eval_solution(self, solution, fitness):
        if not isinstance(fitness, (list, tuple)):
            return super().eval_solution(solution, fitness)

        fitness_str_tuple = tuple(f"{float(fn or 0.0):0.3}" for fn in fitness)
        print(self.primary("Fitness=%s" % repr(fitness_str_tuple)))

    def update_best(self, new_best, new_fn, previous_best, previous_fn):
        if not isinstance(new_fn, (list, tuple)) and not isinstance(
            previous_fn, (list, tuple)
        ):
            return super().update_best(new_best, new_fn, previous_best, previous_fn)

        new_fn_str_tuple = tuple(f"{float(fn):0.3}" for fn in new_fn)
        prev_fn_str_tuple = tuple(f"{float(fn):0.3}" for fn in previous_fn)
        print(
            self.success(
                "Best solution: improved=%s, previous=%s"
                % (repr(new_fn_str_tuple), repr(prev_fn_str_tuple or 0))
            )
        )
