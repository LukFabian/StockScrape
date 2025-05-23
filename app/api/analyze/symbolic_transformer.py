import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer
from joblib import Parallel, delayed
from scipy.stats import spearmanr


class CustomSymbolicTransformer(SymbolicTransformer):
    def __init__(
            self,
            *,
            population_size=1000,
            hall_of_fame=100,
            n_components=10,
            generations=20,
            tournament_size=20,
            stopping_criteria=1.0,
            const_range=(-1., 1.),
            init_depth=(2, 6),
            init_method='half and half',
            function_set=('add', 'sub', 'mul', 'div'),
            metric='pearson',
            parsimony_coefficient=0.001,
            p_crossover=0.9,
            p_subtree_mutation=0.01,
            p_hoist_mutation=0.01,
            p_point_mutation=0.01,
            p_point_replace=0.05,
            max_samples=1.0,
            feature_names=None,
            warm_start=False,
            low_memory=False,
            n_jobs=1,
            verbose=0,
            random_state=None,
            y_return=None  # <-- custom parameter (NOT part of gplearn API)
    ):
        super().__init__(
            population_size=population_size,
            hall_of_fame=hall_of_fame,
            n_components=n_components,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )
        self.y_return = y_return

    def _custom_fitness(self, program_output, y_return, k_groups=5):
        """
        Computes combined fitness score based on:
        - Rank IC
        - Monotonicity of return buckets
        - Bottom group return
        """
        if np.std(program_output) < 1e-6 or np.isnan(program_output).any():
            return -np.inf  # penalize constant or invalid outputs

        # Rank IC
        ic = spearmanr(program_output, y_return).correlation
        if np.isnan(ic):
            ic = 0

        # Monotonicity & bottom group return
        df = pd.DataFrame({'feat': program_output, 'ret': y_return})
        try:
            df['group'] = pd.qcut(df['feat'], q=k_groups, labels=False, duplicates='drop')
        except ValueError:
            return -np.inf  # too few unique values

        group_returns = df.groupby('group')['ret'].mean()
        diffs = group_returns.diff().dropna()
        monotonicity = (diffs > 0).sum() / len(diffs)  # ratio of increasing steps

        bottom_return = group_returns.iloc[0]

        # Combine
        fitness = ic + 0.3 * monotonicity + 0.2 * bottom_return
        return fitness

    def _fitness_function(self, programs, X, y, sample_weight, n_jobs, random_state):
        """
        Override gplearn's internal fitness scoring to use custom function.
        `y` is not used here — instead we use self.y_return.
        """
        n_programs = len(programs)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        def eval_program(i):
            program = programs[i]
            try:
                output = program.execute(X)
            except Exception:
                return -np.inf  # invalid execution
            return self._custom_fitness(output, self.y_return)

        return Parallel(n_jobs=n_jobs)(
            delayed(eval_program)(i) for i in range(n_programs)
        )
