"""
optimizers.py
Faithful re-implementation of the Grey Wolf Optimizer (GWO; Mirjalili et al., 2014)
as coded in the source notebook, plus a Particle Swarm Optimization (PSO;
Kennedy & Eberhart, 1995) baseline used for comparative benchmarking.

Both optimizers share the same interface: minimize(objective_func, dim, bounds,
population_size, max_iter, seed) -> (best_position, best_fitness, convergence_curve)
"""
import numpy as np
import time
from typing import Callable, Tuple, List


class GreyWolfOptimizer:
    """Grey Wolf Optimizer, following Mirjalili, Mirjalili & Lewis (2014)."""

    def __init__(self, objective_func: Callable, dim: int, bounds: Tuple[float, float],
                 population_size: int = 30, max_iter: int = 100, seed: int = None):
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        self.positions = self.rng.uniform(bounds[0], bounds[1], (population_size, dim))
        self.fitness = np.full(population_size, np.inf)

        self.alpha_pos = np.zeros(dim); self.alpha_score = np.inf
        self.beta_pos = np.zeros(dim); self.beta_score = np.inf
        self.delta_pos = np.zeros(dim); self.delta_score = np.inf

        self.a = 2.0
        self.convergence_curve = []
        self.execution_time = 0.0
        self.function_evaluations = 0

    def evaluate_fitness(self, position):
        self.function_evaluations += 1
        return self.objective_func(position)

    def update_social_hierarchy(self):
        order = np.argsort(self.fitness)
        if self.fitness[order[0]] < self.alpha_score:
            self.alpha_pos = self.positions[order[0]].copy(); self.alpha_score = self.fitness[order[0]]
        if self.fitness[order[1]] < self.beta_score:
            self.beta_pos = self.positions[order[1]].copy(); self.beta_score = self.fitness[order[1]]
        if self.fitness[order[2]] < self.delta_score:
            self.delta_pos = self.positions[order[2]].copy(); self.delta_score = self.fitness[order[2]]
        self.omega_indices = order[3:]

    def update_omega_positions(self, current_iter):
        self.a = 2 - (2 * current_iter / self.max_iter)
        for i in self.omega_indices:
            new_position = np.zeros(self.dim)
            for j in range(self.dim):
                r1, r2 = self.rng.random(), self.rng.random()
                A1 = 2 * self.a * r1 - self.a; C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                X1 = self.alpha_pos[j] - A1 * D_alpha

                r1, r2 = self.rng.random(), self.rng.random()
                A2 = 2 * self.a * r1 - self.a; C2 = 2 * r2
                D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                X2 = self.beta_pos[j] - A2 * D_beta

                r1, r2 = self.rng.random(), self.rng.random()
                A3 = 2 * self.a * r1 - self.a; C3 = 2 * r2
                D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                X3 = self.delta_pos[j] - A3 * D_delta

                new_position[j] = (X1 + X2 + X3) / 3
            new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
            self.positions[i] = new_position
            self.fitness[i] = self.evaluate_fitness(new_position)

    def optimize(self):
        start = time.time()
        for i in range(self.population_size):
            self.fitness[i] = self.evaluate_fitness(self.positions[i])
        self.update_social_hierarchy()
        self.convergence_curve = [self.alpha_score]
        for it in range(self.max_iter):
            self.update_omega_positions(it)
            self.update_social_hierarchy()
            self.convergence_curve.append(self.alpha_score)
        self.execution_time = time.time() - start
        return self.alpha_pos, self.alpha_score, self.convergence_curve


class ParticleSwarmOptimizer:
    """Standard inertia-weight PSO, following Kennedy & Eberhart (1995) /
    Shi & Eberhart (1998) inertia-weight variant, used as a comparative baseline."""

    def __init__(self, objective_func: Callable, dim: int, bounds: Tuple[float, float],
                 population_size: int = 30, max_iter: int = 100, seed: int = None,
                 w_max: float = 0.9, w_min: float = 0.4, c1: float = 2.0, c2: float = 2.0):
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self.w_max, self.w_min, self.c1, self.c2 = w_max, w_min, c1, c2

        self.positions = self.rng.uniform(bounds[0], bounds[1], (population_size, dim))
        span = bounds[1] - bounds[0]
        self.velocities = self.rng.uniform(-abs(span) * 0.1, abs(span) * 0.1, (population_size, dim))

        self.pbest_pos = self.positions.copy()
        self.pbest_val = np.full(population_size, np.inf)
        self.gbest_pos = np.zeros(dim)
        self.gbest_val = np.inf

        self.execution_time = 0.0
        self.function_evaluations = 0

    def optimize(self):
        start = time.time()
        fitness = np.array([self.objective_func(p) for p in self.positions])
        self.function_evaluations += self.population_size
        self.pbest_val = fitness.copy()
        self.pbest_pos = self.positions.copy()
        best_idx = np.argmin(fitness)
        self.gbest_val = fitness[best_idx]
        self.gbest_pos = self.positions[best_idx].copy()

        convergence = [self.gbest_val]

        for it in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * (it / self.max_iter)
            r1 = self.rng.random((self.population_size, self.dim))
            r2 = self.rng.random((self.population_size, self.dim))

            self.velocities = (w * self.velocities
                                + self.c1 * r1 * (self.pbest_pos - self.positions)
                                + self.c2 * r2 * (self.gbest_pos - self.positions))
            self.positions = np.clip(self.positions + self.velocities, self.bounds[0], self.bounds[1])

            fitness = np.array([self.objective_func(p) for p in self.positions])
            self.function_evaluations += self.population_size

            improved = fitness < self.pbest_val
            self.pbest_val[improved] = fitness[improved]
            self.pbest_pos[improved] = self.positions[improved]

            best_idx = np.argmin(self.pbest_val)
            if self.pbest_val[best_idx] < self.gbest_val:
                self.gbest_val = self.pbest_val[best_idx]
                self.gbest_pos = self.pbest_pos[best_idx].copy()

            convergence.append(self.gbest_val)

        self.execution_time = time.time() - start
        return self.gbest_pos, self.gbest_val, convergence
