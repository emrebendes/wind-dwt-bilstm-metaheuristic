# -*- coding: utf-8 -*-
"""
optimizers/pso_optimizer.py — Particle Swarm Optimization.

Kennedy & Eberhart (1995) tarafından önerilen klasik PSO. Her parçacık
hem kendi geçmiş en iyi pozisyonunu (pbest) hem global en iyi pozisyonu
(gbest) takip ederek arama yapar.

Referans: Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
Proceedings of ICNN'95-International Conference on Neural Networks.

Klasik update formülü:
    v(t+1) = w·v(t) + c1·r1·(pbest - x) + c2·r2·(gbest - x)
    x(t+1) = x(t) + v(t+1)

Algoritma-spesifik parametreler:
    w_start, w_end: Atalet katsayısı (linear decay; Shi & Eberhart 1998)
    c1: Cognitive coefficient (kişisel deneyim ağırlığı)
    c2: Social coefficient (sürü deneyimi ağırlığı)
    v_max: Hız sınırı (search space'in yüzdesi olarak)
"""

import time
import numpy as np
from typing import Dict, Any, List

from optimizers import register_algorithm
from optimizers.base_optimizer import BaseOptimizer


@register_algorithm("pso")
class PSOOptimizer(BaseOptimizer):
    """Particle Swarm Optimization (Kennedy & Eberhart, 1995)."""

    DEFAULT_PARAMS = {
        'w_start': 0.9,    # Atalet (linear decay başlangıç)
        'w_end': 0.4,      # Atalet (linear decay bitiş)
        'c1': 2.0,         # Cognitive coefficient
        'c2': 2.0,         # Social coefficient
        'v_max': 0.2,      # Hız sınırı (search space [0,1] yüzdesi)
    }

    # =========================================================================
    # POPÜLASYON BAŞLATMA
    # =========================================================================

    def _initialize_population(self) -> None:
        """Random pozisyon, sıfır hız, pbest = pozisyon."""
        self.population = np.random.rand(self.pop_size, self.dim)
        self.fitness = np.zeros(self.pop_size)
        self.f = np.full(self.pop_size, float('inf'))

        # PSO-spesifik state
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.pbest = self.population.copy()
        self.pbest_f = np.full(self.pop_size, float('inf'))

    # =========================================================================
    # STATE SERIALIZATION (PSO-spesifik velocities + pbest)
    # =========================================================================

    def _get_state_dict(self) -> Dict[str, Any]:
        d = super()._get_state_dict()
        d['velocities'] = self.velocities
        d['pbest'] = self.pbest
        d['pbest_f'] = self.pbest_f
        return d

    def _set_state_dict(self, state: Dict[str, Any]) -> None:
        super()._set_state_dict(state)
        self.velocities = state.get('velocities')
        self.pbest = state.get('pbest')
        self.pbest_f = state.get('pbest_f')

    # =========================================================================
    # ANA ITERASYON
    # =========================================================================

    def _run_iteration(self, iteration: int) -> None:
        """Tek bir PSO iterasyonu: hız + pozisyon update + değerlendir."""
        iter_start = time.time()

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"PSO ITERATION {iteration+1}/{self.max_iterations}")

        # Linear decreasing inertia
        w = self.algo_params['w_start'] - (
            (self.algo_params['w_start'] - self.algo_params['w_end'])
            * iteration / max(1, self.max_iterations - 1)
        )
        c1 = self.algo_params['c1']
        c2 = self.algo_params['c2']
        v_max = self.algo_params['v_max']

        # Velocity + position update (vectorize for speed)
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)

        cognitive = c1 * r1 * (self.pbest - self.population)
        social = c2 * r2 * (self.global_vector[None, :] - self.population)

        self.velocities = w * self.velocities + cognitive + social
        self.velocities = np.clip(self.velocities, -v_max, v_max)

        self.population = np.clip(self.population + self.velocities, 0.0, 1.0)

        # Değerlendir
        results = self.evaluate(
            vectors=[self.population[i] for i in range(self.pop_size)],
            indices=list(range(self.pop_size)),
            member_type="P",  # P = Particle
            iteration=iteration,
        )

        improvements = []
        for i, result in enumerate(results):
            self.f[i] = result['loss']
            self.fitness[i] = result['fitness']

            # pbest güncelle
            improved_pbest = result['loss'] < self.pbest_f[i]
            if improved_pbest:
                self.pbest[i] = self.population[i].copy()
                self.pbest_f[i] = result['loss']

            # gbest güncelle
            improved_gbest = self.update_global_best(i, iteration, 'PSO', result)
            improvements.append(improved_pbest or improved_gbest)

        self.save_trials(iteration, 'PSO', results, improvements)

        duration = time.time() - iter_start
        self.save_iteration_stats(
            iteration, results, duration,
            extra={
                'w': w,
                'pbest_improvements': sum(improvements),
            }
        )
        self.save_checkpoint(iteration, 'ITER_COMPLETE')

        self.logger.info(
            f"\n[ITER {iteration+1}] Best: {self.global_opt:.6f} | "
            f"w={w:.3f} | Duration: {duration:.1f}s"
        )
