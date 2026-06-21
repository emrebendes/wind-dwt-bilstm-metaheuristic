# -*- coding: utf-8 -*-
"""
optimizers/gwo_optimizer.py — Grey Wolf Optimizer.

Mirjalili et al. (2014) tarafından önerilen GWO. Sürüdeki en iyi 3 kurt
(alpha, beta, delta) pozisyonlarını referans alarak diğer kurtların
pozisyonu update edilir; "encircling prey" mekanizması ile.

Referans: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf
optimizer. Advances in Engineering Software, 69, 46-61.

Klasik update formülü:
    A_i = 2·a·r1 - a, C_i = 2·r2  (i ∈ {α, β, δ})
    D_i = |C_i · X_i - X|
    X_i_candidate = X_i - A_i · D_i
    X_new = (X_α_cand + X_β_cand + X_δ_cand) / 3

Burada `a` linear olarak 2'den 0'a düşer.

Algoritma-spesifik parametreler:
    a_start, a_end: Encircling katsayısı (linear decay)
"""

import time
import numpy as np
from typing import Dict, Any

from optimizers import register_algorithm
from optimizers.base_optimizer import BaseOptimizer


@register_algorithm("gwo")
class GWOOptimizer(BaseOptimizer):
    """Grey Wolf Optimizer (Mirjalili et al., 2014)."""

    DEFAULT_PARAMS = {
        'a_start': 2.0,  # encircling katsayısı (başlangıç)
        'a_end': 0.0,    # encircling katsayısı (bitiş)
    }

    # =========================================================================
    # POPÜLASYON BAŞLATMA
    # =========================================================================

    def _initialize_population(self) -> None:
        """Random kurt pozisyonları + alpha/beta/delta tracker'lar."""
        self.population = np.random.rand(self.pop_size, self.dim)
        self.fitness = np.zeros(self.pop_size)
        self.f = np.full(self.pop_size, float('inf'))

        # GWO-spesifik state: alpha/beta/delta
        # alpha = en iyi, beta = ikinci, delta = üçüncü
        self.alpha_pos = np.random.rand(self.dim)
        self.alpha_f = float('inf')
        self.beta_pos = np.random.rand(self.dim)
        self.beta_f = float('inf')
        self.delta_pos = np.random.rand(self.dim)
        self.delta_f = float('inf')

    # =========================================================================
    # STATE SERIALIZATION (GWO-spesifik α/β/δ pozisyonları)
    # =========================================================================

    def _get_state_dict(self) -> Dict[str, Any]:
        d = super()._get_state_dict()
        d['alpha_pos'] = self.alpha_pos
        d['alpha_f'] = self.alpha_f
        d['beta_pos'] = self.beta_pos
        d['beta_f'] = self.beta_f
        d['delta_pos'] = self.delta_pos
        d['delta_f'] = self.delta_f
        return d

    def _set_state_dict(self, state: Dict[str, Any]) -> None:
        super()._set_state_dict(state)
        self.alpha_pos = state.get('alpha_pos')
        self.alpha_f = state.get('alpha_f', float('inf'))
        self.beta_pos = state.get('beta_pos')
        self.beta_f = state.get('beta_f', float('inf'))
        self.delta_pos = state.get('delta_pos')
        self.delta_f = state.get('delta_f', float('inf'))

    # =========================================================================
    # GWO HİYERARŞİ GÜNCELLEMESİ
    # =========================================================================

    def _update_hierarchy(self, results) -> None:
        """Tüm popülasyonu tarayıp α/β/δ kurtlarını güncelle."""
        for i, result in enumerate(results):
            loss = result['loss']
            pos = self.population[i]

            if loss < self.alpha_f:
                # alpha → beta → delta cascade
                self.delta_f = self.beta_f
                self.delta_pos = self.beta_pos.copy() if self.beta_pos is not None else pos.copy()
                self.beta_f = self.alpha_f
                self.beta_pos = self.alpha_pos.copy() if self.alpha_pos is not None else pos.copy()
                self.alpha_f = loss
                self.alpha_pos = pos.copy()
            elif loss < self.beta_f:
                self.delta_f = self.beta_f
                self.delta_pos = self.beta_pos.copy() if self.beta_pos is not None else pos.copy()
                self.beta_f = loss
                self.beta_pos = pos.copy()
            elif loss < self.delta_f:
                self.delta_f = loss
                self.delta_pos = pos.copy()

    # =========================================================================
    # ANA ITERASYON
    # =========================================================================

    def _run_iteration(self, iteration: int) -> None:
        """Tek bir GWO iterasyonu."""
        iter_start = time.time()

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"GWO ITERATION {iteration+1}/{self.max_iterations}")

        # Linear decreasing 'a' coefficient
        a = self.algo_params['a_start'] - (
            (self.algo_params['a_start'] - self.algo_params['a_end'])
            * iteration / max(1, self.max_iterations - 1)
        )

        # Position update for each wolf
        new_population = np.zeros_like(self.population)
        for i in range(self.pop_size):
            x = self.population[i]

            # Three random A and C coefficients per leader
            cands = []
            for leader_pos in (self.alpha_pos, self.beta_pos, self.delta_pos):
                if leader_pos is None:
                    continue
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader_pos - x)
                cands.append(leader_pos - A * D)

            # Mean of three candidate positions
            new_x = np.mean(cands, axis=0) if cands else x
            new_population[i] = np.clip(new_x, 0.0, 1.0)

        self.population = new_population

        # Evaluate
        results = self.evaluate(
            vectors=[self.population[i] for i in range(self.pop_size)],
            indices=list(range(self.pop_size)),
            member_type="W",  # W = Wolf
            iteration=iteration,
        )

        # Update fitness arrays + global best + hierarchy
        improvements = []
        for i, result in enumerate(results):
            self.f[i] = result['loss']
            self.fitness[i] = result['fitness']
            improved = self.update_global_best(i, iteration, 'GWO', result)
            improvements.append(improved)

        self._update_hierarchy(results)

        # Global best alpha pozisyonuna eşitle (alpha = en iyi kurt)
        if self.alpha_f < self.global_opt:
            self.global_opt = self.alpha_f
            self.global_vector = self.alpha_pos.copy()

        self.save_trials(iteration, 'GWO', results, improvements)

        duration = time.time() - iter_start
        self.save_iteration_stats(
            iteration, results, duration,
            extra={
                'a': a,
                'alpha_f': self.alpha_f,
                'beta_f': self.beta_f,
                'delta_f': self.delta_f,
            }
        )
        self.save_checkpoint(iteration, 'ITER_COMPLETE')

        self.logger.info(
            f"\n[ITER {iteration+1}] α={self.alpha_f:.6f} β={self.beta_f:.6f} "
            f"δ={self.delta_f:.6f} | a={a:.3f} | Duration: {duration:.1f}s"
        )
