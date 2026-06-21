# -*- coding: utf-8 -*-
"""
optimizers/ho_optimizer.py - Hippopotamus Optimization Algorithm.

Amiri, Hubalovsky & Trojovsky (2024) tarafindan onerilen HO. Hipopotamlarin
3 davranissal fazi metafor olarak kullanir:

1. River/Pond Phase: Sürünün dominant erkegi (en iyi) ve sürü ortalamasina
   gore pozisyon guncellemesi (keşif).
2. Defensive Phase: Yirtici tehdidi (rastgele predator) — yirtici uzaksa
   karşılayıp dairese, yakinsa kaçma (somurum/keşif dengesi).
3. Evasion Phase: Rastgele yer degistirme — local optimadan kaçış.

Referans: Amiri, M.H., Hubálovský, S., & Trojovský, P. (2024).
Hippopotamus optimization algorithm: a novel nature-inspired optimization
algorithm. Scientific Reports, 14, 5032. DOI: 10.1038/s41598-024-54910-3
"""

import time
import numpy as np
from typing import Dict, Any

from optimizers import register_algorithm
from optimizers.base_optimizer import BaseOptimizer


@register_algorithm("ho")
class HOOptimizer(BaseOptimizer):
    """Hippopotamus Optimization (Amiri et al., 2024)."""

    DEFAULT_PARAMS = {
        # 3 fazin populasyonu kapsama orani (toplam 1.0)
        "phase1_ratio": 0.5,    # River/Pond fazi
        "phase2_ratio": 0.3,    # Defensive fazi
        "phase3_ratio": 0.2,    # Evasion fazi
    }

    def _initialize_population(self) -> None:
        self.population = np.random.rand(self.pop_size, self.dim)
        self.fitness = np.zeros(self.pop_size)
        self.f = np.full(self.pop_size, float("inf"))

    def _run_iteration(self, iteration: int) -> None:
        iter_start = time.time()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"HO ITERATION {iteration+1}/{self.max_iterations}")

        T = max(1, self.max_iterations - 1)
        progress = iteration / T

        # Faz boyutlari
        n1 = max(1, int(self.pop_size * self.algo_params["phase1_ratio"]))
        n2 = max(1, int(self.pop_size * self.algo_params["phase2_ratio"]))
        n3 = self.pop_size - n1 - n2

        # En iyi (dominant erkek) ve sürü merkezi
        best_idx = int(np.argmin(self.f))
        dominant = self.population[best_idx].copy()
        herd_center = np.mean(self.population, axis=0)

        new_population = np.zeros_like(self.population)

        # FAZ 1 - River/Pond
        for i in range(n1):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            new_population[i] = self.population[i] + r1 * (
                dominant - self.population[i]
            ) + r2 * (herd_center - self.population[i])

        # FAZ 2 - Defensive against predator
        for i in range(n1, n1 + n2):
            predator = np.random.rand(self.dim)  # Rastgele yirtici pozisyonu
            dist = np.linalg.norm(predator - self.population[i])
            if dist > 0.3:
                # Yirtici uzak — karsilamak ve dairesele almak
                phi = np.random.rand(self.dim)
                new_population[i] = predator - phi * (
                    predator - self.population[i]
                )
            else:
                # Yirtici yakin — kacis
                escape = self.population[i] - np.random.rand(self.dim) * (
                    predator - self.population[i]
                )
                new_population[i] = escape

        # FAZ 3 - Evasion (local optima escape)
        for i in range(n1 + n2, self.pop_size):
            # Adaptive evasion: erken iter daha agresif, sonra yumusar
            scale = 0.5 * (1 - progress)
            perturbation = (np.random.rand(self.dim) - 0.5) * 2 * scale
            new_population[i] = self.population[i] + perturbation

        # Bound clip
        new_population = np.clip(new_population, 0.0, 1.0)
        self.population = new_population

        # Evaluate
        results = self.evaluate(
            vectors=[self.population[i] for i in range(self.pop_size)],
            indices=list(range(self.pop_size)),
            member_type="H",  # H = Hippopotamus
            iteration=iteration,
        )

        improvements = []
        for i, result in enumerate(results):
            self.f[i] = result["loss"]
            self.fitness[i] = result["fitness"]
            improved = self.update_global_best(i, iteration, "HO", result)
            improvements.append(improved)

        self.save_trials(iteration, "HO", results, improvements)
        duration = time.time() - iter_start
        self.save_iteration_stats(
            iteration, results, duration,
            extra={"phase1_n": n1, "phase2_n": n2, "phase3_n": n3, "progress": progress}
        )
        self.save_checkpoint(iteration, "ITER_COMPLETE")

        self.logger.info(
            f"\n[ITER {iteration+1}] Best: {self.global_opt:.6f} | "
            f"Phases: {n1}/{n2}/{n3} | Duration: {duration:.1f}s"
        )
