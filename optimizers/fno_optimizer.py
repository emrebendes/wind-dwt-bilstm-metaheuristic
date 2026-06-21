# -*- coding: utf-8 -*-
"""
optimizers/fno_optimizer.py - Farthest better / Nearest worse Optimizer.

Metaphor-LESS bir metasezgisel - hiçbir hayvan/fizik metaforu kullanmaz,
saf matematik. Her birey için iki referans noktası seçer:

- FB (Farthest Better): Daha iyi fitness'a sahip, en uzak birey
- NW (Nearest Worse):   Daha kötü fitness'a sahip, en yakın birey

Sonra birey FB'ye yaklaştırılır, NW'den uzaklaştırılır.
Sömürü fazında Dynamic Focus Strategy (DFS) ile arama uzayı daraltılır.

Referans: (FNO 2025), Artificial Intelligence Review, Springer.
DOI: 10.1007/s10462-025-11443-z
"""

import time
import numpy as np
from typing import Dict, Any, Optional, Tuple

from optimizers import register_algorithm
from optimizers.base_optimizer import BaseOptimizer


@register_algorithm("fno")
class FNOOptimizer(BaseOptimizer):
    """Farthest better / Nearest worse Optimizer (2025).

    Metaphor-less, distance-based dual-reference algorithm.
    """

    DEFAULT_PARAMS = {
        # Keşif fazından sömürüye geçiş eşiği (iter_progress)
        "exploration_ratio": 0.5,
        # Dynamic Focus Strategy daraltma katsayısı
        "dfs_shrink": 0.7,
        # Step size
        "step_min": 0.1,
        "step_max": 0.6,
    }

    def _initialize_population(self) -> None:
        self.population = np.random.rand(self.pop_size, self.dim)
        self.fitness = np.zeros(self.pop_size)
        self.f = np.full(self.pop_size, float("inf"))

    def _find_fb_nw(self, idx: int) -> Tuple[Optional[int], Optional[int]]:
        """Birey idx için FB ve NW indekslerini bul."""
        my_f = self.f[idx]
        my_pos = self.population[idx]

        fb_idx = None
        fb_dist = -1.0
        nw_idx = None
        nw_dist = float("inf")

        for j in range(self.pop_size):
            if j == idx:
                continue
            dist = np.linalg.norm(my_pos - self.population[j])
            if self.f[j] < my_f:
                # daha iyi → FB adayı; en uzak olanı seç
                if dist > fb_dist:
                    fb_dist = dist
                    fb_idx = j
            elif self.f[j] > my_f:
                # daha kötü → NW adayı; en yakın olanı seç
                if dist < nw_dist:
                    nw_dist = dist
                    nw_idx = j
            # eşit olanlar göz ardı edilir

        return fb_idx, nw_idx

    def _run_iteration(self, iteration: int) -> None:
        iter_start = time.time()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"FNO ITERATION {iteration+1}/{self.max_iterations}")

        T = max(1, self.max_iterations - 1)
        progress = iteration / T

        # Faz tespiti: keşif mi sömürü mü?
        is_exploration = progress < self.algo_params["exploration_ratio"]

        # Step size: keşifte büyük, sömürüde küçük (DFS)
        step_min = self.algo_params["step_min"]
        step_max = self.algo_params["step_max"]
        step = step_max - (step_max - step_min) * progress

        # Sömürü fazında DFS daraltması uygula
        if not is_exploration:
            shrink_progress = (progress - self.algo_params["exploration_ratio"]) / (
                1 - self.algo_params["exploration_ratio"]
            )
            step *= self.algo_params["dfs_shrink"] ** shrink_progress

        new_population = np.zeros_like(self.population)

        for i in range(self.pop_size):
            fb_idx, nw_idx = self._find_fb_nw(i)
            current = self.population[i]

            # Move toward FB, away from NW
            move = np.zeros(self.dim)

            if fb_idx is not None:
                fb = self.population[fb_idx]
                # Yön: FB'ye doğru
                direction_fb = fb - current
                norm = np.linalg.norm(direction_fb)
                if norm > 1e-10:
                    direction_fb = direction_fb / norm
                r1 = np.random.rand(self.dim)
                move += r1 * step * direction_fb

            if nw_idx is not None:
                nw = self.population[nw_idx]
                # Yön: NW'den uzak
                direction_nw = current - nw
                norm = np.linalg.norm(direction_nw)
                if norm > 1e-10:
                    direction_nw = direction_nw / norm
                r2 = np.random.rand(self.dim)
                move += r2 * step * 0.5 * direction_nw  # NW kaçışı yarı ağırlık

            # Eğer hiçbiri yoksa (en iyi/en kötü birey) → rastgele perturbation
            if fb_idx is None and nw_idx is None:
                move = (np.random.rand(self.dim) - 0.5) * 2 * step

            new_population[i] = np.clip(current + move, 0.0, 1.0)

        self.population = new_population

        # Evaluate
        results = self.evaluate(
            vectors=[self.population[i] for i in range(self.pop_size)],
            indices=list(range(self.pop_size)),
            member_type="F",  # F = FNO agent
            iteration=iteration,
        )

        improvements = []
        for i, result in enumerate(results):
            self.f[i] = result["loss"]
            self.fitness[i] = result["fitness"]
            improved = self.update_global_best(i, iteration, "FNO", result)
            improvements.append(improved)

        self.save_trials(iteration, "FNO", results, improvements)
        duration = time.time() - iter_start
        self.save_iteration_stats(
            iteration, results, duration,
            extra={
                "phase": "exploration" if is_exploration else "exploitation",
                "step": float(step),
                "progress": progress,
            }
        )
        self.save_checkpoint(iteration, "ITER_COMPLETE")

        self.logger.info(
            f"\n[ITER {iteration+1}] Best: {self.global_opt:.6f} | "
            f"step={step:.3f} | "
            f"phase={'EXPL' if is_exploration else 'EXPLT'} | "
            f"Duration: {duration:.1f}s"
        )
