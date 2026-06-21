# -*- coding: utf-8 -*-
"""
optimizers/raindrop_optimizer.py - Raindrop Optimizer.

Yagmur damlasinin fiziksel davranisindan esinlenmis metasezgisel.
Dort temel operator:

1. Splash Dispersion: Bir damla yere düşünce çevreye sıçrar (keşif).
2. Gravitational Flow: Damlalar daha düşük noktaya akar (sömürü).
3. Evaporation Dynamics: Düşük kaliteli damlalar buharlaşır (eleme).
4. Overflow Patterns: Sürü çok kalabalıksa sıçrayarak yayılır (diversite).

Referans: (Raindrop Optimizer 2025, Scientific Reports).
DOI: 10.1038/s41598-025-15832-w
"""

import time
import numpy as np
from typing import Dict, Any

from optimizers import register_algorithm
from optimizers.base_optimizer import BaseOptimizer


@register_algorithm("raindrop")
class RaindropOptimizer(BaseOptimizer):
    """Raindrop Optimizer (2025)."""

    DEFAULT_PARAMS = {
        "splash_radius": 0.15,    # Splash sıçrama yarıçapı (search space oranı)
        "flow_rate": 0.3,         # Gravitational flow attraction katsayısı
        "evap_threshold": 0.7,    # Buharlaşma eşiği (fitness percentile)
        "evap_rate": 0.1,         # Her iter buharlaşan damla oranı
    }

    def _initialize_population(self) -> None:
        self.population = np.random.rand(self.pop_size, self.dim)
        self.fitness = np.zeros(self.pop_size)
        self.f = np.full(self.pop_size, float("inf"))

    def _splash(self, position: np.ndarray, radius: float) -> np.ndarray:
        """Splash: bir pozisyondan rastgele yarıçapta yeni pozisyon."""
        offset = (np.random.rand(self.dim) - 0.5) * 2 * radius
        return np.clip(position + offset, 0.0, 1.0)

    def _gravitational_flow(self, position: np.ndarray,
                             attractor: np.ndarray, rate: float) -> np.ndarray:
        """Gravitational flow: pozisyon, attractor'a doğru çekilir."""
        r = np.random.rand(self.dim)
        return np.clip(
            position + rate * r * (attractor - position),
            0.0, 1.0
        )

    def _run_iteration(self, iteration: int) -> None:
        iter_start = time.time()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"RAINDROP ITERATION {iteration+1}/{self.max_iterations}")

        T = max(1, self.max_iterations - 1)
        progress = iteration / T

        # Adaptif splash radius - erken iter genis, sonra daralir
        splash_r = self.algo_params["splash_radius"] * (1 - 0.7 * progress)
        flow_rate = self.algo_params["flow_rate"]

        # En iyi (en alt nokta - gravitational sink)
        best_idx = int(np.argmin(self.f))
        sink = self.population[best_idx].copy()

        new_population = np.zeros_like(self.population)

        # OPERATOR 1+2: Her damla icin splash + gravitational flow
        for i in range(self.pop_size):
            # Splash (kesif): mevcut konumdan rastgele sıçra
            splashed = self._splash(self.population[i], splash_r)
            # Gravitational flow (sömürü): sink'e doğru çek
            flowed = self._gravitational_flow(splashed, sink, flow_rate)
            new_population[i] = flowed

        # OPERATOR 3: Evaporation - en zayıf damlalar yenilenir
        # (bu operator evaluation sonrasinda kullanilir, su anda yer tutar)

        self.population = new_population

        # Evaluate
        results = self.evaluate(
            vectors=[self.population[i] for i in range(self.pop_size)],
            indices=list(range(self.pop_size)),
            member_type="R",  # R = Raindrop
            iteration=iteration,
        )

        improvements = []
        for i, result in enumerate(results):
            self.f[i] = result["loss"]
            self.fitness[i] = result["fitness"]
            improved = self.update_global_best(i, iteration, "RAINDROP", result)
            improvements.append(improved)

        # OPERATOR 3: Evaporation (post-evaluation)
        # En kötü %evap_rate'lik kısmı yeniden başlat (random)
        evap_rate = self.algo_params["evap_rate"]
        n_evap = max(0, int(self.pop_size * evap_rate))
        if n_evap > 0:
            worst_indices = np.argsort(self.f)[-n_evap:]
            for idx in worst_indices:
                self.population[idx] = np.random.rand(self.dim)
                self.f[idx] = float("inf")  # Sonraki iterde yeniden değerlendirilecek

        # OPERATOR 4: Overflow - sürü çok yoğunsa rastgele dağıt
        # (basit implementasyon: en yakın 2 damla varsa birini sıçrat)
        for i in range(self.pop_size):
            min_dist = float("inf")
            for j in range(self.pop_size):
                if i == j:
                    continue
                d = np.linalg.norm(self.population[i] - self.population[j])
                if d < min_dist:
                    min_dist = d
            # Eğer çok yakın (overflow) → splash
            if min_dist < 0.05:
                self.population[i] = self._splash(self.population[i],
                                                    splash_r * 2)

        self.save_trials(iteration, "RAINDROP", results, improvements)
        duration = time.time() - iter_start
        self.save_iteration_stats(
            iteration, results, duration,
            extra={
                "splash_radius": float(splash_r),
                "evaporated": int(n_evap),
                "progress": progress,
            }
        )
        self.save_checkpoint(iteration, "ITER_COMPLETE")

        self.logger.info(
            f"\n[ITER {iteration+1}] Best: {self.global_opt:.6f} | "
            f"splash_r={splash_r:.3f} | evap={n_evap} | Duration: {duration:.1f}s"
        )
