# -*- coding: utf-8 -*-
"""
optimizers/ga_optimizer.py — Genetic Algorithm Optimizer.

Holland (1975) tabanlı klasik GA, SBX crossover + Polynomial mutation +
Tournament selection + Elitism ile. BaseOptimizer'dan türer.

Operatörler:
- Tournament selection (boyut: tournament_size)
- SBX (Simulated Binary Crossover, Deb & Agrawal)
- Polynomial mutation (Deb & Goyal)
- Elitism: en iyi K bireyi koru

Algoritma-spesifik parametreler:
    crossover_rate: Çaprazlama olasılığı (klasik 0.8)
    mutation_rate: Mutasyon olasılığı (klasik 0.1)
    tournament_size: Turnuva büyüklüğü (klasik 3)
    elitism_count: Korunan elit birey sayısı (klasik 2)
    sbx_eta: SBX dağılım indeksi (klasik 20)
    pm_eta: Polynomial mutation dağılım indeksi (klasik 20)
"""

import time
import random
import numpy as np
from typing import Dict, Any, List

from optimizers import register_algorithm
from optimizers.base_optimizer import BaseOptimizer


@register_algorithm("ga")
class GAOptimizer(BaseOptimizer):
    """Klasik Genetic Algorithm.

    Her generation'da:
    1. Elitism: en iyi K bireyi yeni nesle aktar
    2. Geri kalan slotları doldurmak için:
       - Tournament selection ile iki ebeveyn seç
       - SBX crossover (olasılık: crossover_rate) ile çocuk üret
       - Polynomial mutation uygula
    3. Yeni nesli değerlendir, replace et
    """

    DEFAULT_PARAMS = {
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'tournament_size': 3,
        'elitism_count': 2,
        'sbx_eta': 20.0,
        'pm_eta': 20.0,
    }

    def _initialize_population(self) -> None:
        """Random popülasyon, sıfır fitness, sonsuz f."""
        self.population = np.random.rand(self.pop_size, self.dim)
        self.f = np.full(self.pop_size, float('inf'))
        # GA'da fitness = -loss (minimize); ama BaseOptimizer'a uyum için
        # standart fitness saklıyoruz
        self.fitness = np.zeros(self.pop_size)

    # =========================================================================
    # GA OPERATÖRLERİ
    # =========================================================================

    def _tournament_select(self) -> int:
        """Turnuva seçimi — kazanan birey indeksini döndürür.

        Loss'u en küçük olan kazanır.
        """
        k = self.algo_params['tournament_size']
        candidates = random.sample(range(self.pop_size), k)
        # En düşük loss'lu (en iyi) bireyi seç
        return min(candidates, key=lambda i: self.f[i])

    def _sbx_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """SBX (Simulated Binary Crossover).

        Deb & Agrawal (1995). Tek çocuk döndürür (rastgele iki çocuktan biri).
        """
        eta = self.algo_params['sbx_eta']
        child1 = np.zeros(self.dim)
        child2 = np.zeros(self.dim)

        for i in range(self.dim):
            if random.random() < 0.5 and abs(p1[i] - p2[i]) > 1e-10:
                if p1[i] < p2[i]:
                    y1, y2 = p1[i], p2[i]
                else:
                    y1, y2 = p2[i], p1[i]

                rand = random.random()
                beta = 1.0 + (2.0 * y1 / (y2 - y1 + 1e-10))
                alpha = 2.0 - beta ** (-(eta + 1))

                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                child1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                child2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
            else:
                child1[i] = p1[i]
                child2[i] = p2[i]

        # Rastgele birini döndür
        chosen = child1 if random.random() < 0.5 else child2
        return np.clip(chosen, 0.0, 1.0)

    def _polynomial_mutation(self, ind: np.ndarray) -> np.ndarray:
        """Polynomial mutation.

        Deb & Goyal (1996).
        """
        rate = self.algo_params['mutation_rate']
        eta = self.algo_params['pm_eta']
        mutant = ind.copy()

        for i in range(self.dim):
            if random.random() < rate:
                y = mutant[i]
                delta1 = y
                delta2 = 1.0 - y
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.0)

                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                    deltaq = 1.0 - val ** mut_pow

                mutant[i] = y + deltaq

        return np.clip(mutant, 0.0, 1.0)

    # =========================================================================
    # ANA ITERASYON: BIR GENERATION
    # =========================================================================

    def _run_iteration(self, iteration: int) -> None:
        """Tek bir generation."""
        gen_start = time.time()

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"GENERATION {iteration+1}/{self.max_iterations}")

        # 1. Elitism: en iyi K bireyi koru
        elite_count = self.algo_params['elitism_count']
        sorted_idx = np.argsort(self.f)
        elite_indices = sorted_idx[:elite_count]
        new_pop = [self.population[i].copy() for i in elite_indices]

        # 2. Geri kalan slotları crossover + mutation ile doldur
        while len(new_pop) < self.pop_size:
            p1_idx = self._tournament_select()
            p2_idx = self._tournament_select()
            p1 = self.population[p1_idx]
            p2 = self.population[p2_idx]

            if random.random() < self.algo_params['crossover_rate']:
                child = self._sbx_crossover(p1, p2)
            else:
                # Crossover yapılmazsa daha iyi ebeveyni doğrudan al
                child = p1.copy() if self.f[p1_idx] < self.f[p2_idx] else p2.copy()

            child = self._polynomial_mutation(child)
            new_pop.append(child)

        new_pop = np.array(new_pop[:self.pop_size])

        # 3. Yeni nesli değerlendir (elitleri tekrar değerlendirmeye gerek yok
        # ama fitness kaydı için hepsini koşalım — istatistik tutarlı olsun)
        results = self.evaluate(
            vectors=[new_pop[i] for i in range(self.pop_size)],
            indices=list(range(self.pop_size)),
            member_type="C",  # "C" = Child / next generation
            iteration=iteration,
        )

        # Replace + global best update
        improvements = []
        for i, result in enumerate(results):
            old_loss = self.f[i]
            self.population[i] = new_pop[i]
            self.f[i] = result['loss']
            self.fitness[i] = result['fitness']
            improved = result['loss'] < old_loss
            improvements.append(improved)
            self.update_global_best(i, iteration, 'GENERATION', result)

        self.save_trials(iteration, 'GENERATION', results, improvements)

        # Iteration sonu
        duration = time.time() - gen_start
        self.save_iteration_stats(
            iteration, results, duration,
            extra={
                'elite_count': elite_count,
                'improvements': sum(improvements),
            }
        )
        self.save_checkpoint(iteration, 'GENERATION_COMPLETE')

        self.logger.info(
            f"\n[GEN {iteration+1}] Best: {self.global_opt:.6f} | "
            f"Duration: {duration:.1f}s | "
            f"Improvements: {sum(improvements)}/{self.pop_size}"
        )
