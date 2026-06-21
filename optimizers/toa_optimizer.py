# -*- coding: utf-8 -*-
"""
optimizers/toa_optimizer.py - Tuckman Optimization Algorithm.

Sosyal-davranis tabanli, 2026'da onerilmis metasezgisel. Tuckman'in Grup
Gelisim Asamalari teorisini (Forming, Storming, Norming, Performing,
Adjourning) optimizasyon framework'une gomer.

Bes asama:
    1. Forming   : Uyeler tanisiyor, kucuk takimlarla (3-5 kisi) ogrenir
    2. Storming  : Catismalar + lider rehberligi (X_best + peer Xp)
    3. Norming   : Konsensus ve normlar (std faktoru + U2 ile gecis)
    4. Performing: Yuksek verimlilik (Norming devami)
    5. Adjourning: Kapanis - logistic growth ile yetersiz uyelerin yenilenmesi

Referans: Shi, M., & Wang, F. (2026). Tuckman Optimization Algorithm:
A novel metaheuristic inspired by Tuckman's Stages of Group Development.
Journal of Computing and Electronic Information Management, 21(1), 45-58.

Bu calisma, TOA'nin VMD-TCAN hiperparametre optimizasyonuna ilk
uygulamasini sunar (literature gap claim).
"""

import time
import numpy as np
from typing import Dict, Any

from optimizers import register_algorithm
from optimizers.base_optimizer import BaseOptimizer


@register_algorithm("toa")
class TOAOptimizer(BaseOptimizer):
    """Tuckman Optimization Algorithm (Shi & Wang, 2026).

    Sosyal-davranissal metasezgisel; takim gelisim asamalari uzerinden
    populasyonun kesif/somurum dengesini saglar.
    """

    DEFAULT_PARAMS = {
        # Hiçbiri orijinal makalede parametre olarak ayarlanmamis;
        # asagidakiler dahili katsayilardir, sabit kalir.
        "subteam_sizes": [3, 4, 5],   # numTm: alt-takim boyutu (her biri 1/3 olasilik)
        "ci_scale": 0.2,              # CI coordination index = (1/5)
        "logistic_k_max": 1.0,        # MAXF'e ulasinca k -> 1
    }

    # =========================================================================
    # POPÜLASYON BAŞLATMA
    # =========================================================================

    def _initialize_population(self) -> None:
        """Rastgele uye pozisyonlari + TP (team preparation) takipcisi."""
        self.population = np.random.rand(self.pop_size, self.dim)
        self.fitness = np.zeros(self.pop_size)
        self.f = np.full(self.pop_size, float("inf"))
        # TOA-spesifik state
        # PIM = team-preparation member count (kac uye yetersizlikten cikti)
        self.pim = 0.0  # baslangic
        # Onceki iterasyonun "Xchange" degerleri (Forming icin)
        # Basitlik: ilk iter 0
        self.prev_pop = self.population.copy()

    # =========================================================================
    # STATE SERIALIZATION
    # =========================================================================

    def _get_state_dict(self) -> Dict[str, Any]:
        d = super()._get_state_dict()
        d["pim"] = self.pim
        d["prev_pop"] = self.prev_pop
        return d

    def _set_state_dict(self, state: Dict[str, Any]) -> None:
        super()._set_state_dict(state)
        self.pim = state.get("pim", 0.0)
        self.prev_pop = state.get("prev_pop", self.population.copy())

    # =========================================================================
    # YARDIMCI HESAPLAMALAR (Eq. 5-17 paper)
    # =========================================================================

    def _calc_TP(self, g: int) -> float:
        """Team Preparation progress (Eq. 18, simplified): TP = g / MAXF."""
        return min(1.0, g / max(1, self.max_iterations))

    def _calc_KL(self, TP: float) -> float:
        """Learning factor (Eq. 6).

        KL = sign(r1) * sin(pi/2 * sqrt(r2) * TP), r1 ~ N(0,1), r2 ~ U(0,1)
        """
        r1 = np.random.randn()
        r2 = np.random.rand()
        return np.sign(r1) * np.sin(np.pi / 2.0 * np.sqrt(r2) * TP)

    def _calc_CI(self, g: int, TP: float) -> float:
        """Coordination Index (Eq. 10).

        CI = (1/5) * sign(r5) * TP * sin(pi/2 * sqrt(g/MAXF))
        """
        r5 = np.random.randn()
        ratio = g / max(1, self.max_iterations)
        return (
            self.algo_params["ci_scale"]
            * np.sign(r5)
            * TP
            * np.sin(np.pi / 2.0 * np.sqrt(ratio))
        )

    def _calc_U1(self, g: int) -> int:
        """U1 control factor (Eq. 11): 1 if r6 * (g/MAXF) >= r7 else 0."""
        r6 = np.random.rand()
        r7 = np.random.rand()
        return int(r6 * (g / max(1, self.max_iterations)) >= r7)

    def _calc_U2(self, g: int) -> int:
        """U2 control factor (Eq. 14): 1 if g/MAXF >= r9 else 0."""
        r9 = np.random.rand()
        return int(g / max(1, self.max_iterations) >= r9)

    def _pick_subteam_size(self) -> int:
        """numTm: alt-takim boyutu, {3,4,5}'ten esit olasilikla."""
        return int(np.random.choice(self.algo_params["subteam_sizes"]))

    def _logistic_PIM_update(self, g: int) -> float:
        """Logistic growth model ile PIM guncellemesi (Eq. 17).

        PIM(t+1) = k * NP / (1 + ((NP - PIM(t))/PIM(t)) * exp(-TP*g))
        """
        TP = self._calc_TP(g)
        # k: orijinalde sqrt(1 - (MAXF-g)/MAXF)^2 = g/MAXF formu, basit hali
        k = (g / max(1, self.max_iterations)) * self.algo_params["logistic_k_max"]
        k = min(self.algo_params["logistic_k_max"], k + 0.05)

        pim = max(1e-3, self.pim)
        try:
            exp_arg = -TP * g
            exp_arg = np.clip(exp_arg, -50, 50)
            denom = 1.0 + ((self.pop_size - pim) / pim) * np.exp(exp_arg)
            new_pim = k * self.pop_size / max(1e-6, denom)
            return float(np.clip(new_pim, 0.0, float(self.pop_size)))
        except OverflowError:
            return float(self.pop_size)

    # =========================================================================
    # ANA ITERASYON: Forming / Storming / Norming
    # =========================================================================

    def _run_iteration(self, iteration: int) -> None:
        iter_start = time.time()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"TOA ITERATION {iteration+1}/{self.max_iterations}")

        g = iteration + 1
        TP = self._calc_TP(g)

        # En iyi (lider/supervisor)
        best_idx = int(np.argmin(self.f))
        x_best = self.population[best_idx].copy()

        new_population = np.zeros_like(self.population)

        # Asama secimi: U2 ile Forming->Storming, U2=1 sonra Norming/Performing
        # Basit yapilandirma:
        # - Erken iterasyonlar (g/MAXF < 0.4): Forming
        # - Orta (0.4 <= g/MAXF < 0.7): Storming
        # - Geç (g/MAXF >= 0.7): Norming/Performing
        progress = g / max(1, self.max_iterations)

        for i in range(self.pop_size):
            current = self.population[i].copy()
            U1 = self._calc_U1(g)
            U2 = self._calc_U2(g)
            KL = self._calc_KL(TP)
            CI = self._calc_CI(g, TP)

            if progress < 0.4:
                # ============ FORMING (Eq. 3-7) ============
                # Alt-takim sec, ortalama hesapla
                k = self._pick_subteam_size()
                k = min(k, self.pop_size - 1)
                indices = np.random.choice(
                    [j for j in range(self.pop_size) if j != i],
                    size=k, replace=False
                )
                X_change_mean = np.mean(self.prev_pop[indices], axis=0)
                # Boyut secimi (rastgele bir boyutta degisim)
                dim_change = np.random.randint(0, self.dim)
                # Yeni pozisyon: bireyin pozisyonu + KL * (alt-takim ortalamasi - kendisi)
                new_x = current.copy()
                new_x[dim_change] = current[dim_change] + KL * (
                    X_change_mean[dim_change] - current[dim_change]
                )

            elif progress < 0.7:
                # ============ STORMING (Eq. 9-11) ============
                # X(t)_i = X(t)_i * U1 + (X(t)_i + CI*(X(t)_i - Xp)) * (1-U1)
                # Xp: rastgele peer
                p_idx = int(np.random.choice(
                    [j for j in range(self.pop_size) if j != i]
                ))
                xp = self.population[p_idx]
                conflict_term = current + CI * (current - xp)
                leader_term = current + 0.5 * (x_best - current)  # lider rehberligi
                new_x = leader_term * U1 + conflict_term * (1 - U1)

            else:
                # ============ NORMING / PERFORMING (Eq. 12-14) ============
                # std-faktoru + U2 ile gecis
                # Norming: konsensus + lider; Performing: yuksek verimli local search
                std_pop = np.std(self.population, axis=0)
                if U2 == 1:
                    # Norming: konsensus
                    consensus = np.mean(self.population, axis=0)
                    new_x = current + 0.3 * (consensus - current) + 0.3 * (
                        x_best - current
                    ) + 0.1 * std_pop * np.random.randn(self.dim)
                else:
                    # Performing: best'in etrafinda dar arama
                    new_x = x_best + 0.1 * (1 - progress) * np.random.randn(self.dim)

            new_population[i] = np.clip(new_x, 0.0, 1.0)

        # prev_pop guncelle (Forming icin)
        self.prev_pop = self.population.copy()
        self.population = new_population

        # ============ Evaluate ============
        results = self.evaluate(
            vectors=[self.population[i] for i in range(self.pop_size)],
            indices=list(range(self.pop_size)),
            member_type="T",  # T = Team member
            iteration=iteration,
        )

        improvements = []
        for i, result in enumerate(results):
            self.f[i] = result["loss"]
            self.fitness[i] = result["fitness"]
            improved = self.update_global_best(i, iteration, "TOA", result)
            improvements.append(improved)

        # ============ TEAM RECONFIGURATION (Eq. 16-17) ============
        # PIM = logistic growth ile artar; LNP = NP - PIM kadar uye yenilenir
        self.pim = self._logistic_PIM_update(g)
        lnp = max(0, int(self.pop_size - self.pim))
        # En kotu LNP uyeyi rastgele yenile (boylece kesif korunur)
        if lnp > 0 and lnp < self.pop_size:
            worst_idx = np.argsort(self.f)[-lnp:]
            for idx in worst_idx:
                self.population[idx] = np.random.rand(self.dim)
                self.f[idx] = float("inf")  # bir sonraki iter yeniden evaluate edilir

        # ============ Save ============
        self.save_trials(iteration, "TOA", results, improvements)
        duration = time.time() - iter_start
        self.save_iteration_stats(
            iteration, results, duration,
            extra={
                "stage": "FORMING" if progress < 0.4
                         else "STORMING" if progress < 0.7
                         else "NORMING/PERFORMING",
                "pim": float(self.pim),
                "lnp": int(lnp),
                "TP": float(TP),
            }
        )
        self.save_checkpoint(iteration, "ITER_COMPLETE")

        self.logger.info(
            f"\n[ITER {iteration+1}] Best: {self.global_opt:.6f} | "
            f"stage={'F' if progress < 0.4 else 'S' if progress < 0.7 else 'N/P'} | "
            f"PIM={self.pim:.1f}, LNP={lnp} | Duration: {duration:.1f}s"
        )
