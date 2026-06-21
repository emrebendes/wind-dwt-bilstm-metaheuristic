# `optimizers/` — Generic Metaheuristic Optimizer Framework

Bu paket, VMD-TCAN hiperparametre optimizasyonu için **birden fazla metasezgisel algoritmayı ortak bir iskelet altında** çalıştıran modüller içerir. Eski ABC ve GA scriptlerinin yerini alır (eski dosyalar projede dokunulmadan kalır).

## Mimari

```
optimizers/
├── __init__.py            # Algoritma registry (string ID → sınıf)
├── base_optimizer.py      # BaseOptimizer (soyut sınıf)
├── db_manager.py          # GenericDBManager (SQLite + WAL + checkpoint)
├── evaluator.py           # Paralel fitness wrapper (multiprocessing.Pool)
├── param_mapping.py       # [0,1]^10 → VMD-TCAN parametreleri
├── abc_optimizer.py       # Artificial Bee Colony
├── ga_optimizer.py        # Genetic Algorithm
└── README.md              # Bu dosya

run_optimizer.py           # Üst dizinde — birleşik CLI
```

## Tasarım ilkeleri

1. **Tek arayüz, çok algoritma:** Tüm algoritmalar `BaseOptimizer`'dan türer. Yeni algoritma eklemek = yeni dosya + decorator.
2. **TRUBA 3-gün limit'ine dayanıklı:** Her cycle/generation sonu SQLite checkpoint. Çalışma kesilirse `python run_optimizer.py --algo X --db ...` ile aynı komut kaldığı yerden devam eder.
3. **Equal computational budget:** Tüm algoritmalar `pop_size × max_iterations` kadar fitness evaluation alır (default: 40 × 50 = 2000). Adil karşılaştırma protokolü.
4. **Algoritma-spesifik state:** Pickle BLOB olarak `checkpoint.state_blob` kolonunda. Yeni algoritma eklerken DB schema değişmez.
5. **Reproducibility:** Seed yönetimi merkezi; aynı seed ile aynı sonuç.

## Kullanım

```bash
# Algoritma listele
python run_optimizer.py --list-algos

# ABC çalıştır
python run_optimizer.py --algo abc --seed 42 --db abc_runs/run_001/abc_running.db

# GA çalıştır
python run_optimizer.py --algo ga --seed 42 --db ga_runs/run_001/ga_running.db

# Algoritma-spesifik parametre override (ABC limit duyarlılık analizi)
python run_optimizer.py --algo abc --algo-params '{"limit": 7}' --seed 42 --db ...

# Mevcut DB'yi analiz et
python run_optimizer.py --analyze --algo abc --db abc_runs/run_001/abc_running.db
```

TRUBA SLURM scriptlerinde sadece komutu değiştirin:
```bash
# Eski: python run_abc_optimization_v2.py --seed $SEED --db $DB
# Yeni: python run_optimizer.py --algo abc --seed $SEED --db $DB
```

## Yeni algoritma eklemek (rehber)

Diyelim ki PSO ekliyoruz. 4 adımda biter:

### 1) `optimizers/pso_optimizer.py` dosyası yaz

```python
import numpy as np
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
        'v_max': 0.2,      # Hız sınırı (search space'in yüzdesi)
    }

    def _initialize_population(self):
        self.population = np.random.rand(self.pop_size, self.dim)
        self.fitness = np.zeros(self.pop_size)
        self.f = np.full(self.pop_size, float('inf'))
        # PSO-spesifik state
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.pbest = self.population.copy()
        self.pbest_f = np.full(self.pop_size, float('inf'))

    def _get_state_dict(self):
        d = super()._get_state_dict()
        d['velocities'] = self.velocities
        d['pbest'] = self.pbest
        d['pbest_f'] = self.pbest_f
        return d

    def _set_state_dict(self, state):
        super()._set_state_dict(state)
        self.velocities = state['velocities']
        self.pbest = state['pbest']
        self.pbest_f = state['pbest_f']

    def _run_iteration(self, iteration):
        # PSO velocity & position update
        w = self.algo_params['w_start'] - (
            (self.algo_params['w_start'] - self.algo_params['w_end'])
            * iteration / self.max_iterations
        )
        c1, c2 = self.algo_params['c1'], self.algo_params['c2']

        for i in range(self.pop_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.velocities[i] = (
                w * self.velocities[i]
                + c1 * r1 * (self.pbest[i] - self.population[i])
                + c2 * r2 * (self.global_vector - self.population[i])
            )
            v_max = self.algo_params['v_max']
            self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)
            self.population[i] = np.clip(
                self.population[i] + self.velocities[i], 0.0, 1.0
            )

        # Evaluate
        results = self.evaluate(
            vectors=[self.population[i] for i in range(self.pop_size)],
            indices=list(range(self.pop_size)),
            member_type="P",
            iteration=iteration,
        )

        improvements = []
        for i, result in enumerate(results):
            self.f[i] = result['loss']
            self.fitness[i] = result['fitness']
            improved = result['loss'] < self.pbest_f[i]
            improvements.append(improved)
            if improved:
                self.pbest[i] = self.population[i].copy()
                self.pbest_f[i] = result['loss']
                self.update_global_best(i, iteration, 'PSO', result)

        self.save_trials(iteration, 'PSO', results, improvements)
        # ... save_iteration_stats + save_checkpoint
```

### 2) `optimizers/__init__.py` içinde import et

`_ensure_registered()` fonksiyonunda:
```python
from optimizers import pso_optimizer  # noqa: F401
```

### 3) Smoke test yaz

```bash
python run_optimizer.py --algo pso --seed 42 --pop-size 5 --max-iter 3 \
    --db /tmp/pso_smoke.db
```

3-5 dakikada biter. DB'de checkpoint, trial_history, best_history dolu olmalı.

### 4) Üretim koşusu

```bash
python run_optimizer.py --algo pso --seed 42 --db pso_runs/run_001/pso_running.db --workers 35
```

## DB Şeması (özet)

Tüm algoritmalar aynı şemayı kullanır:

| Tablo | Amaç |
|---|---|
| `metadata` | Tek satır: algoritma, status, seed, pop_size, max_iter, config_json |
| `checkpoint` | Tek satır: iter, phase, state_blob (pickle), global_opt, global_vector |
| `trial_history` | Her bir fitness evaluation kaydı (parametreler ayrı kolonlarda) |
| `best_history` | Her global best iyileşmesi |
| `iteration_stats` | Iterasyon başına özet (best/worst/mean loss, duration) |

`extra_json` kolonu (iteration_stats) algoritma-spesifik bilgiyi tutar (ör. ABC için `scouts_triggered`, GA için `improvements`).

## Sequential Mode (deneysel)

`--mode sequential` ile çalıştırıldığında, optimizer iki aşamalı bir protokol uygulayacak: önce VMD parametreleri (envelope entropy fitness), sonra TCAN hiperparametreleri (RMSE fitness). Mevcut implementasyonda mode bayrağı sadece logging amaçlı; gerçek sequential mantığı `SequentialOptimizer` wrapper'ında uygulanacak (next phase).

## Test ve Doğrulama

`smoke_test.py` (proje kökünde) ile yeni iskeletin eski sonuçları reproduce ettiği doğrulanır. Aynı seed ile çalıştırıldığında ilk birkaç trial'ın eski DB'deki ilk trial'larla bire bir aynı olması beklenir.

## Bakım Notları

- **Yeni algoritma eklerken** `__init__.py` registry'sine import eklemeyi unutma
- **DB schema değişikliği** gerekiyorsa migration yaz; doğrudan değiştirme (eski DB'ler bozulur)
- **Algoritma-spesifik state**'i `_get_state_dict()` / `_set_state_dict()` ile pickle BLOB'a koy; sütun ekleme
- **Equal budget**: Tüm algoritmalar `pop_size × max_iterations` evaluation alır; bu ortak değerleri değiştirme
