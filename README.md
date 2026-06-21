# Joint Hyperparameter Optimization of DWT–BiLSTM for Wind Speed Forecasting

Companion code and results for the manuscript:

> Bendeş, E. (2026). *Joint Hyperparameter Optimization of a DWT–BiLSTM
> Architecture for Wind Speed Forecasting: A Systematic Comparison of
> Metaheuristic Algorithms with Component Ablation Analysis.*
> Submitted to **Knowledge-Based Systems** (Elsevier).

This repository contains the complete experimental code, configuration
files, per-iteration trial histories, and analysis outputs for a
four-phase study that benchmarks **eight metaheuristic algorithms** on
the joint nine-dimensional hyperparameter optimization of a DWT–BiLSTM
hybrid wind-speed-forecasting architecture.

---

## Highlights

- Joint 9-dimensional hyperparameter optimization benchmarks **8
  metaheuristics** (ABC, GA, PSO, GWO, HO, FNO, Raindrop, TOA) under an
  identical 2,000-evaluation budget with 30 independent seeded runs each.
- **Grey Wolf Optimizer** ranks first (Friedman χ² = 49.76; p < 10⁻⁸).
- **Joint** optimization dominates the **sequential** alternative across
  all 90 paired runs (Cliff δ = 1.00; p < 10⁻⁹) with a 39% median RMSE
  gain.
- Component ablation shows the **DWT** layer is essential (Cohen's
  d = 13.09) while bidirectionality contributes negligibly at the
  one-hour horizon (p = 0.674).
- Persistence skill score = **0.4364**; GWO is Pareto-dominant in
  accuracy and run-time.

---

## Repository Structure

```
wind_optimization/
├── README.md                         ← this file
├── requirements.txt                  ← Python dependencies
├── LICENSE                           ← MIT (code only)
│
├── ── CORE MODULES ─────────────────────────────────────────────
├── config.py                         ← paths, train/test split, device
├── utils.py                          ← DWT, normalize, dataset, logger
├── model.py                          ← BiLSTM model definition
├── objective.py                      ← unified evaluation objective
├── optimizers/                       ← 8 metaheuristic implementations
│   ├── base_optimizer.py             ← common parent class
│   ├── db_manager.py                 ← trial-history SQLite layer
│   ├── evaluator.py                  ← parallel surrogate evaluator
│   ├── param_mapping.py              ← 9-D HP encoding/decoding
│   ├── abc_optimizer.py              ← Artificial Bee Colony
│   ├── ga_optimizer.py               ← Genetic Algorithm
│   ├── pso_optimizer.py              ← Particle Swarm Optimization
│   ├── gwo_optimizer.py              ← Grey Wolf Optimizer
│   ├── ho_optimizer.py               ← Hippopotamus Optimization
│   ├── fno_optimizer.py              ← Farthest-better/Nearest-worse Optimizer
│   ├── raindrop_optimizer.py         ← Raindrop Optimizer
│   └── toa_optimizer.py              ← Tuckman Optimization Algorithm
│
├── ── ENTRY POINTS ─────────────────────────────────────────────
├── run_optimizer.py                  ← single optimizer run (CLI)
├── manage_multiple_runs_v4.py        ← orchestrate 30-run campaigns
├── final_training_v4.py              ← GPU final training + analysis
├── smoke_test_optimizers.py          ← quick integration check
│
├── ── PHASE-SPECIFIC SCRIPTS ───────────────────────────────────
├── compare_algorithms.py             ← Phase 1: aggregate & test
├── analyze_convergence_diversity_timing.py
│                                      ← Phase 1: convergence / diversity / wall-time
├── dwt_param_selector.py             ← Phase 2: Shannon entropy DWT pick
├── sequential_with_metaheuristic.py  ← Phase 2: orchestrator + joint-vs-seq stats
├── ablation_runner.py                ← Phase 3: nodwt / nobidir orchestrator
├── compare_ablations.py              ← Phase 3: paired Wilcoxon & figures
├── walk_forward_eval.py              ← Phase 4: per-window HP search + training
├── analyze_walk_forward.py           ← Phase 4: cross-window consistency
├── compute_persistence_baseline.py   ← naive persistence + skill score
│
├── ── EXPERIMENTAL DATA (BY PHASE) ─────────────────────────────
├── faz1_comparison_runs/             ← Phase 1: 8 algos × 30 runs
│   ├── {algo}_runs/run_NNN/          ← config.json + trial_history.db
│   └── {algo}_final_results/         ← analysis_summary.json + final RMSE
│
├── faz1_comparison_results/          ← Phase 1: aggregated outputs
│   ├── aggregate_summary.json
│   ├── persistence_results.json      ← skill score 0.4364
│   ├── tables/                       ← LaTeX tables
│   ├── figures/                      ← convergence / diversity / boxplot
│   └── parameter_diversity_analyses/
│
├── faz2_sequential_runs/             ← Phase 2: sequential HP search
│   ├── {algo}_seq/                   ← per-algo run trees
│   ├── {algo}_seq_analysis.json
│   └── {algo}_seq_final_*/
│
├── faz2_sequential_compare/          ← Phase 2: joint-vs-sequential
│   ├── joint_vs_sequential_summary.json
│   └── joint_vs_sequential_boxplot.png
│
├── faz3_ablation_runs/               ← Phase 3: GWO ablation
│   ├── gwo_nodwt/                    ← DWT layer disabled
│   ├── gwo_nobidir/                  ← unidirectional LSTM
│   └── *_analysis.json
│
├── faz3_ablation_results/            ← Phase 3: figures & tables
│   ├── ablation_summary.json
│   └── figures/
│
├── faz4_walk_forward_runs/           ← Phase 4: GWO × 5 annual windows
│   ├── gwo_Y2/ ... gwo_Y6/
│   └── gwo_Y*_analysis.json
│
├── faz4_walk_forward_compare/        ← Phase 4: cross-window consistency
│   ├── gwo_summary.json
│   ├── gwo_boxplot.png
│   └── gwo_hp_table.csv
│
└── data_files/                       ← NOT INCLUDED (see Data Availability)
```

---

## Experimental Protocol (Four Phases)

| Phase | Question | Method | Key Output |
|-------|----------|--------|------------|
| **Phase 1** | Which of 8 metaheuristics finds the lowest RMSE? | 8 algos × 30 seeded runs; joint 9-D search | Friedman ranking, GWO wins |
| **Phase 2** | Is joint optimization better than sequential? | 3 algos × 30 paired runs; Shannon-entropy DWT pick then BiLSTM search | Cliff δ = 1.00; joint dominates |
| **Phase 3** | Which architectural component carries the gain? | GWO + ablations: nodwt, nobidir | DWT essential (d = 13.09), bidirectionality marginal (p = 0.674) |
| **Phase 4** | Is single-year HP search robust across years? | GWO re-run on Y2–Y6 | Cross-window variance 30–45% → ≤7% after retraining |

Each phase has its own `faz{N}_*_runs/` (raw experimental output) and
`faz{N}_*_compare/` or `faz{N}_*_results/` (aggregated analysis).

---

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch (CUDA optional but recommended for `final_training_v4.py`)
- Linux/macOS preferred; Windows works with WSL or native Anaconda

### Setup

```bash
git clone https://github.com/<your-account>/wind_optimization.git
cd wind_optimization
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Reproducing the Analyses (no GPU required)

If you only want to reproduce the **tables and figures** in the paper
from the existing experimental output, no GPU and no re-running of
optimization is required:

```bash
# Phase 1 — aggregate 8-algo benchmark + statistical tests
python compare_algorithms.py

# Phase 1 — convergence, diversity, and wall-time analyses
python analyze_convergence_diversity_timing.py

# Phase 1 — persistence baseline & skill score
python compute_persistence_baseline.py

# Phase 2 — joint vs sequential paired statistics
python sequential_with_metaheuristic.py --analyze --algos gwo,pso,ga

# Phase 3 — ablation comparison
python compare_ablations.py --algo gwo --ablations nodwt,nobidir

# Phase 4 — walk-forward consistency tests
python analyze_walk_forward.py --algo gwo --windows Y2,Y3,Y4,Y5,Y6
```

### Re-running the Optimization Itself (HPC recommended)

A full re-run of all four phases involves **~480,000 surrogate
evaluations** and was conducted on the
[TÜBİTAK ULAKBİM TRUBA cluster](https://www.truba.gov.tr/) using 36-core
nodes. On a single workstation a full re-run may take days. To run a
single Phase-1 campaign for one algorithm:

```bash
# Smoke test (one short run on each algo)
python smoke_test_optimizers.py

# Single campaign: 30 GWO runs
python manage_multiple_runs_v4.py --algo gwo --n-runs 30 \
       --pop-size 40 --max-iter 50 --workers 35
```

Per-trial histories are written to SQLite databases under
`faz1_comparison_runs/{algo}_runs/run_NNN/{algo}_running.db` so that
incomplete runs can be resumed and analysed without loss of data.

---

## Data Availability

> ⚠️ **Raw wind-speed time series are NOT included in this repository.**

The hourly wind speed observations used in the study were obtained from
the **Turkish State Meteorological Service (Meteoroloji Genel Müdürlüğü,
MGM)** under an institutional research-use license. The license granted
to the corresponding author covers **research use only** and **does not
permit redistribution**.

Researchers who wish to reproduce the experiments end-to-end on the
identical dataset are kindly directed to contact MGM directly through
their official data request channel
([https://mgm.gov.tr](https://mgm.gov.tr)) to obtain a comparable
license. The eight Nevşehir stations used in this study are: Acıgöl,
Avanos, Derinkuyu, Gülşehir, Hacıbektaş, Kozaklı, Nevşehir, and Ürgüp.
Time range: 51,144 hourly observations per station (≈ 5.84 years).

The repository's analysis scripts and aggregated result JSONs allow
**verification of every published number, figure, and table** without
access to the raw data.

### Expected Data Layout

If you obtain the data from MGM independently, place it under:

```
data_files/
├── all_station_data.npy              # shape (8, 51144); float32 m/s
├── ACIGÖL_data.npy                   # individual station files (optional)
├── AVANOS_data.npy
├── DERİNKUYU_data.npy
├── GÜLŞEHİR_data.npy
├── HACIBEKTAŞ_data.npy
├── KOZAKLI_data.npy
├── NEVŞEHİR_data.npy
└── ÜRGÜP_data.npy
```

The path is configured in `config.py` (`All_STATION_DATA_PATH`).

---

## Key Files for Verification

If you do **not** need to re-run the optimization but want to confirm the
statistics reported in the paper, the following small files contain the
canonical numerical results:

| File | Contains |
|------|----------|
| `faz1_comparison_runs/{algo}_final_results/analysis_summary.json` | Per-algorithm Phase 1 final summary (best loss, params, 30-run stats) |
| `faz1_comparison_results/aggregate_summary.json` | Phase 1 cross-algorithm aggregate |
| `faz1_comparison_results/persistence_results.json` | Persistence baseline + skill score (0.4364) |
| `faz2_sequential_compare/joint_vs_sequential_summary.json` | Cliff's δ, paired Wilcoxon p-values |
| `faz3_ablation_results/ablation_summary.json` | Ablation effect sizes (Cohen's d, Cliff δ) |
| `faz4_walk_forward_compare/gwo_summary.json` | Walk-forward HP consistency + Kruskal–Wallis |

Each of these is human-readable JSON and corresponds directly to a
specific table or figure in the manuscript.

---

## Dependencies

The core scientific stack is:

```
numpy
scipy
pandas
scikit-learn
torch                  (PyTorch; optional if only running analyses)
PyWavelets
matplotlib
```

A pinned `requirements.txt` is provided. If you only intend to run the
analysis (not the optimization), `torch` is optional.

---

## Acknowledgements

- Hourly wind speed data: Turkish State Meteorological Service (MGM).
- Compute: TÜBİTAK ULAKBİM TRUBA High-Performance and Grid Computing
  Centre.
- The author thanks the open-source community behind PyWavelets, PyTorch,
  SciPy and NumPy.

---

## License

- **Code:** MIT License — see `LICENSE`.
- **Aggregated result JSONs / figures:** CC-BY-4.0 (please cite the paper).
- **Raw wind-speed data:** Not redistributed; subject to MGM licensing.

---

## Citation

If you use this code or any of the result artefacts in your research,
please cite:

```bibtex
@article{bendes2026joint,
  author    = {Bende{\c{s}}, Emre},
  title     = {Joint Hyperparameter Optimization of a {DWT}--{BiLSTM}
               Architecture for Wind Speed Forecasting: A Systematic
               Comparison of Metaheuristic Algorithms with Component
               Ablation Analysis},
  journal   = {Knowledge-Based Systems},
  year      = {2026},
  note      = {Submitted}
}

@software{bendes2026code,
  author    = {Bende{\c{s}}, Emre},
  title     = {Wind-DWT-BiLSTM-Metaheuristic: Joint Hyperparameter
               Optimization Codebase},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.20783711},
  url       = {https://doi.org/10.5281/zenodo.20783711}
}
```

Full archival snapshot (including all `faz*_runs/` trial-history
databases and trained model checkpoints, ~1 GB total) will be mirrored
at Zenodo: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20783711.svg)](https://doi.org/10.5281/zenodo.20783711)

DOI (concept, all versions): [10.5281/zenodo.20783711](https://doi.org/10.5281/zenodo.20783711)  
DOI (this version, v1.0.0): [10.5281/zenodo.20783712](https://doi.org/10.5281/zenodo.20783712)

---

## Contact

**Emre Bendeş**
Department of Computer Engineering
Nevşehir Hacı Bektaş Veli University, Türkiye
✉ <emrebendes@nevsehir.edu.tr>
