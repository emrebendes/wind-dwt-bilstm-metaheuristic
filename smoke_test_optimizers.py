#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smoke_test_optimizers.py - 8 algoritma icin smoke test (DWT+BiLSTM, v9).

Test edilen algoritmalar: ABC, GA, PSO, GWO, HO, FNO, Raindrop, TOA.
GERCEK fitness evaluation YAPILMAZ; sadece import + instantiation +
state pickle round-trip dogrulanir.

Gercek dogrulama icin:
    python run_optimizer.py --algo <X> --seed 42 \\
        --pop-size 5 --max-iter 3 --db abc_runs_yeni\\smoke_real.db --workers 1
"""

import os
import sys
import tempfile
import shutil

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 7 algoritma test edilir
ALGOS = ['abc', 'ga', 'pso', 'gwo', 'ho', 'fno', 'raindrop', 'toa']


def test_imports():
    print("[TEST 1/6] Import testi...")
    from optimizers import (
        register_algorithm, get_optimizer, list_algorithms,
    )
    from optimizers.base_optimizer import BaseOptimizer
    from optimizers.db_manager import GenericDBManager
    from optimizers.param_mapping import (
        BOUNDS, DIMENSION, map_vector_to_params, calculate_fitness,
    )
    from utils import WAVELET_CHOICES, MODE_CHOICES, BATCH_CHOICES
    try:
        from optimizers.evaluator import evaluate_solution_wrapper, init_worker
        print("    OK Tum moduller import edildi (evaluator dahil)")
    except ImportError as e:
        print(f"    UYARI evaluator modulu import edilemedi: {e}")


def test_registry():
    print("\n[TEST 2/6] Registry testi...")
    from optimizers import list_algorithms, get_optimizer
    algos = list_algorithms()
    print(f"    Kayitli algoritmalar: {algos}")
    for name in ALGOS:
        assert name in algos, f"{name} kayitli degil"
        cls = get_optimizer(name)
        assert cls.ALGORITHM_NAME == name
    print(f"    OK {len(ALGOS)} algoritma registry'de mevcut")


def test_param_mapping():
    print("\n[TEST 3/6] Parametre mapping testi...")
    import numpy as np
    from optimizers.param_mapping import map_vector_to_params, DIMENSION

    from utils import HP
    v_min = np.zeros(DIMENSION)
    v_max = np.ones(DIMENSION) * 0.999
    p_min = map_vector_to_params(v_min)
    p_max = map_vector_to_params(v_max)
    print(f"    v=0    -> wavelet={p_min[HP.DWT_WAVELET]}, level={p_min[HP.DWT_LEVEL]}, "
          f"look_back={p_min[HP.LOOK_BACK]}")
    print(f"    v=0.999 -> wavelet={p_max[HP.DWT_WAVELET]}, level={p_max[HP.DWT_LEVEL]}, "
          f"look_back={p_max[HP.LOOK_BACK]}")
    assert p_min[HP.DWT_LEVEL] == 3 and p_max[HP.DWT_LEVEL] == 7
    assert p_min[HP.LOOK_BACK] == 24 and p_max[HP.LOOK_BACK] == 96

    v_rand = np.array([0.5] * DIMENSION)
    p_rand = map_vector_to_params(v_rand)
    print(f"    v=0.5  -> wavelet={p_rand[HP.DWT_WAVELET]}, "
          f"hidden={p_rand[HP.HIDDEN]}, batch={p_rand[HP.BATCH]}")
    print("    OK Parametre mapping dogru")


def test_db_manager():
    print("\n[TEST 4/6] DB manager testi...")
    import numpy as np
    from optimizers.db_manager import GenericDBManager

    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "smoke.db")
    db = GenericDBManager(db_path, algorithm="test")

    db.initialize_run(seed=42, pop_size=10, max_iter=5, dimension=10,
                      config={'test': True})
    assert db.get_status() == 'RUNNING'
    print("    OK Init")

    state = {
        'population': np.random.rand(10, 10),
        'fitness': np.zeros(10),
        'custom_field': 'hello',
    }
    db.save_checkpoint(
        iteration=2, phase='TEST',
        state=state, global_opt=0.5,
        global_vector=np.zeros(10),
    )
    cp = db.load_checkpoint()
    assert cp is not None
    assert cp['state']['custom_field'] == 'hello'
    print("    OK Checkpoint save/load")

    fake_results = [{
        'loss': 0.1 + i * 0.01,
        'fitness': 0.9 - i * 0.01,
        'params': {},
        'params_str': {
            'K': 3, 'alpha': 2000.0, 'look_back': 24,
            'hidden': 32, 'layers': 2, 'kernel_size': 3,
            'att_dim': 16, 'dropout': 0.1, 'lr': 0.001, 'batch': 32,
        },
        'vector': [0.5] * 10,
        'elapsed_seconds': 1.0,
        'member_id': i,
        'member_type': 'I',
        'iteration': 0,
        'timestamp': '2026-05-09T00:00:00',
    } for i in range(5)]
    db.save_trials_batch(0, 'INIT', fake_results)
    summary = db.get_summary()
    assert summary['total_trials'] == 5
    print(f"    OK Trial save (5 trials)")

    db.mark_completed()
    assert db.get_status() == 'COMPLETED'
    print("    OK Mark completed")
    shutil.rmtree(tmpdir)


def test_optimizer_instantiation():
    print("\n[TEST 5/6] Optimizer instantiation testi...")
    from optimizers import get_optimizer

    tmpdir = tempfile.mkdtemp()

    # Algoritma-spesifik attribute beklentileri
    expected_attrs = {
        'abc':       ['trial'],
        'ga':        [],
        'pso':       ['velocities', 'pbest'],
        'gwo':       ['alpha_pos'],
        'ho':        [],
        'fno':       [],
        'raindrop':  [],
        'toa':      [],
    }

    for algo_name in ALGOS:
        Cls = get_optimizer(algo_name)
        db = os.path.join(tmpdir, f"{algo_name}_smoke.db")
        opt = Cls(
            db_file=db, seed=42, pop_size=4, max_iterations=2, n_workers=1,
        )
        assert opt.population is not None
        from optimizers.param_mapping import DIMENSION
        assert opt.population.shape == (4, DIMENSION)
        # Algoritma-spesifik check
        for attr in expected_attrs[algo_name]:
            assert hasattr(opt, attr), f"{algo_name} missing {attr}"
        print(f"    OK {algo_name.upper():8s} olusturuldu (pop: {opt.population.shape})")

    shutil.rmtree(tmpdir)


def test_state_serialization():
    print("\n[TEST 6/6] State serialization testi...")
    import pickle
    from optimizers import get_optimizer

    tmpdir = tempfile.mkdtemp()

    for algo_name in ALGOS:
        Cls = get_optimizer(algo_name)
        opt = Cls(
            db_file=os.path.join(tmpdir, f"{algo_name}_st.db"),
            seed=42, pop_size=4, max_iterations=2, n_workers=1,
        )
        state = opt._get_state_dict()
        # Temel anahtarlar her algoritmada olmalı
        for k in ['population', 'fitness']:
            assert k in state, f"{algo_name}: '{k}' state'te yok"
        pickled = pickle.dumps(state)
        restored = pickle.loads(pickled)
        opt._set_state_dict(restored)
        print(f"    OK {algo_name.upper():8s} state pickle round-trip")

    shutil.rmtree(tmpdir)


def main():
    print("=" * 60)
    print(f"OPTIMIZERS SMOKE TEST ({len(ALGOS)} algoritma)")
    print("=" * 60)

    try:
        test_imports()
        test_registry()
        test_param_mapping()
        test_db_manager()
        test_optimizer_instantiation()
        test_state_serialization()
        print("\n" + "=" * 60)
        print(f"TUM SMOKE TESTLER GECTI ({len(ALGOS)} algoritma)")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\nASSERTION HATASI: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\nHATA: {e}")
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
