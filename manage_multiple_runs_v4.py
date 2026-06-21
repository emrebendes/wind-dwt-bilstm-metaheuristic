#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
manage_multiple_runs_v4.py - 6 algoritma icin coklu kosu yonetimi (v4).

Eski manage_multiple_runs.py (sadece ABC) ve manage_multiple_runs_ga.py'nin
yerine gecer. Tek scriptle tum algoritmalari yonetir:
    abc, ga, pso, ho, fno, raindrop (varsayilan 6)

Komutlar:
    prepare  : Klasor yapisi + seed + TRUBA SLURM scriptleri olustur
    status   : Tum algoritmalarin durumunu rapor et
    collect  : Tamamlanan DB'leri toplama klasorune kopyala
    run      : Lokal'de tek bir kosuyu calistir (test icin)

Kullanim:
    # 6 algoritmaya 30'ar kosu hazirla (tum SLURM scriptleri otomatik)
    python manage_multiple_runs_v4.py prepare \\
        --algos abc,ga,pso,ho,fno,raindrop --n-runs 30

    # Tum durumlari rapor
    python manage_multiple_runs_v4.py status

    # Lokal test
    python manage_multiple_runs_v4.py run --algo abc --run-id 1

TRUBA kullanimi (prepare ciktisi sonrasi):
    cd <output_dir>
    bash truba_run_all_algos_v4.sh           # tum algoritmalar
    bash truba_run_one_algo_v4.sh abc        # tek algoritma
    bash truba_status_v4.sh                  # durum

3 hesapli yapilandirma:
    truba_run_one_algo_v4.sh hesap basina 10 SLURM job (3-paralel)
    User 3 hesapla 30 job paralel calisabilir
    3-gun limit gecince checkpoint/resume otomatik (run_optimizer.py)
"""

import os
import sys
import json
import shutil
import sqlite3
import random
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


# =============================================================================
# AYARLAR
# =============================================================================

DEFAULT_ALGOS = ['abc', 'ga', 'pso', 'ho', 'fno', 'raindrop', 'toa']
DEFAULT_N_RUNS = 30
DEFAULT_BASE_SEED = 42
DEFAULT_OUTPUT_DIR = "."

# TRUBA ARF Ayarlari
PROJECT_DIR = "/arf/scratch/gbilgictuzemen/wind2"
ACCOUNT = "gbilgictuzemen"
CONDA_ENV = "ai2"

# Compute parametreleri (v5'de azaltildi)
DEFAULT_POP_SIZE = 40      # 40 -> 30 (compute %25 azaldi)
DEFAULT_MAX_ITER = 50      # 50 -> 35 (compute %30 azaldi)
DEFAULT_WORKERS = 40       # orfoz node basina 35 worker


# =============================================================================
# SEED YONETIMI
# =============================================================================

def generate_seeds(n: int, base_seed: int = DEFAULT_BASE_SEED) -> list:
    """Tekrarlanabilir rastgele seedler uretir.

    Tum algoritmalar AYNI seed setini kullanir (fair comparison).
    """
    rng = random.Random(base_seed)
    return [rng.randint(1, 999999) for _ in range(n)]


# =============================================================================
# HAZIRLIK
# =============================================================================

def _runs_dir_name(algo: str, ablation: str = None) -> str:
    """Klasor adi: <algo>_runs veya <algo>_<ablation>_runs (ablation icin)."""
    if ablation:
        return f"{algo}_{ablation}_runs"
    return f"{algo}_runs"


def prepare_one_algo(algo: str, n_runs: int, seeds: list, output_base: str, ablation: str = None) -> str:
    """Tek bir algoritma icin klasor yapisi olustur.

    Returns: algo_dir yolu
    """
    algo_dir = os.path.join(output_base, _runs_dir_name(algo, ablation))
    os.makedirs(algo_dir, exist_ok=True)

    # Seedleri kaydet (ablation alanini da yaz)
    seeds_path = os.path.join(algo_dir, "seeds.json")
    with open(seeds_path, 'w') as f:
        json.dump({
            'algorithm': algo,
            'ablation': ablation,
            'n_runs': n_runs,
            'seeds': seeds,
            'created_at': datetime.now().isoformat(),
        }, f, indent=2)

    # Her kosu icin alt klasor + config.json
    for i, seed in enumerate(seeds, 1):
        run_dir = os.path.join(algo_dir, f"run_{i:03d}")
        os.makedirs(run_dir, exist_ok=True)
        config = {
            'algorithm': algo,
            'ablation': ablation,
            'run_id': f"run_{i:03d}",
            'run_number': i,
            'seed': seed,
            'status': 'PENDING',
            'created_at': datetime.now().isoformat(),
        }
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    return algo_dir


def prepare_all(algos: list, n_runs: int, base_seed: int, output_base: str,
                ablation: str = None):
    """Tum algoritmalar icin klasor yapisini hazirla (ablation opsiyonel)."""
    print(f"\n{'='*60}")
    print(f"HAZIRLIK: {len(algos)} algoritma × {n_runs} kosu = {len(algos)*n_runs} toplam")
    if ablation:
        print(f"Ablation modu: {ablation} (bilesen kapali calisacak)")
    print(f"{'='*60}")
    print(f"Algoritmalar: {algos}")
    print(f"Base seed: {base_seed}")

    # Tum algoritmalar AYNI seed setini kullanir (joint ile ayni!)
    seeds = generate_seeds(n_runs, base_seed)
    print(f"Seedler: {seeds[:5]}... ({n_runs} adet)")
    print()

    for algo in algos:
        algo_dir = prepare_one_algo(algo, n_runs, seeds, output_base, ablation)
        print(f"  OK {algo_dir} ({n_runs} kosu)")

    print(f"\n{'='*60}")
    print("KLASOR YAPISI HAZIR")
    print(f"{'='*60}")


# =============================================================================
# TRUBA SLURM SCRIPTLERI
# =============================================================================

def create_truba_slurm_scripts(output_base: str, algos: list, n_runs: int,
                                  pop_size: int, max_iter: int, workers: int,
                                  ablation: str = None):
    """TRUBA orfoz icin SLURM scriptleri olustur.

    Ablation modu (Faz 3) destegi: ablation argumani verilirse:
      - Klasor: <algo>_<ablation>_runs/
      - run_optimizer.py'ye --ablation X argumani gecirilir
      - SLURM scripti adlari: truba_*_v4_<ablation>.sh

    Olusturulan scriptler:
    - truba_run_one_v4[_<abl>].sh <algo> <run_num>      : Tek kosu (test)
    - truba_3parallel_v4[_<abl>].sh <algo> <r1> <r2> <r3>: 3-paralel kosu
    - truba_run_one_algo_v4[_<abl>].sh <algo>           : Tek algoritma 30 kosu
    - truba_run_all_algos_v4[_<abl>].sh                 : Tum algoritmalar
    - truba_status_v4.sh                                : Durum sorgu
    """
    logs_dir = os.path.join(output_base, 'logs_v4')
    os.makedirs(logs_dir, exist_ok=True)

    # Ablation placeholders
    abl_suffix = f"_{ablation}" if ablation else ""
    abl_arg = f"--ablation {ablation} " if ablation else ""
    abl_tag_str = f" [ablation: {ablation}]" if ablation else ""
    abl_filename = f"_{ablation}" if ablation else ""

    # =========================================================================
    # SCRIPT 1: Tek kosu (test/debug)
    # =========================================================================
    script_path = os.path.join(output_base, f'truba_run_one_v4{abl_filename}.sh')
    with open(script_path, 'w') as f:
        f.write(f"""#!/bin/bash
#======================================================================
# TRUBA - Tek kosu (test/debug)
# Kullanim: sbatch truba_run_one_v4.sh <algo> <run_num>
#   ornek: sbatch truba_run_one_v4.sh abc 1
#======================================================================

#SBATCH -p orfoz
#SBATCH -A {ACCOUNT}
#SBATCH -J optv4_test
#SBATCH -o {logs_dir}/test_%j.out
#SBATCH -e {logs_dir}/test_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c {workers + 1}
#SBATCH -C weka
#SBATCH --time=3-00:00:00

ALGO=${{1:-abc}}
RUN_NUM=${{2:-1}}
RUN_ID=$(printf "%03d" $RUN_NUM)

module load miniconda3
conda activate {CONDA_ENV}
cd {PROJECT_DIR}

SEED=$(python -c "import json; print(json.load(open('${{ALGO}}{abl_suffix}_runs/run_${{RUN_ID}}/config.json'))['seed'])")
DB_FILE="${{ALGO}}{abl_suffix}_runs/run_${{RUN_ID}}/${{ALGO}}_running.db"

echo "Test: $ALGO run_$RUN_ID seed=$SEED"
echo "DB: $DB_FILE"

python run_optimizer.py \\
    --algo $ALGO \\
    {abl_arg}\\
    --seed $SEED \\
    --db $DB_FILE \\
    --pop-size {pop_size} \\
    --max-iter {max_iter} \\
    --workers {workers}
""")
    os.chmod(script_path, 0o755)

    # =========================================================================
    # SCRIPT 2: 3-paralel orfoz (eski kullaniciya tanidik gelen format)
    # =========================================================================
    script_path = os.path.join(output_base, f'truba_3parallel_v4{abl_filename}.sh')
    with open(script_path, 'w') as f:
        f.write(f"""#!/bin/bash
#======================================================================
# TRUBA orfoz - 3 paralel kosu (tek node, 3 farkli kosu)
# Kullanim: sbatch truba_3parallel_v4.sh <algo> <r1> <r2> <r3>
#   ornek: sbatch truba_3parallel_v4.sh abc 1 2 3
#
# Her kosu ~36 core kullanir (toplam ~108 core); orfoz 110 core'a sigar.
#======================================================================

#SBATCH -p orfoz
#SBATCH -A {ACCOUNT}
#SBATCH -J optv4_3par
#SBATCH -o {logs_dir}/3par_%j.out
#SBATCH -e {logs_dir}/3par_%j.err
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -c 36
#SBATCH -C weka
#SBATCH --time=3-00:00:00
#SBATCH --mem=240G

module load miniconda3
conda activate {CONDA_ENV}
cd {PROJECT_DIR}

ALGO=${{1:-abc}}
R1=${{2:-1}}
R2=${{3:-2}}
R3=${{4:-3}}

ID1=$(printf "%03d" $R1)
ID2=$(printf "%03d" $R2)
ID3=$(printf "%03d" $R3)

S1=$(python -c "import json; print(json.load(open('${{ALGO}}{abl_suffix}_runs/run_${{ID1}}/config.json'))['seed'])")
S2=$(python -c "import json; print(json.load(open('${{ALGO}}{abl_suffix}_runs/run_${{ID2}}/config.json'))['seed'])")
S3=$(python -c "import json; print(json.load(open('${{ALGO}}{abl_suffix}_runs/run_${{ID3}}/config.json'))['seed'])")

echo "$ALGO: 3-paralel kosu (run $ID1, $ID2, $ID3)"

# 3 paralel job
srun -n 1 -c 36 python run_optimizer.py --algo $ALGO {abl_arg}--seed $S1 \\
    --db ${{ALGO}}{abl_suffix}_runs/run_${{ID1}}/${{ALGO}}_running.db \\
    --pop-size {pop_size} --max-iter {max_iter} --workers 35 \\
    > {logs_dir}/${{ALGO}}{abl_suffix}_run_${{ID1}}.log 2>&1 &

srun -n 1 -c 36 python run_optimizer.py --algo $ALGO {abl_arg}--seed $S2 \\
    --db ${{ALGO}}{abl_suffix}_runs/run_${{ID2}}/${{ALGO}}_running.db \\
    --pop-size {pop_size} --max-iter {max_iter} --workers 35 \\
    > {logs_dir}/${{ALGO}}{abl_suffix}_run_${{ID2}}.log 2>&1 &

srun -n 1 -c 36 python run_optimizer.py --algo $ALGO {abl_arg}--seed $S3 \\
    --db ${{ALGO}}{abl_suffix}_runs/run_${{ID3}}/${{ALGO}}_running.db \\
    --pop-size {pop_size} --max-iter {max_iter} --workers 35 \\
    > {logs_dir}/${{ALGO}}{abl_suffix}_run_${{ID3}}.log 2>&1 &

wait
echo "Tamamlandi: $(date)"
""")
    os.chmod(script_path, 0o755)

    # =========================================================================
    # SCRIPT 3: Tek algoritma, 30 kosu (10 job × 3-paralel)
    # =========================================================================
    n_jobs = (n_runs + 2) // 3  # 30/3 = 10
    script_path = os.path.join(output_base, f'truba_run_one_algo_v4{abl_filename}.sh')
    with open(script_path, 'w') as f:
        f.write(f"""#!/bin/bash
#======================================================================
# Bir algoritmanin {n_runs} kosusunu {n_jobs} job'da tamamla (her job 3-paralel)
# Kullanim: bash truba_run_one_algo_v4.sh <algo>
#   ornek: bash truba_run_one_algo_v4.sh abc
#======================================================================

ALGO=${{1:-abc}}

if [ -z "$ALGO" ]; then
    echo "Kullanim: $0 <algo>"
    echo "Algoritmalar: abc ga pso ho fno raindrop"
    exit 1
fi

echo "==============================================================="
echo "$ALGO icin {n_jobs} job × 3 paralel = {n_runs} kosu"
echo "==============================================================="
""")

        for i in range(n_jobs):
            r1 = i * 3 + 1
            r2 = i * 3 + 2
            r3 = i * 3 + 3
            if r3 > n_runs:
                # Son job eksik kosu olabilir
                pass
            f.write(f"""
echo "Job {i+1}/{n_jobs}: $ALGO kosu {r1}, {r2}, {r3}"
sbatch truba_3parallel_v4{abl_filename}.sh $ALGO {r1} {r2} {r3}
sleep 2
""")
        f.write("""
echo "==============================================================="
echo "Tum joblar gonderildi. Durum: squeue -u $USER"
echo "==============================================================="
""")
    os.chmod(script_path, 0o755)

    # =========================================================================
    # SCRIPT 4: Tum algoritmalar (master script)
    # =========================================================================
    script_path = os.path.join(output_base, f'truba_run_all_algos_v4{abl_filename}.sh')
    with open(script_path, 'w') as f:
        f.write("""#!/bin/bash
#======================================================================
# TUM 6 algoritmayi tek seferde TRUBA'ya gonder.
# Toplam: 6 algo × 10 job = 60 SLURM job
# 3 hesap paralel kullanmak istersen bu scripti her hesapta 2 algoritma
# icin parcala.
#======================================================================

ALGOS=(""")
        for algo in algos:
            f.write(f' "{algo}"')
        f.write(""")

for ALGO in "${ALGOS[@]}"; do
    echo "==============================================================="
    echo "Submitting: $ALGO"
    echo "==============================================================="
    bash truba_run_one_algo_v4{abl_filename}.sh $ALGO
    echo "Sleep 5s before next algorithm..."
    sleep 5
done

echo ""
echo "TUM ALGORITMALAR GONDERILDI"
echo "Durum: bash truba_status_v4.sh"
""")
    os.chmod(script_path, 0o755)

    # =========================================================================
    # SCRIPT 5: Durum sorgu (Python tabanli)
    # =========================================================================
    script_path = os.path.join(output_base, 'truba_status_v4.sh')
    with open(script_path, 'w') as f:
        f.write(f"""#!/bin/bash
#======================================================================
# Tum algoritmalarin durumunu rapor et
#======================================================================

echo "=== SLURM Kuyrugu ==="
squeue -u $USER

echo ""
echo "=== Algoritma Durumlari ==="

module load miniconda3
conda activate {CONDA_ENV}
cd {PROJECT_DIR}

python manage_multiple_runs_v4.py status --algos """)
        f.write(",".join(algos))
        f.write("""
""")
    os.chmod(script_path, 0o755)

    print(f"\nTRUBA scriptleri olusturuldu:")
    print(f"  {os.path.join(output_base, f'truba_run_one_v4{abl_filename}.sh')} - test")
    print(f"  {os.path.join(output_base, f'truba_3parallel_v4{abl_filename}.sh')} - 3-paralel")
    print(f"  {os.path.join(output_base, f'truba_run_one_algo_v4{abl_filename}.sh')} - tek algo {n_runs} kosu")
    print(f"  {os.path.join(output_base, f'truba_run_all_algos_v4{abl_filename}.sh')} - tum algoritmalar")
    print(f"  {os.path.join(output_base, 'truba_status_v4.sh')} - durum")


# =============================================================================
# DURUM SORGU
# =============================================================================

def check_one_algo(algo: str, output_base: str) -> dict:
    """Tek algoritmanin durumunu rapor et."""
    algo_dir = os.path.join(output_base, f"{algo}_runs")
    if not os.path.exists(algo_dir):
        return {'algo': algo, 'error': 'Klasor yok'}

    stats = {'PENDING': 0, 'RUNNING': 0, 'COMPLETED': 0, 'FAILED': 0}
    losses = []
    total = 0

    for run_dir in sorted(Path(algo_dir).glob("run_*")):
        config_path = run_dir / 'config.json'
        if not config_path.exists():
            continue
        total += 1

        # DB var mi?
        db_path = run_dir / f"{algo}_running.db"
        if not db_path.exists():
            stats['PENDING'] += 1
            continue

        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("SELECT status FROM metadata WHERE id=1")
            row = cur.fetchone()
            status = row[0] if row else 'UNKNOWN'
            stats[status] = stats.get(status, 0) + 1

            if status == 'COMPLETED':
                cur.execute("SELECT MIN(loss) FROM trial_history WHERE loss < 1e10")
                best = cur.fetchone()[0]
                if best:
                    losses.append(best)
            conn.close()
        except Exception as e:
            stats['FAILED'] += 1

    return {
        'algo': algo,
        'total': total,
        'stats': stats,
        'mean_loss': sum(losses) / len(losses) if losses else None,
        'min_loss': min(losses) if losses else None,
        'completed_count': len(losses),
    }


def check_status_all(output_base: str, algos: list):
    """Tum algoritmalarin durumunu rapor et."""
    print(f"\n{'='*70}")
    print(f"{'Algoritma':<12}{'Toplam':<8}{'Pending':<10}{'Running':<10}{'Completed':<12}{'MinLoss':<12}")
    print(f"{'='*70}")

    for algo in algos:
        r = check_one_algo(algo, output_base)
        if 'error' in r:
            print(f"{algo:<12}{'-':<8}{'-':<10}{'-':<10}{'-':<12}({r['error']})")
            continue

        s = r['stats']
        min_loss_str = f"{r['min_loss']:.6f}" if r['min_loss'] else "-"
        print(f"{algo:<12}{r['total']:<8}{s['PENDING']:<10}{s['RUNNING']:<10}"
              f"{s['COMPLETED']:<12}{min_loss_str:<12}")

    print(f"{'='*70}")


# =============================================================================
# SONUCLARI TOPLAMA
# =============================================================================

def collect_results(output_base: str, algos: list, target_dir: str = None):
    """Tamamlanan DB'leri toplama klasorune kopyala."""
    if target_dir is None:
        target_dir = os.path.join(output_base, 'collected_dbs_v4')
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SONUCLAR TOPLANIYOR -> {target_dir}")
    print(f"{'='*60}")

    for algo in algos:
        algo_dir = os.path.join(output_base, f"{algo}_runs")
        if not os.path.exists(algo_dir):
            continue

        algo_target = os.path.join(target_dir, f"{algo}_dbs")
        os.makedirs(algo_target, exist_ok=True)
        collected = 0

        for run_dir in sorted(Path(algo_dir).glob("run_*")):
            db_path = run_dir / f"{algo}_running.db"
            if not db_path.exists():
                continue

            try:
                conn = sqlite3.connect(str(db_path))
                cur = conn.cursor()
                cur.execute("SELECT status FROM metadata WHERE id=1")
                status = cur.fetchone()[0]
                conn.close()
                if status == 'COMPLETED':
                    target_path = os.path.join(algo_target, f"{run_dir.name}.db")
                    shutil.copy(str(db_path), target_path)
                    collected += 1
            except Exception:
                pass

        print(f"  {algo}: {collected} DB toplandi -> {algo_target}")

    print(f"\nToplam: tamamlananlar {target_dir} altina kopyalandi")


# =============================================================================
# LOKAL KOSU (test icin)
# =============================================================================

def run_local(algo: str, run_num: int, output_base: str):
    """Lokal'de tek kosu calistir (test/debug)."""
    run_dir = os.path.join(output_base, f"{algo}_runs", f"run_{run_num:03d}")
    config_path = os.path.join(run_dir, 'config.json')

    if not os.path.exists(config_path):
        print(f"HATA: {config_path} yok. Once 'prepare' calistir.")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)
    seed = config['seed']
    db_path = os.path.join(run_dir, f"{algo}_running.db")

    print(f"\nLokal kosu: {algo} run_{run_num:03d} (seed={seed})")
    cmd = [
        sys.executable, "run_optimizer.py",
        "--algo", algo,
        "--seed", str(seed),
        "--db", db_path,
        "--pop-size", str(DEFAULT_POP_SIZE),
        "--max-iter", str(DEFAULT_MAX_ITER),
        "--workers", "4",
    ]
    print(f"Komut: {' '.join(cmd)}")
    subprocess.run(cmd)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="6 algoritma icin coklu kosu yonetimi v4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # prepare
    p = sub.add_parser('prepare', help='Klasor + seed + SLURM scriptleri olustur')
    p.add_argument('--algos', type=str, default=','.join(DEFAULT_ALGOS))
    p.add_argument('--n-runs', type=int, default=DEFAULT_N_RUNS)
    p.add_argument('--base-seed', type=int, default=DEFAULT_BASE_SEED)
    p.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--pop-size', type=int, default=DEFAULT_POP_SIZE)
    p.add_argument('--max-iter', type=int, default=DEFAULT_MAX_ITER)
    p.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    p.add_argument('--ablation', type=str, default=None,
                   choices=['nodwt', 'nobidir', 'waveletfixed'],
                   help='Faz 4 ablation modu (nodwt/nobidir/waveletfixed). '
                        'Klasor adi <algo>_<ablation>_runs olur.')

    # status
    s = sub.add_parser('status', help='Durum sorgu')
    s.add_argument('--algos', type=str, default=','.join(DEFAULT_ALGOS))
    s.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)

    # collect
    c = sub.add_parser('collect', help='Tamamlanan DB\'leri topla')
    c.add_argument('--algos', type=str, default=','.join(DEFAULT_ALGOS))
    c.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    c.add_argument('--target-dir', type=str, default=None)

    # run (lokal test)
    r = sub.add_parser('run', help='Lokal\'de tek kosu calistir')
    r.add_argument('--algo', type=str, required=True)
    r.add_argument('--run-id', type=int, required=True)
    r.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()

    if args.command == 'prepare':
        algos = [a.strip() for a in args.algos.split(',')]
        prepare_all(algos, args.n_runs, args.base_seed, args.output_dir,
                    ablation=args.ablation)
        create_truba_slurm_scripts(args.output_dir, algos, args.n_runs,
                                     args.pop_size, args.max_iter, args.workers,
                                     ablation=args.ablation)
    elif args.command == 'status':
        algos = [a.strip() for a in args.algos.split(',')]
        check_status_all(args.output_dir, algos)
    elif args.command == 'collect':
        algos = [a.strip() for a in args.algos.split(',')]
        collect_results(args.output_dir, algos, args.target_dir)
    elif args.command == 'run':
        run_local(args.algo, args.run_id, args.output_dir)


if __name__ == '__main__':
    main()
