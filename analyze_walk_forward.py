#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_walk_forward.py — Faz 2 sonuc karsilastirmasi (Hakem 2.6 & 3.Q4 cevabi).

Girdiler (her pencere icin):
    walk_forward_runs/{algo}_{window}_analysis.json     (HP search ozeti)
    walk_forward_runs/{algo}_{window}_analysis.json["final_training_loss"]
    {algo}_final_results/analysis_summary.json          (Y6 = Faz 1 sonucu)

Cikti:
    walk_forward_compare/{algo}_summary.json
    walk_forward_compare/{algo}_hp_table.csv
    walk_forward_compare/{algo}_rmse_table.csv
    walk_forward_compare/{algo}_hp_consistency.tex   (paper'a kopyala)
    walk_forward_compare/{algo}_rmse_consistency.tex
    walk_forward_compare/{algo}_boxplot.png

Analizler:
    A) HP tutarlilik tablosu (Y2..Y6 best HP'ler yan yana)
    B) search_RMSE Wilcoxon test (H0: Y2..Y5 ortancasi == Y6'nin search RMSE'si)
    C) final_RMSE Wilcoxon test (H0: Y2..Y5 ortancasi == Y6'nin final RMSE'si)
    D) Kategorik HP'lerin tutarliligi: mod, en sik secilen deger frekansi
    E) Surekli HP'lerin CV%, IQR

Kullanim:
    python analyze_walk_forward.py --algo ga --windows Y2,Y3,Y4,Y5,Y6
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [WFC] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("WFC")

# Matplotlib akademik format
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# Kategorik vs surekli HP'leri ayir
CATEGORICAL_HPS = ["dwt_wavelet", "dwt_mode", "batch"]
CONTINUOUS_HPS  = ["dwt_level", "look_back", "hidden", "layers", "dropout", "lr"]


# =============================================================================
# JSON YUKLEME
# =============================================================================

def load_window_result(algo: str, window: str) -> dict:
    """Bir pencerenin (HP search + final training) sonuclarini yukler.

    Y6 ozeldir: {algo}_final_results/analysis_summary.json'dan okunur.
    """
    if window == "Y6":
        path = Path(f"{algo}_final_results/analysis_summary.json")
        if not path.exists():
            logger.warning(f"Y6 (Faz 1) sonucu yok: {path}")
            return None
        with open(path) as f:
            d = json.load(f)
        return {
            "window": "Y6",
            "best_loss": d.get("best_loss"),
            "best_params": d.get("best_params"),
            "final_training_loss": d.get("final_training_loss"),
            "statistics": d.get("statistics", {}),
            "all_losses": d.get("all_losses", []),
        }
    else:
        path = Path(f"walk_forward_runs/{algo}_{window}_analysis.json")
        if not path.exists():
            logger.warning(f"Pencere sonucu yok: {path}")
            return None
        with open(path) as f:
            d = json.load(f)
        return {
            "window": window,
            "best_loss": d.get("best_loss"),
            "best_params": d.get("best_params"),
            "final_training_loss": d.get("final_training_loss"),
            "statistics": d.get("statistics", {}),
            "all_losses": d.get("all_losses", []),
        }


# =============================================================================
# A) HP TUTARLILIK TABLOSU
# =============================================================================

def build_hp_table(results: dict) -> list:
    """Tum pencerelerin best HP'sini tek tabloda topla.

    Donus: rows = [{window, dwt_wavelet, dwt_level, ...}, ...]
    """
    rows = []
    for win, res in results.items():
        if res is None or res.get("best_params") is None:
            continue
        row = {"window": win}
        row.update(res["best_params"])
        row["search_rmse"] = res.get("best_loss")
        row["final_rmse"] = res.get("final_training_loss")
        rows.append(row)
    return rows


def hp_consistency_metrics(rows: list) -> dict:
    """Kategorik ve surekli HP'lerin pencereler arasi tutarliligi.

    Kategorik (wavelet, mode, batch): mod + frekans
    Surekli (level, look_back, ...): mean, std, CV%
    """
    metrics = {}

    # Kategorik
    for hp in CATEGORICAL_HPS:
        vals = [r.get(hp) for r in rows if r.get(hp) is not None]
        if not vals:
            continue
        from collections import Counter
        cnt = Counter(vals)
        most, freq = cnt.most_common(1)[0]
        metrics[hp] = {
            "type": "categorical",
            "mode": str(most),
            "mode_frequency": freq / len(vals),
            "unique_count": len(cnt),
            "values": vals,
        }

    # Surekli
    for hp in CONTINUOUS_HPS:
        vals = [r.get(hp) for r in rows if r.get(hp) is not None]
        if not vals:
            continue
        a = np.array(vals, dtype=float)
        mean = float(a.mean())
        std  = float(a.std())
        cv_pct = float((std / abs(mean) * 100) if abs(mean) > 1e-12 else 0.0)
        metrics[hp] = {
            "type": "continuous",
            "mean":   mean,
            "std":    std,
            "min":    float(a.min()),
            "max":    float(a.max()),
            "cv_pct": cv_pct,
            "values": vals,
        }

    return metrics


# =============================================================================
# B) WILCOXON TESTLERI
# =============================================================================

def kruskal_wallis_search(results: dict) -> dict:
    """Tum pencerelerin 30-koSu SEARCH RMSE dagilimlari icin Kruskal-Wallis testi.

    H0: Tum pencerelerin medyani esit
    n_per_window = 30, total = 30 * n_windows

    Bu test n=4 one-sample testten cok daha guclu (n=150 vs n=4).
    Post-hoc: Mann-Whitney U (unpaired, seed bagimsiz) + Holm correction.
    """
    from scipy.stats import kruskal, mannwhitneyu

    # Tum pencerelerin all_losses listesini topla
    window_losses = {}
    for w, r in results.items():
        if r is None:
            continue
        losses = r.get("all_losses")
        if not losses or len(losses) < 5:
            continue
        window_losses[w] = np.array(losses, dtype=float)

    if "Y6" not in window_losses:
        return {"error": "Y6 (Faz 1) all_losses yok"}
    if len(window_losses) < 2:
        return {"error": f"En az 2 pencere gerek (var: {len(window_losses)})"}

    # Kruskal-Wallis (5 grup)
    groups = [window_losses[w] for w in window_losses]
    H, p_kw = kruskal(*groups)

    # Post-hoc: Her Y_i (i != Y6) vs Y6 Mann-Whitney U + Holm
    y6 = window_losses["Y6"]
    posthoc = []
    for w, losses in window_losses.items():
        if w == "Y6":
            continue
        U, p_raw = mannwhitneyu(losses, y6, alternative="two-sided")
        posthoc.append({
            "window": w,
            "U": float(U),
            "p_raw": float(p_raw),
            "median_y_i": float(np.median(losses)),
            "median_y6":  float(np.median(y6)),
            "mean_y_i":   float(losses.mean()),
            "std_y_i":    float(losses.std()),
            "n_y_i":      int(len(losses)),
        })

    # Holm correction (sirali p artan)
    posthoc.sort(key=lambda r: r["p_raw"])
    m = len(posthoc)
    for i, r in enumerate(posthoc):
        r["p_holm"] = min(1.0, r["p_raw"] * (m - i))
        r["significant_at_005"] = bool(r["p_holm"] < 0.05)
        pct_diff = (r["median_y_i"] - r["median_y6"]) / r["median_y6"] * 100
        r["pct_diff_vs_y6"] = float(pct_diff)

    return {
        "test_name": "Kruskal-Wallis (n=" + str(sum(len(g) for g in groups)) +
                     ") + Mann-Whitney U + Holm",
        "kruskal_H": float(H),
        "kruskal_p": float(p_kw),
        "kruskal_significant": bool(p_kw < 0.05),
        "n_per_window": int(np.mean([len(g) for g in groups])),
        "n_total": int(sum(len(g) for g in groups)),
        "n_windows": len(window_losses),
        "post_hoc": posthoc,
        "interpretation": _interpret_kw(p_kw, posthoc),
    }


def _interpret_kw(p_kw: float, posthoc: list) -> str:
    if p_kw >= 0.05:
        return ("OK: Pencereler arasi anlamli fark YOK -> HP arama tutarli")
    n_sig = sum(1 for r in posthoc if r.get("significant_at_005"))
    return (f"ANLAMLI FARK VAR (p={p_kw:.2e}). "
            f"Post-hoc Holm-corrected: {n_sig}/{len(posthoc)} pencere Y6'dan "
            f"anlamli farkli.")


def final_rmse_descriptive(results: dict) -> dict:
    """Final training RMSE tanimlayici istatistik (n=1 per pencere, test yok)."""
    final_data = []
    for w, r in results.items():
        if r is None or r.get("final_training_loss") is None:
            continue
        final_data.append({
            "window": w,
            "final_rmse": float(r["final_training_loss"]),
        })

    if not final_data:
        return {"error": "Hicbir pencerenin final RMSE'si yok"}

    y6_final = next((d["final_rmse"] for d in final_data if d["window"] == "Y6"), None)
    if y6_final is None:
        return {"error": "Y6 final yok"}

    for d in final_data:
        if d["window"] != "Y6":
            d["pct_diff_vs_y6"] = (d["final_rmse"] - y6_final) / y6_final * 100

    finals = np.array([d["final_rmse"] for d in final_data])
    return {
        "n": len(finals),
        "y6": float(y6_final),
        "mean": float(finals.mean()),
        "median": float(np.median(finals)),
        "std": float(finals.std()),
        "min": float(finals.min()),
        "max": float(finals.max()),
        "max_pct_diff_vs_y6": float(max(abs(d.get("pct_diff_vs_y6", 0)) for d in final_data)),
        "values": final_data,
        "note": ("Tek deger per pencere — formal test yapilamiyor; "
                 "fark oranlari descriptive raporlanir."),
    }


# =============================================================================
# C) BOXPLOT
# =============================================================================

def plot_rmse_boxplot(results: dict, algo: str, out_dir: Path) -> Path:
    """Pencere bazli RMSE dagilim boxplot'i."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    windows = []
    search_losses_per_win = []
    final_losses_per_win = []
    for w in ["Y2", "Y3", "Y4", "Y5", "Y6"]:
        r = results.get(w)
        if r is None:
            continue
        windows.append(w)
        search_losses_per_win.append(r.get("all_losses", []))
        final_losses_per_win.append(r.get("final_training_loss"))

    # Sol: HP search loss dagilimi (her pencerede 30 kosu)
    if search_losses_per_win and any(search_losses_per_win):
        axes[0].boxplot(
            [s for s in search_losses_per_win if s],
            labels=[w for w, s in zip(windows, search_losses_per_win) if s],
        )
        axes[0].set_ylabel("HP Search RMSE")
        axes[0].set_title(f"{algo.upper()} — Pencereler Arasi HP Search Dagilimi")
        axes[0].grid(True, alpha=0.3)

    # Sag: Final RMSE (her pencereden 1 tek deger)
    valid_finals = [(w, f) for w, f in zip(windows, final_losses_per_win)
                    if f is not None]
    if valid_finals:
        ws, fs = zip(*valid_finals)
        axes[1].bar(ws, fs, color="steelblue", edgecolor="black")
        axes[1].set_ylabel("Final Training RMSE")
        axes[1].set_title(f"{algo.upper()} — Pencerelerden Cikan Final Modellerin RMSE'si")
        axes[1].grid(True, alpha=0.3, axis="y")
        for i, (w, f) in enumerate(zip(ws, fs)):
            axes[1].text(i, f, f"{f:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = out_dir / f"{algo}_boxplot.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# =============================================================================
# D) LATEX TABLOLAR
# =============================================================================

def emit_hp_consistency_latex(rows: list, metrics: dict, algo: str,
                              out_dir: Path) -> Path:
    """HP tutarlilik LaTeX tablosu (paper'a kopyala-yapistir)."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Walk-forward HP consistency across yearly windows for {algo.upper()}}}",
        rf"\label{{tab:wf_hp_{algo}}}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"Window & Wavelet & Level & Look-back & Hidden & Layers & Dropout \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{r.get('window'):<4s} & {r.get('dwt_wavelet','-'):<6s} & "
            f"{r.get('dwt_level','-'):>2} & {r.get('look_back','-'):>3} & "
            f"{r.get('hidden','-'):>4} & {r.get('layers','-'):>2} & "
            f"{r.get('dropout',0):.3f} \\\\"
        )
    lines.append(r"\hline")
    # Tutarlilik ozeti satiri
    if "dwt_wavelet" in metrics:
        wm = metrics["dwt_wavelet"]
        lines.append(
            f"Most freq. wavelet & \\multicolumn{{6}}{{l}}{{"
            f"{wm['mode']} ({wm['mode_frequency']*100:.0f}\\% of windows)}} \\\\"
        )
    if "dwt_level" in metrics:
        lm = metrics["dwt_level"]
        lines.append(
            f"Level CV\\% & \\multicolumn{{6}}{{l}}{{{lm['cv_pct']:.1f}\\%}} \\\\"
        )
    if "look_back" in metrics:
        lb = metrics["look_back"]
        lines.append(
            f"Look-back CV\\% & \\multicolumn{{6}}{{l}}{{{lb['cv_pct']:.1f}\\%}} \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out = out_dir / f"{algo}_hp_consistency.tex"
    out.write_text("\n".join(lines))
    return out


def emit_rmse_consistency_latex(rows: list, kw_search: dict,
                                final_desc: dict, algo: str,
                                out_dir: Path) -> Path:
    """RMSE tutarlilik + Kruskal-Wallis + Mann-Whitney U + Holm LaTeX tablosu."""
    # Tablo A: Per-window summary (Y2..Y6) — search RMSE 30-run dist + final RMSE
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Walk-forward per-window RMSE summary ({algo.upper()}): "
        rf"30-run search RMSE distribution and single-shot final retrain RMSE on "
        rf"full 6-year data.}}",
        rf"\label{{tab:wf_rmse_summary_{algo}}}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"Window & Search median & Search mean & Search std & Final RMSE & \%diff vs Y6 (search) & \%diff vs Y6 (final) \\",
        r"\hline",
    ]
    # Y6 ref icin medyan ve final
    y6_search_median = None
    y6_final = None
    for r in rows:
        if r.get("window") == "Y6":
            y6_search_median = r.get("search_rmse")
            y6_final = r.get("final_rmse")
            break
    # Search median'i 30-koSu dagilimindan al
    ph_lookup = {ph["window"]: ph for ph in kw_search.get("post_hoc", [])}
    if y6_search_median is None and ph_lookup:
        # Y6 dagilimini posthoc icinde tutmuyoruz; fallback
        pass

    # Y6 statistikleri kw_search'in groups verisinden cikarilamadigi icin
    # results'a referansli olarak istatistik tablosunu pencere bazli ayrica hesapla.
    # rows icindeki window 'Y6' satirinin search_rmse'si best_loss; 30-koSu mean/std
    # icin algo_final_results JSON'inin statistics alani gerek. Cagiri zamani okunsun.
    import json
    from pathlib import Path
    y6_summary_path = Path(f"{algo}_final_results/analysis_summary.json")
    y6_stats = {}
    if y6_summary_path.exists():
        with open(y6_summary_path, "rb") as f:
            raw = f.read().replace(b"\x00", b"").strip()
        try:
            y6j = json.loads(raw)
            y6_stats = y6j.get("statistics", {})
            y6_all_losses = y6j.get("all_losses", [])
            import numpy as _np
            y6_search_median = float(_np.median(y6_all_losses)) if y6_all_losses else y6_search_median
        except Exception:
            pass

    for r in rows:
        w = r.get("window")
        if w == "Y6":
            search_med = y6_search_median if y6_search_median else r.get("search_rmse", 0)
            search_mean = y6_stats.get("mean", search_med)
            search_std  = y6_stats.get("std", 0)
        else:
            ph = ph_lookup.get(w, {})
            search_med = ph.get("median_y_i", r.get("search_rmse", 0))
            search_mean = ph.get("mean_y_i", r.get("search_rmse", 0))
            search_std = ph.get("std_y_i", 0)
        f = r.get("final_rmse")
        pct_s = (search_med - y6_search_median) / y6_search_median * 100 if y6_search_median else 0
        pct_f = (f - y6_final) / y6_final * 100 if (f is not None and y6_final) else 0
        f_str = f"{f:.5f}" if f is not None else "-"
        lines.append(
            f"{w:<4s} & {search_med:.5f} & {search_mean:.5f} & {search_std:.5f} & "
            f"{f_str} & {pct_s:+.1f}\\% & {pct_f:+.1f}\\% \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        r"",
    ]

    # Tablo B: Kruskal-Wallis + Mann-Whitney U + Holm
    if "error" not in kw_search:
        lines += [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{Walk-forward statistical analysis ({algo.upper()}): "
            rf"Kruskal-Wallis test on 30-run search RMSE distributions across "
            rf"{kw_search['n_windows']} windows ($n={kw_search['n_total']}$), with "
            rf"post-hoc Mann-Whitney U vs Y6 (Phase 1), Holm-corrected.}}",
            rf"\label{{tab:wf_stat_{algo}}}",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            rf"\multicolumn{{5}}{{l}}{{\textbf{{Kruskal-Wallis: H = {kw_search['kruskal_H']:.4f}, "
            rf"p = {kw_search['kruskal_p']:.3e}, $n$ per window = "
            rf"{kw_search['n_per_window']}}}}} \\\\",
            r"\hline",
            r"Window vs Y6 & Mann-Whitney U & p (raw) & p (Holm-corr.) & Significant? \\",
            r"\hline",
        ]
        for ph in kw_search.get("post_hoc", []):
            sig = "Yes" if ph.get("significant_at_005") else "No"
            lines.append(
                f"{ph['window']:<4s} & {ph['U']:.1f} & "
                f"{ph['p_raw']:.3e} & {ph['p_holm']:.3e} & {sig} \\\\"
            )
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    out = out_dir / f"{algo}_rmse_consistency.tex"
    out.write_text("\n".join(lines))
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", required=True,
                        help="Algoritma adi (ga, fno, raindrop, ...)")
    parser.add_argument("--windows", default="Y2,Y3,Y4,Y5,Y6",
                        help="Pencere listesi, default: Y2,Y3,Y4,Y5,Y6")
    args = parser.parse_args()

    windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    out_dir = Path(f"walk_forward_compare")
    out_dir.mkdir(exist_ok=True)

    # 1) Tum pencere sonuclarini yukle
    results = {}
    for w in windows:
        r = load_window_result(args.algo, w)
        results[w] = r

    valid_n = sum(1 for v in results.values() if v is not None)
    logger.info(f"Yuklenen pencere sayisi: {valid_n}/{len(windows)}")

    if valid_n < 2:
        logger.error("En az 2 pencere gerek. Cikiliyor.")
        return 1

    # 2) HP tablosu
    rows = build_hp_table(results)
    metrics = hp_consistency_metrics(rows)

    logger.info("\n=== HP TUTARLILIK ===")
    for hp, m in metrics.items():
        if m["type"] == "categorical":
            logger.info(f"  {hp}: mode={m['mode']} "
                        f"({m['mode_frequency']*100:.0f}%), "
                        f"unique={m['unique_count']}")
        else:
            logger.info(f"  {hp}: mean={m['mean']:.4f}, "
                        f"std={m['std']:.4f}, CV%={m['cv_pct']:.1f}")

    # 3) Kruskal-Wallis (n=150) + post-hoc Mann-Whitney U + Holm
    logger.info("\n=== SEARCH RMSE: KRUSKAL-WALLIS + POST-HOC ===")
    kw_search = kruskal_wallis_search(results)
    wilcoxon_search = kw_search  # geri uyumluluk icin ayni isim, JSON cikis
    if "error" in kw_search:
        logger.warning(f"Kruskal-Wallis: {kw_search['error']}")
    else:
        logger.info(f"  Test: {kw_search['test_name']}")
        logger.info(f"  Kruskal-Wallis H={kw_search['kruskal_H']:.4f}, "
                    f"p={kw_search['kruskal_p']:.3e}")
        logger.info(f"  -> {kw_search['interpretation']}")
        logger.info(f"\n  Post-hoc Mann-Whitney U (Y_i vs Y6, Holm-corrected):")
        logger.info(f"  {'Window':<8} {'U':<8} {'p_raw':<12} {'p_Holm':<12} "
                    f"{'median Y_i':<11} {'%diff Y6':<8} {'Sig':<5}")
        for ph in kw_search["post_hoc"]:
            sig = "EVET" if ph["significant_at_005"] else "hayir"
            logger.info(f"  {ph['window']:<8} {ph['U']:<8.1f} "
                        f"{ph['p_raw']:<12.3e} {ph['p_holm']:<12.3e} "
                        f"{ph['median_y_i']:<11.5f} "
                        f"{ph['pct_diff_vs_y6']:+8.1f} {sig:<5}")

    logger.info("\n=== FINAL RMSE: DESCRIPTIVE (n=1 per pencere) ===")
    final_desc = final_rmse_descriptive(results)
    wilcoxon_final = final_desc  # geri uyumluluk
    if "error" in final_desc:
        logger.warning(f"Final descriptive: {final_desc['error']}")
    else:
        logger.info(f"  n={final_desc['n']} pencere, Y6={final_desc['y6']:.5f}")
        logger.info(f"  mean={final_desc['mean']:.5f}, median={final_desc['median']:.5f}, "
                    f"std={final_desc['std']:.5f}")
        logger.info(f"  Max %diff vs Y6: {final_desc['max_pct_diff_vs_y6']:.1f}%")
        for v in final_desc["values"]:
            pct = v.get("pct_diff_vs_y6")
            pct_str = f"{pct:+.1f}%" if pct is not None else "ref"
            logger.info(f"    {v['window']}: final={v['final_rmse']:.5f} ({pct_str})")
        logger.info(f"  Not: {final_desc['note']}")

    # 4) Boxplot
    plot_path = plot_rmse_boxplot(results, args.algo, out_dir)
    logger.info(f"Grafik: {plot_path}")

    # 5) LaTeX tablolar
    hp_tex = emit_hp_consistency_latex(rows, metrics, args.algo, out_dir)
    rmse_tex = emit_rmse_consistency_latex(rows, wilcoxon_search,
                                           wilcoxon_final, args.algo, out_dir)
    logger.info(f"LaTeX HP tablosu: {hp_tex}")
    logger.info(f"LaTeX RMSE tablosu: {rmse_tex}")

    # 6) CSV ciktilar
    import csv
    hp_csv = out_dir / f"{args.algo}_hp_table.csv"
    with open(hp_csv, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    # 7) Toplu ozet JSON
    summary = {
        "algorithm": args.algo,
        "windows": windows,
        "rows": rows,
        "hp_consistency": metrics,
        "wilcoxon_search": wilcoxon_search,
        "wilcoxon_final": wilcoxon_final,
        "created_at": datetime.now().isoformat(),
    }
    sum_path = out_dir / f"{args.algo}_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Ozet JSON: {sum_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
