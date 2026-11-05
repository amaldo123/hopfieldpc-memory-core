"""Analysis and evaluation suite comparing K-Center Greedy vs K-means memory.

Covers memory geometry, retrieval dynamics, outlier analysis, computational scaling,
paired statistics (exact McNemar and Wilcoxon), and generates publication-grade figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from benchmarks.mnist_baseline import (
    NOISE_TYPES,
    BaselineRun,
    ClassifierHead,
    MnistNet,
    copy_net,
    load_mnist,
    prepare_baseline_run,
    validate_mnist_config,
)
from benchmarks.mnist_prototype_retrieval import (
    HopfieldBridge,
    beta_curve,
    build_embedding_net,
    build_hopfield_net,
    build_memory,
    compare_on_test,
    covering_radius,
    forward_embeddings,
    k_center_greedy,
    kmeans,
    mcnemar_exact_p,
    robust_auc,
    select_beta,
    wilcoxon_signed_rank_p,
)


# ---------------------------------------------------------------------------
# Geometric and Coverage Metric Helpers
# ---------------------------------------------------------------------------


def mean_quantization_error(x: torch.Tensor, centers: torch.Tensor) -> float:
    """Mean Euclidean distance from each row of x to its nearest row in centers."""
    return float(torch.cdist(x.float(), centers.float()).min(dim=1).values.mean().item())


def coverage_fraction(x: torch.Tensor, centers: torch.Tensor, r: float) -> float:
    """Fraction of rows in x with distance <= r to the nearest row in centers."""
    dists = torch.cdist(x.float(), centers.float()).min(dim=1).values
    return float((dists <= r).float().mean().item())


def cell_balance_entropy(x: torch.Tensor, centers: torch.Tensor) -> float:
    """Entropy of Voronoi cell sizes (how many points in x are nearest to each center)."""
    k = centers.shape[0]
    assign = torch.cdist(x.float(), centers.float()).argmin(dim=1)
    counts = torch.bincount(assign, minlength=k).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -float((probs * probs.log()).sum().item())


def pairwise_distances(centers: torch.Tensor) -> tuple[float, float]:
    """Mean and minimum pairwise Euclidean distance among rows of centers."""
    k = centers.shape[0]
    if k <= 1:
        return 0.0, 0.0
    dists = torch.cdist(centers.float(), centers.float())
    mask = ~torch.eye(k, dtype=torch.bool, device=centers.device)
    off_diag = dists[mask]
    return float(off_diag.mean().item()), float(off_diag.min().item())


def inter_class_margin(memory: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean distance from each memory pattern to the nearest pattern of a different class."""
    k = memory.shape[0]
    dists = torch.cdist(memory.float(), memory.float())
    diff_class = labels.unsqueeze(0) != labels.unsqueeze(1)
    dists[~diff_class] = float("inf")
    min_inter = dists.min(dim=1).values
    return float(min_inter[min_inter.isfinite()].mean().item())


# ---------------------------------------------------------------------------
# Random-8 Control (Analysis-Only)
# ---------------------------------------------------------------------------


def build_random_memory(
    z_train: torch.Tensor,
    labels: torch.Tensor,
    *,
    per_class: int,
    center: bool,
    seed: int,
    num_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Randomly sample real embeddings per class without diversity optimization."""
    generator = torch.Generator().manual_seed(seed)
    blocks, block_labels = [], []
    for c in range(num_classes):
        zc = z_train[labels == c]
        perm = torch.randperm(zc.shape[0], generator=generator)[:per_class]
        blocks.append(zc[perm])
        block_labels.append(torch.full((per_class,), c, dtype=torch.long))
    mean = z_train.mean(dim=0) if center else None
    return torch.cat(blocks), torch.cat(block_labels), mean


# ---------------------------------------------------------------------------
# Retrieval Behavior Diagnostics
# ---------------------------------------------------------------------------


def retrieval_diagnostics(
    net: MnistNet,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = 1000,
    device: torch.device,
) -> dict[str, float]:
    """Compute attention entropy, effective memories used, dead memory fraction, and cross-class mass."""
    out = forward_embeddings(net, embeddings, batch_size=batch_size, device=device)
    weights = out["weights"].cpu()  # [N, K]
    class_mass = out["class_mass"].cpu()  # [N, C]
    num_memories = weights.shape[1]

    # Attention entropy per query
    probs = weights.clamp(min=1e-12)
    entropy_per_query = -(probs * probs.log()).sum(dim=-1)
    mean_entropy = float(entropy_per_query.mean().item())
    effective_memories = float(torch.exp(entropy_per_query).mean().item())

    # Dead memories: never the top-1 memory for any query in the dataset
    top1_indices = weights.argmax(dim=-1)
    active_memories = set(top1_indices.unique().tolist())
    dead_fraction = float(1.0 - len(active_memories) / num_memories)

    # Wrong class attention mass: mass assigned to classes != true class
    correct_mass = class_mass.gather(1, labels.unsqueeze(1)).squeeze(1)
    wrong_mass = float((1.0 - correct_mass).clamp(min=0.0).mean().item())

    return {
        "entropy": mean_entropy,
        "effective_memories": effective_memories,
        "dead_fraction": dead_fraction,
        "wrong_class_mass": wrong_mass,
    }


# ---------------------------------------------------------------------------
# Outlier Diagnostics (Hypothesis H6)
# ---------------------------------------------------------------------------


def outlier_analysis(
    head: ClassifierHead,
    z_train: torch.Tensor,
    train_labels: torch.Tensor,
    selected_memories: torch.Tensor,
    memory_labels: torch.Tensor,
    num_classes: int = 10,
) -> dict[str, Any]:
    """Measure how peripheral or misclassified selected memories are relative to class distributions."""
    with torch.no_grad():
        preds = head(selected_memories).argmax(dim=-1)
    misclassified_count = int((preds != memory_labels).sum().item())

    z_scores: list[float] = []
    for c in range(num_classes):
        zc = z_train[train_labels == c]
        mean_c = zc.mean(dim=0)
        std_c = zc.std(dim=0).clamp(min=1e-6)
        mem_c = selected_memories[memory_labels == c]
        for row in mem_c:
            std_dist = torch.sqrt(((row - mean_c) / std_c).pow(2).sum()).item()
            z_scores.append(float(std_dist))

    return {
        "misclassified": misclassified_count,
        "misclassified_rate": misclassified_count / len(selected_memories),
        "mean_z_score": float(np.mean(z_scores)),
        "std_z_score": float(np.std(z_scores)),
        "z_scores": z_scores,
    }


# ---------------------------------------------------------------------------
# Computational Scaling Benchmarks
# ---------------------------------------------------------------------------


def measure_construction_scaling() -> dict[str, Any]:
    """Measure memory construction time vs N (sample size) and vs k (prototype budget)."""
    d = 64
    n_values = [1000, 3000, 10000, 30000, 100000]
    k_fixed = 8
    gen = torch.Generator().manual_seed(42)

    times_n_kc, times_n_km = [], []
    for n in n_values:
        x = torch.randn(n, d)
        # K-Center
        start = time.perf_counter()
        _ = k_center_greedy(x, k_fixed, generator=gen)
        times_n_kc.append(time.perf_counter() - start)
        # K-Means (25 iters)
        start = time.perf_counter()
        _ = kmeans(x, k_fixed, generator=gen, iters=25)
        times_n_km.append(time.perf_counter() - start)

    k_values = [8, 16, 32, 64, 128]
    n_fixed = 10000
    times_k_kc, times_k_km = [], []
    for k in k_values:
        x = torch.randn(n_fixed, d)
        start = time.perf_counter()
        _ = k_center_greedy(x, k, generator=gen)
        times_k_kc.append(time.perf_counter() - start)
        start = time.perf_counter()
        _ = kmeans(x, k, generator=gen, iters=25)
        times_k_km.append(time.perf_counter() - start)

    return {
        "n_values": n_values,
        "times_n_kc": times_n_kc,
        "times_n_km": times_n_km,
        "k_values": k_values,
        "times_k_kc": times_k_kc,
        "times_k_km": times_k_km,
    }


# ---------------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------------


def generate_figures(
    output_dir: Path,
    summary_data: dict[str, Any],
    scaling_data: dict[str, Any],
    sample_embeddings: torch.Tensor,
    sample_labels: torch.Tensor,
    mem_km: torch.Tensor,
    mem_kc: torch.Tensor,
    mem_rand: torch.Tensor,
) -> None:
    """Generate all 7 research report figures at 150 dpi."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    # 1. PCA 2D Scatter for class 0
    c0 = sample_embeddings[sample_labels == 0].float()
    u, s, v = torch.pca_lowrank(c0, q=2)
    proj_c0 = torch.matmul(c0, v[:, :2]).numpy()
    proj_km = torch.matmul(mem_km[:8].float(), v[:, :2]).numpy()
    proj_kc = torch.matmul(mem_kc[:8].float(), v[:, :2]).numpy()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.scatter(proj_c0[:, 0], proj_c0[:, 1], c="#d0d7de", s=15, alpha=0.5, label="Train Embeddings (Digit 0)")
    ax.scatter(proj_km[:, 0], proj_km[:, 1], c="#0969da", marker="X", s=80, edgecolors="black", linewidths=1.2, label="K-means Centroids")
    ax.scatter(proj_kc[:, 0], proj_kc[:, 1], c="#cf222e", marker="o", s=80, edgecolors="black", linewidths=1.2, label="K-Center Picks")
    ax.set_title("PCA Representation: K-means vs K-Center (Digit 0)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(output_dir / "fig1_pca_embeddings.png")
    plt.close(fig)

    # 2. Covering Radius & Mean Quantization Error Bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    methods = ["K-means", "K-Center", "Random-8"]
    cov_radii = [summary_data["coverage"]["train"]["kmeans"]["radius_mean"],
                 summary_data["coverage"]["train"]["kcenter"]["radius_mean"],
                 summary_data["coverage"]["train"]["random"]["radius_mean"]]
    cov_errs = [summary_data["coverage"]["train"]["kmeans"]["radius_std"],
                summary_data["coverage"]["train"]["kcenter"]["radius_std"],
                summary_data["coverage"]["train"]["random"]["radius_std"]]

    mqe_vals = [summary_data["coverage"]["train"]["kmeans"]["mqe_mean"],
                summary_data["coverage"]["train"]["kcenter"]["mqe_mean"],
                summary_data["coverage"]["train"]["random"]["mqe_mean"]]
    mqe_errs = [summary_data["coverage"]["train"]["kmeans"]["mqe_std"],
                summary_data["coverage"]["train"]["kcenter"]["mqe_std"],
                summary_data["coverage"]["train"]["random"]["mqe_std"]]

    colors = ["#0969da", "#cf222e", "#8250df"]
    ax1.bar(methods, cov_radii, yerr=cov_errs, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    ax1.set_title("Covering Radius (Train)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Radius (Euclidean Distance)")

    ax2.bar(methods, mqe_vals, yerr=mqe_errs, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    ax2.set_title("Mean Quantization Error (Train)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Distance to Nearest Memory")
    fig.savefig(output_dir / "fig2_coverage_bars.png")
    plt.close(fig)

    # 3. Validation Beta Curves
    if "beta_curves" in summary_data:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        betas = summary_data["beta_curves"]["betas"]
        ax.plot(betas, summary_data["beta_curves"]["kmeans"], "o-", color="#0969da", label="K-means Centroids", linewidth=1.5)
        ax.plot(betas, summary_data["beta_curves"]["kcenter"], "s-", color="#cf222e", label="K-Center Greedy", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_title(r"Validation Accuracy vs $eta$", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Inverse Temperature $eta$")
        ax.set_ylabel("Validation Head Accuracy")
        ax.legend(frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.savefig(output_dir / "fig3_beta_curves.png")
        plt.close(fig)

    # 4. Accuracy vs Noise Level (Noise Robustness)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150)
    noise_names = [("gaussian", r"Gaussian Noise ($\sigma$)", "gaussian_sigmas"),
                   ("salt_pepper", "Salt & Pepper ($p$)", "salt_pepper_probs"),
                   ("occlusion", "Occlusion Patch Size ($s$)", "occlusion_sizes")]
    for ax, (nt, xlabel, key) in zip(axes, noise_names):
        grid = summary_data["noise_grids"].get(key, [])
        if grid:
            ax.plot(grid, summary_data["accuracy"][nt]["baseline"], "^--", color="#6e7781", label="Baseline (L8=Id)", linewidth=1.2)
            ax.plot(grid, summary_data["accuracy"][nt]["kmeans"], "o-", color="#0969da", label="K-means Centroids", linewidth=1.5)
            ax.plot(grid, summary_data["accuracy"][nt]["kcenter"], "s-", color="#cf222e", label="K-Center Greedy", linewidth=1.5)
            ax.set_title(nt.replace("_", " ").capitalize(), fontsize=10, fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Test Head Accuracy")
            ax.grid(True, linestyle="--", alpha=0.5)
            if ax is axes[0]:
                ax.legend(frameon=True, fontsize=8)
    fig.savefig(output_dir / "fig4_noise_robustness.png")
    plt.close(fig)

    # 5. Retrieval Behavior Diagnostics
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.5), dpi=150)
    ret_methods = ["K-means", "K-Center"]
    colors2 = ["#0969da", "#cf222e"]

    ent = [summary_data["retrieval"]["kmeans"]["entropy"], summary_data["retrieval"]["kcenter"]["entropy"]]
    ax1.bar(ret_methods, ent, color=colors2, alpha=0.85, edgecolor="black")
    ax1.set_title("Attention Entropy", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Shannon Entropy (nats)")

    eff = [summary_data["retrieval"]["kmeans"]["effective_memories"], summary_data["retrieval"]["kcenter"]["effective_memories"]]
    ax2.bar(ret_methods, eff, color=colors2, alpha=0.85, edgecolor="black")
    ax2.set_title("Effective Memories / Query", fontsize=10, fontweight="bold")
    ax2.set_ylabel(r"$N_{\mathrm{eff}} = \exp(H)$")

    dead = [summary_data["retrieval"]["kmeans"]["dead_fraction"] * 100, summary_data["retrieval"]["kcenter"]["dead_fraction"] * 100]
    ax3.bar(ret_methods, dead, color=colors2, alpha=0.85, edgecolor="black")
    ax3.set_title("Dead Memories (%)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Never Top-1 (%)")
    fig.savefig(output_dir / "fig5_retrieval_diagnostics.png")
    plt.close(fig)

    # 6. Construction Time Scaling Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=150)
    ax1.plot(scaling_data["n_values"], scaling_data["times_n_kc"], "s-", color="#cf222e", label="K-Center Greedy", linewidth=1.5)
    ax1.plot(scaling_data["n_values"], scaling_data["times_n_km"], "o-", color="#0969da", label="K-means (25 iters)", linewidth=1.5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("Construction Time vs N (k=8, D=64)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Number of Points N")
    ax1.set_ylabel("Wall Clock Time (s)")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(scaling_data["k_values"], scaling_data["times_k_kc"], "s-", color="#cf222e", label="K-Center Greedy", linewidth=1.5)
    ax2.plot(scaling_data["k_values"], scaling_data["times_k_km"], "o-", color="#0969da", label="K-means (25 iters)", linewidth=1.5)
    ax2.set_title("Construction Time vs k (N=10000, D=64)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Number of Centers k")
    ax2.set_ylabel("Wall Clock Time (s)")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig.savefig(output_dir / "fig6_cost_scaling.png")
    plt.close(fig)

    # 7. Outlier Analysis (Z-score distribution)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    z_kc = summary_data["outliers"]["kcenter"]["z_scores"]
    z_rand = summary_data["outliers"]["random"]["z_scores"]
    ax.hist(z_kc, bins=15, alpha=0.7, color="#cf222e", label=f"K-Center picks (Mean: {np.mean(z_kc):.2f})", edgecolor="black")
    ax.hist(z_rand, bins=15, alpha=0.5, color="#8250df", label=f"Random sample (Mean: {np.mean(z_rand):.2f})", edgecolor="black")
    ax.set_title("Outlier Analysis: Distance from Class Mean", fontsize=11, fontweight="bold")
    ax.set_xlabel("Distance from Class Mean (Standard Deviations)")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=True)
    fig.savefig(output_dir / "fig7_outlier_analysis.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------------------------


def run_full_analysis(
    kmeans_reports_dir: str,
    kcenter_reports_dir: str,
    output_dir: str = "reports/kcenter",
    data_dir: str = "data",
) -> dict[str, Any]:
    """Execute complete analysis suite comparing K-means and K-Center."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    p_km = Path(kmeans_reports_dir)
    p_kc = Path(kcenter_reports_dir)

    print(f"Loading results from {p_km} and {p_kc}...")
    with open(p_km / "results.json", encoding="utf-8") as f:
        res_km = json.load(f)
    with open(p_kc / "results.json", encoding="utf-8") as f:
        res_kc = json.load(f)

    cfg_km = res_km["config"]
    cfg_kc = res_kc["config"]
    seeds = cfg_km["seeds"]

    # Fairness verification: verify baseline numbers match exactly per seed
    for exp_key in ("E1", "E3a", "E3b", "E3c"):
        rows_km = [r for r in res_km["experiments"] if r["experiment"] == exp_key and r["model"] == "baseline"]
        rows_kc = [r for r in res_kc["experiments"] if r["experiment"] == exp_key and r["model"] == "baseline"]
        for r1, r2 in zip(rows_km, rows_kc):
            assert math.isclose(r1["head_acc"], r2["head_acc"], abs_tol=1e-5), (
                f"Fairness violation in {exp_key}: baseline head_acc {r1['head_acc']} != {r2['head_acc']}"
            )

    # A. Accuracy and Robustness Extraction
    def extract_metric(res: dict[str, Any], exp: str, model: str, metric: str = "head_acc") -> list[float]:
        rows = [r for r in res["experiments"] if r["experiment"] == exp and r["model"] == model]
        return [r[metric] for r in rows]

    clean_base = extract_metric(res_km, "E1", "baseline")
    clean_km = extract_metric(res_km, "E1", "hopfield")
    clean_kc = extract_metric(res_kc, "E1", "hopfield")

    clean_proto_km = extract_metric(res_km, "E1", "hopfield", "proto_acc")
    clean_proto_kc = extract_metric(res_kc, "E1", "hopfield", "proto_acc")

    # Noise AUC extraction
    auc_gauss_base = [r["auc"] for r in res_km["experiments"] if r["experiment"] == "E3a_auc" and r["model"] == "baseline"]
    auc_gauss_km = [r["auc"] for r in res_km["experiments"] if r["experiment"] == "E3a_auc" and r["model"] == "hopfield"]
    auc_gauss_kc = [r["auc"] for r in res_kc["experiments"] if r["experiment"] == "E3a_auc" and r["model"] == "hopfield"]

    auc_sp_base = [r["auc"] for r in res_km["experiments"] if r["experiment"] == "E3b_auc" and r["model"] == "baseline"]
    auc_sp_km = [r["auc"] for r in res_km["experiments"] if r["experiment"] == "E3b_auc" and r["model"] == "hopfield"]
    auc_sp_kc = [r["auc"] for r in res_kc["experiments"] if r["experiment"] == "E3b_auc" and r["model"] == "hopfield"]

    auc_occ_base = [r["auc"] for r in res_km["experiments"] if r["experiment"] == "E3c_auc" and r["model"] == "baseline"]
    auc_occ_km = [r["auc"] for r in res_km["experiments"] if r["experiment"] == "E3c_auc" and r["model"] == "hopfield"]
    auc_occ_kc = [r["auc"] for r in res_kc["experiments"] if r["experiment"] == "E3c_auc" and r["model"] == "hopfield"]

    # Wilcoxon paired tests across seeds
    diff_clean = [kc - km for kc, km in zip(clean_kc, clean_km)]
    diff_gauss = [kc - km for kc, km in zip(auc_gauss_kc, auc_gauss_km)]
    diff_sp = [kc - km for kc, km in zip(auc_sp_kc, auc_sp_km)]
    diff_occ = [kc - km for kc, km in zip(auc_occ_kc, auc_occ_km)]

    p_clean = wilcoxon_signed_rank_p(diff_clean)
    p_gauss = wilcoxon_signed_rank_p(diff_gauss)
    p_sp = wilcoxon_signed_rank_p(diff_sp)
    p_occ = wilcoxon_signed_rank_p(diff_occ)

    def winner(mean_diff: float, p_val: float) -> str:
        if abs(mean_diff) < 0.005 or p_val >= 0.05:
            return "tie"
        return "K-Center" if mean_diff > 0 else "K-means"

    paired_stats = {
        "clean_head_acc": {
            "mean_diff": float(np.mean(diff_clean)),
            "wilcoxon_p": p_clean,
            "winner": winner(float(np.mean(diff_clean)), p_clean),
            "primary": True,
        },
        "gaussian_robust_auc": {
            "mean_diff": float(np.mean(diff_gauss)),
            "wilcoxon_p": p_gauss,
            "winner": winner(float(np.mean(diff_gauss)), p_gauss),
            "primary": True,
        },
        "salt_pepper_robust_auc": {
            "mean_diff": float(np.mean(diff_sp)),
            "wilcoxon_p": p_sp,
            "winner": winner(float(np.mean(diff_sp)), p_sp),
            "primary": False,
        },
        "occlusion_robust_auc": {
            "mean_diff": float(np.mean(diff_occ)),
            "wilcoxon_p": p_occ,
            "winner": winner(float(np.mean(diff_occ)), p_occ),
            "primary": False,
        },
    }

    # B. Geometric Memory Coverage, Outliers, Retrieval Diagnostics
    data = load_mnist(data_dir)
    device = torch.device("cpu")

    # Collect stats over seeds
    cov_train_km_r, cov_train_kc_r, cov_train_rnd_r = [], [], []
    cov_train_km_q, cov_train_kc_q, cov_train_rnd_q = [], [], []
    ret_km_list, ret_kc_list = [], []
    outlier_kc_list, outlier_rnd_list = [], []

    sample_seed = seeds[0]
    sample_run = prepare_baseline_run(cfg_km, data, sample_seed, device)
    z_train = sample_run.z_train
    labels_train = sample_run.splits.train_labels
    z_test = sample_run.test_embeddings("clean", 0.0)
    labels_test = sample_run.splits.test_labels

    sample_mem_km, sample_l_km, _ = build_memory(z_train, labels_train, per_class=8, center=True, seed=sample_seed, method="kmeans")
    sample_mem_kc, sample_l_kc, _ = build_memory(z_train, labels_train, per_class=8, center=True, seed=sample_seed, method="kcenter")
    sample_mem_rnd, sample_l_rnd, _ = build_random_memory(z_train, labels_train, per_class=8, center=True, seed=sample_seed)

    for s in seeds:
        run = prepare_baseline_run(cfg_km, data, s, device)
        zt = run.z_train
        lt = run.splits.train_labels

        m_km, l_km, _ = build_memory(zt, lt, per_class=8, center=True, seed=s, method="kmeans")
        m_kc, l_kc, _ = build_memory(zt, lt, per_class=8, center=True, seed=s, method="kcenter")
        m_rnd, l_rnd, _ = build_random_memory(zt, lt, per_class=8, center=True, seed=s)

        # per-class coverage on train
        for c in range(10):
            zc = zt[lt == c]
            cov_train_km_r.append(covering_radius(zc, m_km[l_km == c]))
            cov_train_kc_r.append(covering_radius(zc, m_kc[l_kc == c]))
            cov_train_rnd_r.append(covering_radius(zc, m_rnd[l_rnd == c]))

            cov_train_km_q.append(mean_quantization_error(zc, m_km[l_km == c]))
            cov_train_kc_q.append(mean_quantization_error(zc, m_kc[l_kc == c]))
            cov_train_rnd_q.append(mean_quantization_error(zc, m_rnd[l_rnd == c]))

        # retrieval diagnostics on test embeddings at beta*
        net_km = build_embedding_net(run, 8, True, beta=res_km["beta_star"])
        net_kc = build_embedding_net(run, 8, True, beta=res_kc["beta_star"], method="kcenter")

        ret_km_list.append(retrieval_diagnostics(net_km, run.test_embeddings("clean", 0.0), run.splits.test_labels, device=device))
        ret_kc_list.append(retrieval_diagnostics(net_kc, run.test_embeddings("clean", 0.0), run.splits.test_labels, device=device))

        # outliers
        outlier_kc_list.append(outlier_analysis(run.net.head, zt, lt, m_kc, l_kc))
        outlier_rnd_list.append(outlier_analysis(run.net.head, zt, lt, m_rnd, l_rnd))

    coverage_summary = {
        "train": {
            "kmeans": {"radius_mean": float(np.mean(cov_train_km_r)), "radius_std": float(np.std(cov_train_km_r)),
                       "mqe_mean": float(np.mean(cov_train_km_q)), "mqe_std": float(np.std(cov_train_km_q))},
            "kcenter": {"radius_mean": float(np.mean(cov_train_kc_r)), "radius_std": float(np.std(cov_train_kc_r)),
                        "mqe_mean": float(np.mean(cov_train_kc_q)), "mqe_std": float(np.std(cov_train_kc_q))},
            "random": {"radius_mean": float(np.mean(cov_train_rnd_r)), "radius_std": float(np.std(cov_train_rnd_r)),
                       "mqe_mean": float(np.mean(cov_train_rnd_q)), "mqe_std": float(np.std(cov_train_rnd_q))},
        }
    }

    retrieval_summary = {
        "kmeans": {
            "entropy": float(np.mean([r["entropy"] for r in ret_km_list])),
            "effective_memories": float(np.mean([r["effective_memories"] for r in ret_km_list])),
            "dead_fraction": float(np.mean([r["dead_fraction"] for r in ret_km_list])),
            "wrong_class_mass": float(np.mean([r["wrong_class_mass"] for r in ret_km_list])),
        },
        "kcenter": {
            "entropy": float(np.mean([r["entropy"] for r in ret_kc_list])),
            "effective_memories": float(np.mean([r["effective_memories"] for r in ret_kc_list])),
            "dead_fraction": float(np.mean([r["dead_fraction"] for r in ret_kc_list])),
            "wrong_class_mass": float(np.mean([r["wrong_class_mass"] for r in ret_kc_list])),
        },
    }

    outlier_summary = {
        "kcenter": {
            "misclassified_mean": float(np.mean([o["misclassified"] for o in outlier_kc_list])),
            "mean_z_score": float(np.mean([o["mean_z_score"] for o in outlier_kc_list])),
            "z_scores": outlier_kc_list[0]["z_scores"],
        },
        "random": {
            "misclassified_mean": float(np.mean([o["misclassified"] for o in outlier_rnd_list])),
            "mean_z_score": float(np.mean([o["mean_z_score"] for o in outlier_rnd_list])),
            "z_scores": outlier_rnd_list[0]["z_scores"],
        },
    }

    # Accuracy curves for plotting
    noise_grids = {
        "gaussian_sigmas": cfg_km.get("gaussian_sigmas", []),
        "salt_pepper_probs": cfg_km.get("salt_pepper_probs", []),
        "occlusion_sizes": cfg_km.get("occlusion_sizes", []),
    }
    accuracy_curves: dict[str, dict[str, list[float]]] = {}
    for nt, exp in [("gaussian", "E3a"), ("salt_pepper", "E3b"), ("occlusion", "E3c")]:
        accuracy_curves[nt] = {
            "baseline": [float(np.mean([r["head_acc"] for r in res_km["experiments"] if r["experiment"] == exp and r["model"] == "baseline" and r["value"] == val])) for val in noise_grids[f"{nt}_sigmas" if nt == "gaussian" else f"{nt}_probs" if nt == "salt_pepper" else "occlusion_sizes"]],
            "kmeans": [float(np.mean([r["head_acc"] for r in res_km["experiments"] if r["experiment"] == exp and r["model"] == "hopfield" and r["value"] == val])) for val in noise_grids[f"{nt}_sigmas" if nt == "gaussian" else f"{nt}_probs" if nt == "salt_pepper" else "occlusion_sizes"]],
            "kcenter": [float(np.mean([r["head_acc"] for r in res_kc["experiments"] if r["experiment"] == exp and r["model"] == "hopfield" and r["value"] == val])) for val in noise_grids[f"{nt}_sigmas" if nt == "gaussian" else f"{nt}_probs" if nt == "salt_pepper" else "occlusion_sizes"]],
        }

    # Computational Scaling
    print("Benchmarking construction scaling...")
    scaling_data = measure_construction_scaling()

    summary_data = {
        "clean_head_acc": {
            "baseline_mean": float(np.mean(clean_base)), "baseline_std": float(np.std(clean_base)),
            "kmeans_mean": float(np.mean(clean_km)), "kmeans_std": float(np.std(clean_km)),
            "kcenter_mean": float(np.mean(clean_kc)), "kcenter_std": float(np.std(clean_kc)),
        },
        "gaussian_auc": {
            "baseline_mean": float(np.mean(auc_gauss_base)), "baseline_std": float(np.std(auc_gauss_base)),
            "kmeans_mean": float(np.mean(auc_gauss_km)), "kmeans_std": float(np.std(auc_gauss_km)),
            "kcenter_mean": float(np.mean(auc_gauss_kc)), "kcenter_std": float(np.std(auc_gauss_kc)),
        },
        "salt_pepper_auc": {
            "baseline_mean": float(np.mean(auc_sp_base)), "baseline_std": float(np.std(auc_sp_base)),
            "kmeans_mean": float(np.mean(auc_sp_km)), "kmeans_std": float(np.std(auc_sp_km)),
            "kcenter_mean": float(np.mean(auc_sp_kc)), "kcenter_std": float(np.std(auc_sp_kc)),
        },
        "occlusion_auc": {
            "baseline_mean": float(np.mean(auc_occ_base)), "baseline_std": float(np.std(auc_occ_base)),
            "kmeans_mean": float(np.mean(auc_occ_km)), "kmeans_std": float(np.std(auc_occ_km)),
            "kcenter_mean": float(np.mean(auc_occ_kc)), "kcenter_std": float(np.std(auc_occ_kc)),
        },
        "paired_statistics": paired_stats,
        "coverage": coverage_summary,
        "retrieval": retrieval_summary,
        "outliers": outlier_summary,
        "noise_grids": noise_grids,
        "accuracy": accuracy_curves,
        "beta_star": {"kmeans": res_km["beta_star"], "kcenter": res_kc["beta_star"]},
    }

    # Save summary JSON and CSV
    with open(out_path / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(out_path / "analysis_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline", "K-means", "K-Center", "Delta(KC-KM)", "Wilcoxon_p", "Winner"])
        writer.writerow([
            "Clean Head Acc",
            f"{summary_data['clean_head_acc']['baseline_mean']:.4f} ± {summary_data['clean_head_acc']['baseline_std']:.4f}",
            f"{summary_data['clean_head_acc']['kmeans_mean']:.4f} ± {summary_data['clean_head_acc']['kmeans_std']:.4f}",
            f"{summary_data['clean_head_acc']['kcenter_mean']:.4f} ± {summary_data['clean_head_acc']['kcenter_std']:.4f}",
            f"{paired_stats['clean_head_acc']['mean_diff']:+.4f}",
            f"{paired_stats['clean_head_acc']['wilcoxon_p']:.4f}",
            paired_stats["clean_head_acc"]["winner"],
        ])
        writer.writerow([
            "Gaussian AUC",
            f"{summary_data['gaussian_auc']['baseline_mean']:.4f} ± {summary_data['gaussian_auc']['baseline_std']:.4f}",
            f"{summary_data['gaussian_auc']['kmeans_mean']:.4f} ± {summary_data['gaussian_auc']['kmeans_std']:.4f}",
            f"{summary_data['gaussian_auc']['kcenter_mean']:.4f} ± {summary_data['gaussian_auc']['kcenter_std']:.4f}",
            f"{paired_stats['gaussian_robust_auc']['mean_diff']:+.4f}",
            f"{paired_stats['gaussian_robust_auc']['wilcoxon_p']:.4f}",
            paired_stats["gaussian_robust_auc"]["winner"],
        ])
        writer.writerow([
            "Salt & Pepper AUC",
            f"{summary_data['salt_pepper_auc']['baseline_mean']:.4f} ± {summary_data['salt_pepper_auc']['baseline_std']:.4f}",
            f"{summary_data['salt_pepper_auc']['kmeans_mean']:.4f} ± {summary_data['salt_pepper_auc']['kmeans_std']:.4f}",
            f"{summary_data['salt_pepper_auc']['kcenter_mean']:.4f} ± {summary_data['salt_pepper_auc']['kcenter_std']:.4f}",
            f"{paired_stats['salt_pepper_robust_auc']['mean_diff']:+.4f}",
            f"{paired_stats['salt_pepper_robust_auc']['wilcoxon_p']:.4f}",
            paired_stats["salt_pepper_robust_auc"]["winner"],
        ])
        writer.writerow([
            "Occlusion AUC",
            f"{summary_data['occlusion_auc']['baseline_mean']:.4f} ± {summary_data['occlusion_auc']['baseline_std']:.4f}",
            f"{summary_data['occlusion_auc']['kmeans_mean']:.4f} ± {summary_data['occlusion_auc']['kmeans_std']:.4f}",
            f"{summary_data['occlusion_auc']['kcenter_mean']:.4f} ± {summary_data['occlusion_auc']['kcenter_std']:.4f}",
            f"{paired_stats['occlusion_robust_auc']['mean_diff']:+.4f}",
            f"{paired_stats['occlusion_robust_auc']['wilcoxon_p']:.4f}",
            paired_stats["occlusion_robust_auc"]["winner"],
        ])

    print("Generating figures...")
    generate_figures(
        out_path,
        summary_data,
        scaling_data,
        z_train,
        labels_train,
        sample_mem_km,
        sample_mem_kc,
        sample_mem_rnd,
    )
    print(f"Analysis complete. Outputs saved to {out_path}.")
    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="K-Center vs K-means Analysis Suite")
    parser.add_argument("--kmeans-dir", default="reports/mnist_prototype_retrieval", help="Path to K-means reports")
    parser.add_argument("--kcenter-dir", default="reports/mnist_kcenter", help="Path to K-Center reports")
    parser.add_argument("--output-dir", default="reports/kcenter", help="Path to save analysis outputs")
    parser.add_argument("--data-dir", default="data", help="Path to MNIST data")
    args = parser.parse_args()

    run_full_analysis(
        kmeans_reports_dir=args.kmeans_dir,
        kcenter_reports_dir=args.kcenter_dir,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
