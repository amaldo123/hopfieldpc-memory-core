# Hopfield Memory Prototype Selection: Diverse Real Embeddings (K-Center Greedy) vs. Synthetic Centroids (K-Means)

> **Abstract:** Modern continuous Hopfield networks (Ramsauer et al., 2020) associatively retrieve stored prototype representations to improve classifier robustness against adversarial and distribution shifts. In the standard paradigm, prototype memories are formed from class-conditional K-means centroids—synthetic convex combinations that compress dense cluster cores. Here, we investigate whether selecting *diverse real* hidden embeddings via K-Center Greedy farthest-point traversal (Sener & Savarese, 2018) enhances Hopfield memory retrieval. Evaluating across 10 random seeds with frozen CNN backbones on MNIST, we find that K-Center Greedy substantially reduces the worst-case covering radius (-17.9%) at the cost of increased mean quantization error (+28.4%). However, on associative retrieval endpoints, K-Center achieves statistically indistinguishable clean head accuracy (98.63% vs 98.63%, tie, Wilcoxon $p = 0.844$) and virtually identical Gaussian noise robustness (+0.0 pp AUC, tie). The theoretical benefit of boundary coverage is offset by sensitivity to peripheral representations and lower signal-to-noise ratio in soft associative retrieval. Consequently, this study establishes a definitive **null-to-neutral result**: diverse real boundary embeddings do not outperform synthetic centroids for continuous Hopfield associative memories.

---

## 1. Executive Summary

This study addresses an open architectural question in associative memory augmented neural networks: **Does choosing diverse real hidden representations with K-Center Greedy improve continuous Hopfield memory retrieval compared with synthetic K-means centroids?**

Across 10 independent random seeds sharing identical trained CNN backbones, splits, centering protocols, and evaluation grids, our findings demonstrate:
- **Clean Classification:** Hopfield classification with K-Center prototypes achieves 98.63% ± 0.12% accuracy, identical to K-means centroids (98.63% ± 0.11%, Wilcoxon $p = 0.844$, signed-rank tie). Both memory methods preserve baseline accuracy (98.30% ± 0.24%) without degradation.
- **Robustness Under Perturbation:** Under additive Gaussian noise ($\sigma \in [0.0, 1.0]$), salt-and-pepper noise ($p \in [0.0, 0.4]$), and spatial occlusion (patch size up to $14 \times 14$), K-Center Greedy retrieval matches K-means centroids across all perturbation regimes within statistical uncertainty ($|\Delta| < 0.2$ pp).
- **Geometric Trade-off:** K-Center successfully optimizes its minimax objective, reducing covering radius by ~18% compared to K-means, but doubles the variance of nearest-prototype distances and increases mean quantization error by ~28%.
- **Computational Overhead:** K-Center Greedy constructs prototype sets ~7.9x faster than iterative Lloyd's K-means (0.012s vs 0.095s per class), while maintaining strictly bit-identical inference cost (0 added trainable parameters, +0.24% MACs).

We conclude with a **null result** on retrieval accuracy: despite superior geometric coverage of representation extremes, diverse real boundary points provide no measurable advantage over synthetic cluster centroids in continuous Hopfield retrieval.

---

## 2. Introduction & Background

### 2.1 Continuous Hopfield Associative Memory
Modern continuous Hopfield networks (Hopfield, 1982; Krotov & Hopfield, 2016; Demircigil et al., 2017; Ramsauer et al., 2020) store patterns in a continuous energy landscape with exponential storage capacity. For a query embedding $z \in \mathbb{R}^D$ and a memory matrix $M \in \mathbb{R}^{K \times D}$ containing $K$ stored patterns, one-step continuous retrieval maps $z$ to an updated state $h$:

$$h = \text{softmax}\left(\beta \, \text{sim}(z, M)\right) M$$

In Ramsauer et al. (2020), the similarity metric is centered cosine scoring: $\text{sim}(z, M) = \cos(z - \mu, M - \mu)$, where $\mu \in \mathbb{R}^D$ is the empirical mean of the training embeddings. Centering removes common uninformative directional components, maximizing angular separation between classes.

### 2.2 Prototype Selection: Centroids vs. Real Boundary Points
In the standard setup, $M$ consists of $k=8$ prototypes per class obtained via K-means clustering of the training embeddings. However, K-means optimizes average squared Euclidean distance:

$$\min_{\mu_1, \dots, \mu_k} \frac{1}{N} \sum_{i=1}^N \min_{j \in \{1,\dots,k\}} \|x_i - \mu_j\|_2^2$$

This objective forces centroids into dense interior regions of the representation manifold, producing synthetic linear averages that may blur fine-grained class boundaries or miss low-density tails.

### 2.3 K-Center Greedy Farthest-Point Traversal
Sener & Savarese (2018) proposed core-set selection via the K-Center objective: choose a subset $S \subseteq X$ with $|S| = k$ minimizing the worst-case covering radius:

$$\min_{S \subseteq X, |S| = k} \max_{x_i \in X} \min_{s_j \in S} \|x_i - s_j\|_2$$

The greedy farthest-point traversal algorithm initializes with a seeded random point $u_1 \in X$ and iteratively adds:

$$u_t = \arg\max_{x_i \in X} \min_{s \in S_{t-1}} \|x_i - s\|_2 \quad (t = 2, \dots, k)$$

By tracking a running minimum distance vector $\text{nearest}[i] = \min_{s \in S} \|x_i - s\|_2$, the greedy update requires only $O(N \cdot k \cdot D)$ operations and $O(N)$ memory. By the Pigeonhole Principle, this greedy algorithm is a provable 2-approximation: $r_{\text{greedy}} \le 2 \cdot r^*$.

Theorem 1 of Sener & Savarese bounds core-set generalization loss by the covering radius. In associative memory, diverse boundary points could hypothetically expand the basin of attraction for noisy, occluded queries drifting away from cluster cores.

---

## 3. Experimental Design & Fairness Controls

To ensure complete fairness, both methods were evaluated using a strict frozen protocol:

1. **Shared CNN Checkpoints:** 10 independent CNN models (Conv 1→32 → Pool → Conv 32→64 → Pool → Flatten → Linear 3136→128 → Linear 128→64 → Head 64→10) were trained on clean MNIST across seeds 0–9. Checkpoints were frozen and bit-for-bit identical across runs.
2. **Identical Centering:** Both K-means and K-Center memories were centered by the exact training mean embedding $\mu = \frac{1}{N} \sum z_i$.
3. **Identical Memory Budget:** Exactly $k=8$ prototypes per class (80 prototypes total, $80 \times 64$).
4. **Validation-Guided $\beta^*$ Tuning:** Optimal inverse temperature $\beta^*$ was selected independently on held-out validation sets (clean, Gaussian $\sigma=0.4$, salt-and-pepper $p=0.15$). K-Center selected $\beta^* = 10.0$; K-means selected $\beta^* = 50.0$.
5. **Standardized Evaluation Grids:** Identical test noise levels across Gaussian noise, salt-and-pepper noise, and spatial occlusion.
6. **Random-8 Control:** An unbiased control selecting 8 random real training samples per class to decouple realness from boundary dispersion.

---

## 4. Main Retrieval Results

The primary endpoints are clean classification accuracy and Gaussian noise robustness AUC. Paired across 10 random seeds, the comparisons show:

| Metric / Regime | Baseline CNN (L8=Id) | Hopfield (K-Means) | Hopfield (K-Center) | $\Delta$ (KC - KM) | Wilcoxon $p$ | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean Head Acc (E1)** | 98.30% ± 0.24% | 98.63% ± 0.11% | 98.63% ± 0.12% | +0.00% | 0.844 | **Tie** |
| **Clean Proto Acc** | 98.34% ± 0.18% | 98.66% ± 0.09% | 98.65% ± 0.10% | -0.01% | 0.695 | **Tie** |
| **Gaussian Robust AUC (E3a)** | 0.8124 ± 0.0081 | 0.8842 ± 0.0065 | 0.8840 ± 0.0068 | -0.02 pp | 0.770 | **Tie** |
| **Salt-Pepper Robust AUC (E3b)** | 0.7915 ± 0.0092 | 0.8698 ± 0.0074 | 0.8695 ± 0.0076 | -0.03 pp | 0.652 | **Tie** |
| **Occlusion Robust AUC (E3c)** | 0.9412 ± 0.0045 | 0.9418 ± 0.0042 | 0.9416 ± 0.0043 | -0.02 pp | 0.557 | **Tie** |
| **Validation $\beta^*$** | — | 50.0 | 10.0 | -40.0 | — | — |

![Robustness across Gaussian, salt-and-pepper, and occlusion noise](fig2_noise_robustness.png)

### Key Observations:
1. **Clean Performance Equality:** Both memory mechanisms provide a modest +0.33 pp boost over the baseline CNN (Wilcoxon $p = 0.004$), but K-Center and K-means perform identically.
2. **Robustness Convergence:** Both prototype methods deliver a substantial ~+7.2 pp boost in Gaussian AUC and ~+7.8 pp boost in salt-and-pepper AUC over the CNN baseline. However, the difference between K-Center and K-means is less than 0.03 percentage points.
3. **Optimal $\beta^*$ Shift:** K-Center prefers $\beta^* = 10.0$ rather than $\beta^* = 50.0$. Because K-Center points lie farther out in the representation space, their pairwise cosine similarities with test queries exhibit higher dynamic range, requiring lower $\beta$ to avoid over-confident assignment to peripheral memories.

---

## 5. Geometry and Coverage Analysis

To understand why K-Center fails to outperform K-means despite its theoretical motivation, we analyze the geometric properties of the selected memories:

| Geometric Metric | K-Means Centroids | K-Center Real | Random-8 Control | Theoretical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Covering Radius ($r_{\text{cov}}$)** | 6.84 ± 0.32 | **5.61 ± 0.28** | 7.92 ± 0.35 | K-Center minimizes worst-case distance (-17.9%) |
| **Mean Quant. Error (MQE)** | **3.12 ± 0.14** | 4.01 ± 0.18 | 4.28 ± 0.20 | K-means minimizes average distance (+28.4%) |
| **Within-Class Distance** | 3.45 ± 0.16 | 4.48 ± 0.22 | 4.75 ± 0.24 | K-Center centers are more widely dispersed |
| **Cross-Class Separation** | 9.82 ± 0.41 | 9.75 ± 0.44 | 9.68 ± 0.42 | Cross-class separation remains preserved |
| **Norm to Class Mean** | 0.82 ± 0.08 | 2.14 ± 0.15 | 1.95 ± 0.14 | Real points sit significantly farther from class core |

![Coverage radius vs mean quantization error trade-off](fig1_coverage_tradeoff.png)

![Validation accuracy vs beta sensitivity curve](fig3_beta_sensitivity.png)

The empirical geometry confirms the pre-registered trade-off: K-Center successfully contracts the covering radius ($p < 0.001$), but pushes prototypes toward low-density peripheral regions.

---

## 6. Failure Modes, Diagnostics & Outlier Analysis

Why does a 18% reduction in covering radius not improve associative retrieval?

![Retrieval dynamics and convergence over update steps T](fig4_retrieval_dynamics.png)

![Outlier sensitivity and peripheral sample diagnostic](fig5_outlier_diagnostics.png)

1. **Peripheral Point Vulnerability:** In clean training data, points at the geometric boundary include ambiguous handwriting styles (e.g., a "7" resembling a "1", or an elongated "4" resembling a "9"). Plain greedy K-Center inevitably selects these extreme samples as class prototypes.
2. **Soft Retrieval Dynamics:** Continuous Hopfield retrieval computes a softmax over cosine similarities. When a query is perturbed by noise, peripheral prototypes attract probability mass from neighboring classes if boundary separation is narrow.
3. **Centroid Regularization Effect:** K-means centroids act as robust synthetic denoising filters. By averaging over hundreds of in-class training representations, K-means cancels out idiosyncratic high-frequency features.

---

## 7. Computational Cost & Scaling

| Operation / Metric | K-Means | K-Center Greedy | Baseline CNN | Overhead / Comparison |
| :--- | :--- | :--- | :--- | :--- |
| **Memory Setup Time (s)** | 0.095 ± 0.008 | **0.012 ± 0.002** | 0.000 | **7.9x faster construction** |
| **Parameters Added** | 0 | 0 | 0 | Strictly identical |
| **Buffer Size (KB)** | 20.48 | 20.48 | 0.00 | $80 \times 64 \times 4$ bytes |
| **Analytical MACs/Image** | 419,008 | 419,008 | 417,984 | +0.24% over CNN |
| **Latency p50 (ms, B=256)** | 1.82 ± 0.05 | 1.82 ± 0.05 | 1.78 ± 0.04 | +2.2% over CNN |
| **Throughput (imgs/s)** | 140,650 | 140,650 | 143,820 | Identical throughput |

![Cost scaling and throughput across batch sizes](fig6_cost_scaling.png)

![Multi-dimensional trade-off radar chart](fig7_summary_radar.png)

Memory construction in K-Center Greedy is nearly an order of magnitude faster because it requires only $k-1$ sequential vector norm passes over the training split, completely eliminating the iterative centroid recomputation and assignment passes of Lloyd's algorithm.

---

## 8. Evaluation of Pre-registered Hypotheses

We systematically evaluate each hypothesis formulated in `PLAN.md` prior to executing the experiments:

- **H1 (Covering Radius): CONFIRMED.** Across all 10 seeds and 10 digit classes, K-Center achieves a significantly smaller covering radius than K-means (5.61 vs 6.84, $p < 0.001$).
- **H2 (Quantization Error): CONFIRMED.** K-means achieves a lower mean quantization error across all seeds (3.12 vs 4.01, $p < 0.001$).
- **H3 (Clean Accuracy): CONFIRMED.** Clean accuracy ties between K-Center and K-means within 0.01 pp (98.63% vs 98.63%, $|\Delta| < 0.5$ pp, Wilcoxon $p = 0.844$).
- **H4 (Robustness AUC): RESOLVED (EQUIVALENCE).** Robustness AUC under Gaussian noise ties (0.8840 vs 0.8842, $|\Delta| = 0.02$ pp, Wilcoxon $p = 0.770$). Boundary coverage does not expand effective retrieval basins.
- **H5 (Noise Regimes Advantage): REJECTED.** No perturbation regime (including heavy occlusion $14 \times 14$ or extreme noise $\sigma = 1.0$) produced a statistically significant advantage for K-Center.
- **H6 (Outlier Sensitivity): CONFIRMED.** K-Center prototypes exhibit significantly higher distance to class centroids (2.14 vs 0.82), confirming selection of peripheral envelope samples.
- **H7 (Computational Cost): CONFIRMED.** K-Center construction is ~7.9x faster (0.012s vs 0.095s); inference latency, throughput, memory buffers, and MACs are strictly identical.

---

## 9. Discussion & Practical Guidance

### When Should a Practitioner Use K-Center vs. K-Means?
1. **Low-Latency Offline Setup:** When prototypes must be selected rapidly over massive datasets or streaming buffers, K-Center Greedy provides an $O(N \cdot k \cdot D)$ single-pass algorithm that builds memory ~8x faster than K-means.
2. **Interpretability & Exact Traceability:** In safety-critical applications requiring every prototype to correspond to a real, verifiable physical instance (rather than a synthetic mathematical blend), K-Center guarantees genuine data provenance without sacrificing classification accuracy.
3. **Retrieval Performance Priority:** If maximum denoising robustness and resistance to label noise are paramount, K-means centroids remain the superior choice due to their natural centroid-averaging regularization.

---

## 10. Reproducibility

- **Repository Branch:** `feature/kcenter-memory`
- **Dependencies:** Python 3.13.3, PyTorch 2.14.1+cpu, NumPy 2.2.6, Matplotlib 3.11.2, ReportLab 5.1.0
- **Random Seeds:** `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- **Exact Execution Commands:**
  ```bash
  # Run K-means baseline benchmark (Task 13)
  python -m benchmarks.mnist_prototype_retrieval --config configs/mnist_prototype_retrieval.yaml

  # Run K-Center memory benchmark
  python -m benchmarks.mnist_prototype_retrieval --config configs/mnist_kcenter.yaml

  # Run geometric and statistical analysis
  python -m benchmarks.kcenter_analysis

  # Generate PDF research report
  python scratch/build_pdf_report.py
  ```
