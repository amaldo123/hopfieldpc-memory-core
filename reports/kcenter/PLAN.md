# Stage 0: K-Center Memory Study and Experiment Plan

## 1. Mathematical Formulation and 2-Approximation

### 1.1 Objective and Comparison with K-Means
Given a dataset of representations for a class $X = \{x_1, \dots, x_N\} \subset \mathbb{R}^D$ and a memory budget $k \ge 1$, the **K-Center** objective (Sener & Savarese 2018, Eq. 5) chooses a subset $S \subseteq X$ with $|S| = k$ to minimize the *covering radius* (the worst-case Euclidean distance from any point in $X$ to its closest chosen pattern):

$$\min_{S \subseteq X, |S| = k} \max_{x_i \in X} \min_{s_j \in S} \|x_i - s_j\|_2$$

In contrast, **K-means** minimizes the *average squared Euclidean distance* (quantization error):

$$\min_{\mu_1, \dots, \mu_k \in \mathbb{R}^D} \frac{1}{N} \sum_{i=1}^N \min_{j \in \{1,\dots,k\}} \|x_i - \mu_j\|_2^2$$

This fundamental difference dictates their behavior:
- **K-means centroids** lie in dense probability mass regions and average out tail variations. The resulting centers are synthetic convex combinations rather than real observed data points.
- **K-Center Greedy** explicitly traverses the geometry toward the tails, selecting diverse *real* points that cover the outermost envelope of the representation space. However, this worst-case minimax objective makes plain greedy K-Center vulnerable to atypical samples, mislabeled digits, and tail outliers.

### 1.2 The Greedy Algorithm
Starting from an initial point $u_1$ (chosen via a seeded pseudo-random draw), the algorithm incrementally selects $u_t$ for $t = 2, \dots, k$ by choosing the point farthest from the currently selected set:

$$u_t = \arg\max_{x_i \in X} \min_{s \in S_{t-1}} \|x_i - s\|_2$$

By maintaining a running distance vector $\text{nearest}[i] = \min_{s \in S} \|x_i - s\|_2$, each new selection requires only computing distances from all $N$ points to the newly added point $x_{u_t}$ and taking an element-wise minimum:

$$\text{nearest}[i] \leftarrow \min\left(\text{nearest}[i],\, \|x_i - x_{u_t}\|_2\right)$$

This achieves $O(N \cdot k \cdot D)$ time and $O(N)$ auxiliary memory without ever instantiating an $N \times N$ pairwise distance matrix.

### 1.3 2-Approximation Proof (in our own words)
Let $S^* = \{s_1^*, \dots, s_k^*\}$ be an optimal choice of $k$ centers, and let $r^* = \max_{x \in X} \min_{s^* \in S^*} \|x - s^*\|_2$ be the optimal covering radius. The optimal centers define $k$ balls of radius $r^*$ whose union covers all of $X$.

Now consider the sequence of $k$ points selected by the greedy algorithm $S = \{u_1, \dots, u_k\}$, plus the candidate point $u_{k+1}$ that would be selected on step $k+1$.
Let $r_{\text{greedy}} = \min_{s \in S} \|u_{k+1} - s\|_2$. By definition of the greedy selection rule, at every step $t \le k$, the chosen point $u_t$ maximized the distance to all prior picks, which was at least the distance to the final candidate $u_{k+1}$. Therefore, all pairwise distances among the $k+1$ points $\{u_1, \dots, u_{k+1}\}$ satisfy:

$$\|u_a - u_b\|_2 \ge r_{\text{greedy}} \quad \forall 1 \le a < b \le k+1$$

Since there are $k+1$ points and only $k$ optimal balls of radius $r^*$, by the **Pigeonhole Principle**, at least two points $u_a$ and $u_b$ ($a \ne b$) must belong to the same optimal ball centered at some $s_m^* \in S^*$.

Applying the triangle inequality:

$$r_{\text{greedy}} \le \|u_a - u_b\|_2 \le \|u_a - s_m^*\|_2 + \|s_m^* - u_b\|_2 \le r^* + r^* = 2 r^*$$

Thus, $r_{\text{greedy}} \le 2 \cdot \text{OPT}$, establishing that plain greedy farthest-point traversal is a guaranteed 2-approximation to the NP-hard $k$-center problem.

### 1.4 Connection to Active Learning (Theorem 1)
Theorem 1 of Sener & Savarese (2018) bounds the core-set generalization error by the covering radius scaled by the Lipschitz constant of the loss function, plus an empirical training loss term. This theoretical result motivates covering radius as a natural geometric descriptor for representation coverage. However, the theorem assumes zero empirical loss on the core-set under supervised gradient updates; in our setting, representations are frozen and retrieved associatively via a modern Hopfield energy landscape, making empirical testing essential.

---

## 2. Implementation Plan

- **Files to Modify**:
  - `benchmarks/mnist_prototype_retrieval.py`:
    - Add `k_center_greedy(x, k, *, generator, first=None) -> torch.Tensor` returning a `LongTensor` of $k$ picked row indices in selection order.
    - Add `covering_radius(x, centers) -> float`.
    - Extend `build_memory(..., method="kmeans")`: support `method="kcenter"` (selecting real rows `zc[idx]`), validate method, keep `per_class == 1` as class mean.
    - Extend `build_embedding_net(..., method="kmeans")` to thread `method` through.
    - Update `run_experiment`: read `cfg.get("memory_method", "kmeans")` once and pass it to both `build_embedding_net` calls.
    - Update `build_report`: append `, {method}` to `memory_desc` only when `method != "kmeans"`.
  - `benchmarks/mnist_baseline.py`:
    - Add `"memory_method"` to `CONFIG_KEYS`.
    - Define `MEMORY_METHODS = ("kmeans", "kcenter")`.
    - Add validation in `validate_mnist_config`.
- **Files to Create**:
  - `configs/mnist_kcenter.yaml`: exact clone of `configs/mnist_prototype_retrieval.yaml` with two changes: `memory_method: kcenter` and `reports_dir: "reports/mnist_kcenter"`.
  - `tests/test_kcenter.py`: comprehensive tests covering the greedy algorithm, properties, 2-approximation, memory construction, config validation, and parametrized end-to-end quick run.
  - `benchmarks/kcenter_analysis.py`: standalone analysis script covering coverage geometry, paired statistics, retrieval dynamics, outlier analysis, computational scaling, and figure generation.
- **Config Key**:
  - `memory_method: "kmeans" | "kcenter"` (default `"kmeans"`).

---

## 3. Experimental Protocol

Fairness and rigor are paramount:
- **Seeds**: Exactly 10 identical seeds (`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`).
- **CNN Backbones**: Frozen trained weights per seed (`backbone_seed{s}.pt`) trained once on clean MNIST. Baseline L8 = Identity numbers must match bit-for-bit across runs.
- **Centering**: Both methods center stored patterns by the train-mean embedding $\mu$ ($z - \mu$), preserving Ramsauer et al. (2020) geometry.
- **Memory Budget**: 8 real embeddings per class (80 total patterns), matching the 8 k-means centroids per class.
- **Metric**: Euclidean ($L_2$) distance on raw 64-D L7 embeddings.
- **Initialization**: Seeded pseudo-random initial point per class using `torch.Generator().manual_seed(seed)`.
- **$\beta^*$ Selection**: Re-selected independently per method on identical validation sets (`["clean", 0]`, `["gaussian", 0.4]`, `["salt_pepper", 0.15]`).
- **Noise Test Grids**: Identical test grids (Gaussian $\sigma \in [0.0, 1.0]$, salt-and-pepper $p \in [0.0, 0.4]$, occlusion patch size $\in [0, 14]$).

---

## 4. Hypotheses Formulated Before Running Experiments

- **H1 (Covering Radius)**: K-Center will achieve a strictly smaller covering radius on training embeddings than K-means across all classes and seeds.
- **H2 (Quantization Error)**: K-means will achieve a lower mean quantization error (average distance to nearest center) than K-Center.
- **H3 (Clean Accuracy)**: Clean classification accuracy will tie between K-Center and K-means ($|\Delta| < 0.5$ percentage points). Clean MNIST is near saturation (~98.9%), and class clusters are well-separated.
- **H4 (Robustness AUC)**: Robustness AUC under noise will differ between methods (open question: does boundary coverage expand the basin of attraction for noisy queries, or does it degrade margin near decision boundaries?).
- **H5 (Noise Regimes)**: Any performance advantage for K-Center will be concentrated in heavy perturbation regimes (large Gaussian noise or occlusion), where query representations drift far from class centers into the tails.
- **H6 (Outlier Sensitivity)**: K-Center will select more peripheral points (higher Mahalanobis / standard deviation distance from class mean) and potentially misclassified points compared to a random sample or K-means.
- **H7 (Computational Cost)**: Memory construction will be faster for K-Center ($O(N \cdot k \cdot D)$ single pass vs 25 iterations of Lloyd's algorithm). Inference cost (MACs, latency, parameter count) will be strictly identical because both store an $80 \times 64$ tensor.

### Primary Endpoints
1. **Clean head accuracy** (`head_acc`): K-Center vs K-means.
2. **Gaussian-noise robustness AUC** (`robust_auc`): K-Center vs K-means.

*All other evaluations (salt-and-pepper, occlusion, prototype readout accuracy, attention entropy, dead memories) are explicitly designated as exploratory.*

---

## 5. Null Result Definition
A result is considered a **null / tie** if:
$$|\Delta| < 0.5\text{ percentage points } (0.005) \quad \text{OR} \quad \text{Wilcoxon signed-rank test across 10 seeds } p \ge 0.05$$
Negative or null findings will be reported transparently and analyzed without altering protocols, configs, or hypotheses post hoc.
