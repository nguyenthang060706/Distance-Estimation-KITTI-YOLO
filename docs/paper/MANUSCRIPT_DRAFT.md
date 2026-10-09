# Calibrated Hybrid Geometry–Learning Monocular Vehicle Distance Estimation with Lightweight YOLO Detectors

**Authors:** Anonymous Submission  
**Target Venue:** 8th Asia Digital Image Processing Conference (ADIP 2026), Tokyo, Japan (SPIE Proceedings)  

---

## Abstract
Estimating the distance of preceding vehicles using a single monocular camera is a vital task for advanced driver assistance systems (ADAS) and autonomous driving under stringent computational constraints. While deep learning models offer competitive precision, direct depth regression lacks physical explainability and operates without transparent error attribution to optical geometry or 2D detector bounding box localization. Conversely, pure pinhole geometry offers explicit physical grounding but suffers from systematic biases due to vehicle orientation, 3D center-to-surface offset, and boundary clipping. In this work, we propose a calibrated hybrid monocular distance estimation framework integrating perspective geometry with learned residual correction and conformalized quantile regression (CQR). Three pinhole cues (width, height, and ground-plane constraint) are fused via empirical covariance-weighted log-space pooling to establish a physically grounded baseline. A lightweight gradient-boosted residual model compensates for systematic optical and perspective discrepancies, supplemented by a direct ranging fallback for severely truncated bounding boxes. Furthermore, conformal prediction provides prediction intervals targeting nominal coverage under explicit calibration protocols. Evaluated on the held-out Split T of the KITTI benchmark ($N_{\text{gt}} = 3,212$ Car Hard objects across 10 independent driving sequences), our hybrid pipeline with YOLO11s achieves an AbsRel of 0.0463 ($\delta_1 = 0.9967$) on $N_{\text{tp}} = 2,712$ True Positive detections (detector recall 0.8443). Conformalized quantile regression produces an empirical coverage of 0.9639 at the nominal 90% confidence level (exceeding nominal coverage due to cross-drive difficulty shift between calibration and test drives). Across the common support of 2,528 vehicles detected simultaneously by all three models, the three YOLO detectors exhibit comparable distance estimation precision. While the hybrid pipeline achieves comparable numerical precision to direct bounding-box regression, its core value resides in transparent error decomposition, interpretable physical foundations, and a structured fallback pathway via direct regression under geometric invalidation.

---

## 1. Introduction
Monocular vehicle distance estimation is a foundational perception primitive for forward collision warning (FCW), autonomous emergency braking (AEB), and adaptive cruise control (ACC). Standard automotive setups rely heavily on active sensors such as LiDAR and radar; however, passive camera-based distance estimation provides an indispensable, cost-effective modality with rich semantic perception.

Despite extensive research, existing literature presents three prominent gaps:
- **Gap 1 (Unresolved Systematic Geometric Biases):** Classical perspective geometry relies on rigid assumptions (known 3D vehicle dimensions, flat ground plane, frontal orientation). Few works systematically isolate the error contributions of individual geometric cues across operational distance regimes or decouple geometric formulation errors from 2D detector bounding box jitter.
- **Gap 2 (Uncontrolled Multi-Detector Comparison):** Lightweight detectors (e.g., YOLOv5, YOLOv8, YOLO11) are rarely benchmarked under strictly uniform training recipes and evaluated on common support True Positive sets.
- **Gap 3 (Lack of Calibrated Safety Intervals):** Autonomous decision-making requires well-calibrated confidence bounds. While conformal prediction offers distribution-free guarantees under exchangeability, its empirical behavior under real-world cluster-correlated driving drives remains underexplored.

To bridge these gaps, this work establishes three empirically supported contributions:
1. **Drive-Disjoint Evaluation Protocol & Distribution Shift Diagnosis:** We design a drive-disjoint dataset protocol (`splits-v2`) preventing scene overlap between detector training (A), parameter tuning (V/B), conformal calibration (C), and zero-touch testing (T). We provide empirical diagnosis showing that detector performance shifts substantially from seen sequences (Split A, Recall $\approx 95\%$) to held-out sequences (Split B, Recall dropping to $72\%\text{--}74\%$, with Kolmogorov-Smirnov distance $D_{\text{KS}} > 0.53, p < 10^{-4}$ on bounding box dispersion), whereas unseen splits (B and C) maintain stable distributions ($D_{\text{KS}} \le 0.048$).
2. **Decoupled Geometric Error Decomposition on True Positives:** We systematically decouple 2D detector localization jitter from perspective geometry errors across operational distance bands on True Positive detections (conditioning explicitly documented to bound survivorship bias). We isolate vehicle height ($Z_h$) as the primary physical anchor and show that bounding box pixel jitter in mature detectors has near-zero rank correlation with distance errors.
3. **Conformal Coverage Sensitivity under Natural Drive Clusters:** We evaluate Conformalized Quantile Regression (CQR) across 20 drive-disjoint resplits, demonstrating that theoretical marginal coverage guarantees under exchangeability do not transfer stably when the sampling unit is a drive sequence (yielding 85% average coverage across resplits versus 96% on Split T, with per-seed coverage spanning 63.6% to 99.6% driven by specific sequence heterogeneity).

---

## 2. Related Work
- **Monocular Ranging via Geometry & Neural Approximations:** Dist-YOLO (Vajgl et al., 2022) incorporated distance estimation heads directly into YOLOv3. DisNet (Haseeb et al., 2018) utilized multilayer perceptrons operating on bounding box features. DECADE (Shahzad et al., 2024) benchmarked lightweight YOLO variants for mobile ADAS. Formulations such as Anisotropic Geometry Loss (AGL) (Christanto & Miaou, 2026), soft-sensor perspective geometry with optimized YOLOv5 (Ni et al., 2026), MonoLoco (Bertoni et al., 2019), and classical monocular visual ranging foundations (Dagan et al., 2004; Geiger et al., 2012) established geometric baselines. We position our architecture as a disciplined hybrid synthesis of these precedents without claiming priority.
- **Conformal Prediction in Autonomous Perception:** Conformalized Quantile Regression (CQR) (Romano et al., 2019) extends conformal inference to heteroscedastic interval estimation. In autonomous perception, f-Cal (Bhatt et al., 2021) addressed aleatoric uncertainty calibration. Covariance shrinkage methods such as Oracle Approximating Shrinkage (OAS) (Chen et al., 2010) and scalable tree ensembles (Chen & Guestrin, 2016) provide stable regression backbones. Conformal prediction has also been explored for safety bounds in adaptive cruise control (Dewan & Althoff, 2024).

The table below summarizes our methodological positioning against existing monocular ranging and uncertainty literature across seven dimensions:

| Method | Output Modality | Explicit Geometry | UQ Mechanism | Split Protocol | Multi-Detector Evaluation | Target Scope |
|---|---|---|---|---|---|---|
| Dist-YOLO (2022) | Point depth ($\hat{Z}$) | No (Direct head) | None | Frame-based random | Single (YOLOv3) | Multi-class |
| DisNet (2018) | Point distance ($\hat{d}$) | Partial (Bbox MLP) | None | Frame-based random | Single (YOLOv2) | Bbox objects |
| DECADE (2024) | Point depth ($\hat{Z}$) | No (Direct head) | None | Frame-based random | Multi (YOLOv8 variants) | Mobile ADAS |
| AGL (2026) | Point depth ($\hat{Z}$) | Implicit (Loss function) | None | Frame-based random | Single (YOLOv5) | Low-light |
| f-Cal (2021) | Calibrated $\sigma^2$ | No (Neural depth) | Aleatoric recalibration | Sequence-based | Single (DenseDepth) | Perception |
| ACC Conformal (2024) | Distance bounds | Yes (Kinematic) | Conformal prediction | Synthetic / Trajectory | None (Control model) | ACC distance |
| **Ours** | Point $\hat{Z}$ + 90% $[Z_{\text{lo}}, Z_{\text{hi}}]$ | Yes (3-cue OAS fusion) | Log-space CQR | Strict drive-disjoint (A/V/B/C/T) | Multi (YOLO11s/v8s/v5su) | Car Hard (TP) |

---

## 3. Proposed Methodology
### 3.1 Perspective Geometry Cues
For a calibrated pinhole camera with intrinsic focal lengths $(f_x, f_y)$ and principal point $(c_x, c_y)$, we compute three independent depth cues for detected 2D bounding boxes $(x_1, y_1, x_2, y_2)$:
1. **Width Cue ($Z_w$):** $Z_w = f_x \cdot W_{\text{eff}} / w$, where $w = x_2 - x_1$ and $W_{\text{eff}}$ is the effective vehicle width.
2. **Height Cue ($Z_h$):** $Z_h = f_y \cdot H_{\text{obj}} / h$, where $h = y_2 - y_1$ and $H_{\text{obj}}$ is the median vehicle height.
3. **Ground Contact Cue ($Z_g$):** $Z_g = f_y \cdot H_{\text{cam}} / (y_2 - (c_y + \delta))$, where $H_{\text{cam}}$ and $\delta$ represent effective ground-plane parameters.

Cues subject to image boundary clipping ($\epsilon \le 2\text{ px}$) or horizon violations are masked out dynamically.

### 3.2 Log-Space Covariance Shrinkage Fusion
When at least one cue is valid, depths are fused in log-space:
$$\ln Z_d = \sum_{k \in \mathcal{V}} w_k \ln Z_k$$
where weights $\mathbf{w}$ are obtained by solving the minimum-variance portfolio problem $\min_{\mathbf{w}} \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$ subject to $\sum w_k = 1, w_k \ge 0$ on the Oracle Approximating Shrinkage (OAS) covariance matrix $\mathbf{\Sigma}$ (Chen et al., 2010). The analytical unconstrained minimum-variance weights $\mathbf{w} \propto \mathbf{\Sigma}^{-1} \mathbf{1}$ are evaluated first; if any weight is negative, non-negative least squares (NNLS) active-set projection is invoked to clamp weights to non-negative values.

### 3.3 Residual Calibration Model and Fallback Mechanism
The baseline estimation is defined as $Z_{\text{base}} = Z_d$ for pattern combinations with valid cues, and $Z_{\text{base}} = Z_e$ (a direct regression model trained on bounding box features) when all geometric cues are invalidated (Pattern 000). A gradient-boosted tree ensemble (Chen & Guestrin, 2016) models the log-ratio residual:
$$\hat{r} = f(\mathbf{x}) \approx \ln Z_{\text{gt}} - \ln Z_{\text{base}}, \quad \hat{Z}_f = Z_{\text{base}} \exp(\hat{r})$$
operating strictly on 17 test-time observable features (16 base bounding-box and geometric cue features plus derived baseline $\ln Z_{\text{base}}$).

### 3.4 Conformalized Quantile Regression (CQR)
Two quantile regression models estimate the 5th and 95th percentiles of residual error: $\hat{q}_{0.05}(\mathbf{x})$ and $\hat{q}_{0.95}(\mathbf{x})$. On an independent calibration set (Split C, comprising $N_{\text{gt}}=1,826$ Car Hard objects, yielding $n=1,489$ calibration True Positives for YOLO11s), nonconformity scores $s_i = \max(\hat{q}_{0.05}(\mathbf{x}_i) - r_i, r_i - \hat{q}_{0.95}(\mathbf{x}_i))$ are evaluated. The calibration threshold $\hat{Q}$ is selected at order statistic level $k = \lceil (n+1)(1-\alpha) \rceil$. The physical distance confidence interval is constructed as:
$$[Z_{\text{lo}}, Z_{\text{hi}}] = \left[ Z_{\text{base}} \exp(\hat{q}_{0.05} - \hat{Q}), \; Z_{\text{base}} \exp(\hat{q}_{0.95} + \hat{Q}) \right]$$

---

## 4. Experimental Setup
- **Dataset Partitioning:** 7,481 annotated KITTI frames are partitioned into five drive-disjoint splits, with Split A and V established first and Splits B, C, T repartitioned via simulated annealing (seed 85): Split A (3,740 frames, 35 drives) for detector training; Split V (374 frames, 25 drives) for validation; Split B (1,499 frames, 34 drives) for geometry and residual fitting; Split C (766 frames, 18 drives) for conformal calibration; and Split T (1,102 frames, 29 drives, $N_{\text{gt}} = 3,212$ Car Hard) as the zero-touch test benchmark.
- **Detector Training:** YOLOv5su, YOLOv8s, and YOLO11s are trained for 100 epochs on Split A under fixed hyperparameters. Checkpoint selection freezes `last.pt` (epoch 100) to ensure annealing stability. Confidence thresholds are tuned on Split V using moving-average smoothed $F_1$ scores (0.700 for YOLO11s, 0.790 for YOLOv8s, 0.740 for YOLOv5su).
- **Evaluation Criteria:** All ranging metrics are evaluated on True Positive (TP) detections matched to Ground Truth Car Hard boxes under Greedy matching (IoU $\ge 0.5$).

---

## 5. Experimental Results and Discussion

### 5.1 Error Decomposition and Geometric Baseline Analysis (Split B OOF and Split T)
As summarized in Table 2, evaluating individual perspective cues on detector bounding boxes across Split B out-of-fold predictions isolates fundamental optical limitations:
1. **Width Cue ($Z_w$):** Yields the highest error (Pooled AbsRel 0.2462, MAE 6.44 m, $\delta_1 =$ 48.4%), heavily compromised by vehicle orientation and aspect ratio distortion under oblique angles.
2. **Ground Contact Cue ($Z_g$):** Suffers from pitch variations and non-flat road profiles (Pooled AbsRel 0.0931, MAE 2.41 m, $\delta_1 =$ 93.9%).
3. **Height Cue ($Z_h$):** Represents the most reliable single cue (Pooled AbsRel 0.0650, MAE 1.47 m, $\delta_1 =$ 97.8%), benefitting from relatively stable 3D vehicle heights across passenger car classes.

Optimal covariance shrinkage fusion ($Z_d$) effectively combines $Z_h$ and $Z_g$, reducing the pooled AbsRel to 0.0607 (MAE 1.34 m). Ablation experiments on Split B further confirm that dropping $Z_h$ incurs a measurable degradation ($\Delta_{\text{pooled}} = +0.0070$, 95% bootstrap CI [0.0024, 0.0119], strictly excluding zero), isolating height as the dominant physical anchor. Dropping the bounding box geometry feature group yields a negligible shift ($\Delta_{\text{pooled}} = +0.0013$, 95% bootstrap CI [-0.0041, 0.0059], spanning zero), while removing validity flags produces a marginal shift ($\Delta_{\text{pooled}} = +0.0003$, CI [-0.0000, 0.0007]).

Transferring this grounded geometric base to the locked held-out Split T (Table 3), the fused geometric baseline $Z_d$ achieves a pooled AbsRel of 0.0640 on YOLO11s. Incorporating learned residual correction $\hat{Z}_f$ reduces the pooled AbsRel to 0.0463 ($\delta_1 = 0.9967$, $\text{MAE} = 1.10 m$). Paired cluster bootstrap tests over 10 driving sequences show an estimated improvement over geometric fusion (d) with a difference of -0.0188 (95% CI [-0.0264, -0.0107], excluding zero; exploratory given 10 clusters). Note that the difference between the pooled AbsRel of (d) (0.0640) and (f) (0.0463) is 0.0177, whereas the paired cluster bootstrap difference is -0.0188; this minor divergence arises because the pooled AbsRel of (d) excludes fallback cases (where geometry is invalid), whereas the paired bootstrap evaluates $(Z_f - Z_d)$ strictly on the common valid subset.

Crucially, comparison between the hybrid residual model (f) and the direct bounding-box regression model (e) reveals an estimated difference of -0.0002 with a 95% bootstrap confidence interval of [-0.0012, 0.0023], which spans zero at the 10-cluster level. **Hence, there is no empirical evidence of numerical precision divergence between learned residual modeling and direct depth regression on Split T.** We explicitly highlight that the merit of the hybrid framework lies not in numerical superiority, but in: (1) providing an interpretable physics-based anchor ($Z_d$), (2) enabling principled error decomposition between optical geometry and detector bounding box jitter, and (3) offering a structured direct regression fallback when perspective cues are clipped at frame boundaries (though empirical coverage on these boundary cases drops to 77.8%–86.1%).

**Mechanistic Explanation and Boundary Behavior (Evaluation on Split B OOF):**
Why do Model (f) and Model (e) achieve nearly identical numerical precision on Split T? Mechanistically, direct regression Model (e) receives bounding box dimensions ($h$ and $y_{\text{bottom}} - c_y$). Because camera intrinsics ($f_y$) and vehicle physical height ($H_{\text{obj}}$) are fixed priors, gradient-boosted decision trees with sufficient splits on $h$ piecewise-linearly approximate the inverse perspective function $\ln (f_y H_{\text{obj}} / h) = \text{const} - \ln h$. Controlled experiments on Split B OOF validate this explanation:
1. *Data Efficiency (Learning Curve across Drives):* When training data is scarce ($k = 2$ drives), direct regression exhibits high error (AbsRel 0.1803), whereas the hybrid residual model achieves AbsRel 0.0575—a 68% relative error reduction anchored by fused geometry ($Z_d = 0.0607$). As training data scales to $k = 11$ drives, direct regression learns the inverse perspective mapping and converges to $(f) \approx (e)$ (0.0476 vs 0.0467).
2. *Range Extrapolation ($Z > 30$ m):* When trained only on near/mid distances ($Z \le 30$ m) and evaluated on distant vehicles ($Z > 30$ m, $N = 929$), direct tree regression degrades markedly (AbsRel 0.2235, MAE 9.23 m, $\delta_1 = 44.8\%$) because axis-aligned decision trees cannot extrapolate beyond observed feature thresholds. Conversely, the hybrid model maintains bounded physical errors (AbsRel 0.0760, MAE 2.94 m, $\delta_1 = 98.2\%$) by anchoring to pinhole perspective geometry.

### 5.2 RQ2: Cross-Detector Comparison on Common Support
On the common support set of 2,528 vehicles detected simultaneously by all three models, ranging performances are practically indistinguishable: Model (f) AbsRel reaches 0.0446 for YOLO11s, 0.0449 for YOLOv8s, and 0.0457 for YOLOv5su. Pairwise cluster bootstrap differences between detectors all contain zero. Furthermore, correlation analysis (Table 4) shows that Spearman rank correlations between bounding box IoU / bottom edge jitter and ranging error remain largely bounded near zero ($\rho \in [-0.087, +0.104]$ with $|\rho| \le 0.11$, and 95% CIs containing zero across 23 of 24 configurations; the only exception being detector confidence on YOLO11s with $\rho = -0.0867$, 95% CI [-0.1741, -0.0144]), indicating that within successfully localized True Positive detections, local 2D pixel jitter exhibits negligible monotonic correlation with relative distance estimation errors.

### 5.3 RQ3: Uncertainty Quantification and Conformal Coverage
Standard CQR achieves an empirical pooled coverage of 0.9639 for YOLO11s (0.9714 for YOLOv8s, 0.9660 for YOLOv5su), exceeding the nominal 90.0% confidence level. We emphasize that this conservative over-coverage is an empirical post-hoc discovery. Statistical diagnostics indicate a domain difficulty shift between calibration Split C and test Split T: Split C exhibits higher residual dispersion (Mean $|r| \approx 0.083$ vs $0.067$ on T, driven by two challenging clustered sequences comprising 39.3% of Split C). Consequently, calibration on Split C inflates $\hat{Q}$, conferring conservative coverage on Split T. Across all evaluations, zero interval crossing violations ($r_{\text{lo}} > r_{\text{hi}}$) were observed.

**Tripartite Coverage Comparison and Partition Sensitivity:** To rigorously contextualize conformal validity, we juxtapose three empirical coverage figures across different calibration-evaluation partitions:
1. *Development Leave-One-Drive-Out on Split C (C-LODO):* Pooled coverage achieves 87.1%–88.0% (and 90.0%–90.1% macro coverage on clusters with $n \ge 30$).
2. *20 Drive-Disjoint Resplits on $B \cup C$:* Across 20 random partitions (seed 0–19) of the 22 development drives (ratio $\approx$ 50% fit / 25% calib / 25% eval), empirical pooled coverage averages 85.15% ($\pm 8.62%$, range [69.80%, 98.75%]) for YOLO11s, 86.14% ($\pm 8.67%$, range [69.59%, 99.56%]) for YOLOv8s, and 84.99% ($\pm 8.37%$, range [68.97%, 97.91%]) for YOLOv5su. Macro coverage across drives drops further to 77.9%–84.5% (with per-seed values spanning 63.6% to 99.6%). Only 35%–40% of partition seeds attain empirical coverage $\ge 90\%$.
3. *Held-Out Test Split T:* Achieves 96.4%–97.1% empirical coverage.

This juxtaposition demonstrates that theoretical marginal coverage guarantees established under exchangeability do not transfer stably when the operational sampling unit is a drive sequence rather than an independent object. The direction of divergence is fundamentally governed by which specific drives populate the calibration versus evaluation folds. Resplits are inherently conservative/pessimistic because calibration sets receive only 5–6 drives ($n_{\text{calib}} \approx 1,100\text{--}1,800$), whereas Split C provides 10 drives ($n_{\text{calib}} = 1,489$).

**Drive-Level Cluster Heterogeneity (`0057` and `0004`):** Stratifying the 20 resplit evaluations on YOLO11s by the assignment of two high-dispersion drives reveals a distinct three-tier partition effect:
- When both drives `0057` and `0004` fall into the evaluation fold (5 seeds), empirical CQR coverage drops sharply to an average of **72.38%** (range [69.80%, 75.25%]);
- When exactly one of the two drives is present in evaluation (11 seeds), coverage averages **87.58%** (range [80.11%, 93.08%]);
- When neither drive is present in evaluation (4 seeds), coverage averages **94.44%** (range [89.93%, 98.75%]).

This demonstrates that drive-level heterogeneity is the primary driver of out-of-sample coverage fluctuation, directly explaining why alternate partitions exhibit under-coverage while Split T experiences conservative over-coverage. We do not rank Split Conformal, Standard CQR, and Mondrian CQR because their performance differences ($\le 3$ percentage points) are minor compared to cross-partition standard deviation ($\sigma \approx 7.4\%\text{--}9.0\%$).

### 5.4 RQ4: Hardware Latency Benchmark and Deployment Viability (Tier 1 Benchmark)
As detailed in Table 7 (designated with the preliminary label `PRELIMINARY-v2`), end-to-end inference benchmarked over 200 in-memory frames of Split B indicates real-time feasibility on workstation laptop hardware. On an NVIDIA RTX 5060 Laptop GPU (FP16 CUDA), total pipeline latency achieves a median of 37.99 ms (P95 48.49 ms, 26.3 FPS) for YOLO11s, 32.05 ms (31.2 FPS) for YOLOv8s, and 33.01 ms (30.3 FPS) for YOLOv5su. On multi-core CPU execution (ONNX Runtime FP32, 4 threads), latencies scale to 116.17 ms (8.6 FPS) for YOLO11s, 143.37 ms (7.0 FPS) for YOLOv8s, and 120.48 ms (8.3 FPS) for YOLOv5su.

Crucially, the post-detector processing stages—comprising geometric cue extraction (0.18 ms), residual XGBoost inference (0.57–0.60 ms), and conformal quantile interval bounding (0.58–0.60 ms)—incur a combined overhead of only $\approx 1.35$ ms per image, accounting for less than $4.2\%$ of total GPU execution time. PyTorch vs. ONNX parity diagnostics confirm that Count Parity ratios remain well within the acceptable $[0.95, 1.05]$ tolerance ($0.974\text{--}1.009$). We hypothesize that the lower IoU match rate ($91.0\%\text{--}94.4\%$) arises from static $640 \times 640$ square padding versus dynamic $640 \times 224$ letterboxing.

The empirical findings are comprehensively summarized across five publication figures: Fig. 1 illustrates the modular hybrid architecture; Fig. 2 presents the drive-clustered split distributions; Fig. 3 depicts error attenuation across distance ranges; Fig. 4 visualizes conformal coverage and interval width trade-offs; and Fig. 5 provides representative qualitative case studies with the required evaluation disclaimers.

---

## 6. Limitations and Threats to Validity
We explicitly document 14 methodological and practical limitations governing our study, incorporating the six core boundary conditions required by §11:

1. **Truck Class Distribution Imbalance & Single-Dataset Domain Constraint (KITTI Only):** Trucks remain concentrated in specific large drives (4.8% of vehicles / 3.8% of objects in Split A vs 1.3% of vehicles / 1.2% of objects in Split T). Drive-level partitioning cannot fully reconcile heavy vehicle balance across 141 natural driving sequences. Furthermore, the benchmark is conducted solely on the single-dataset KITTI benchmark (fair weather, daytime); cross-dataset transfer remains unvalidated.
2. **Survivorship Bias on True Positives and Target Matching Scope (Car Hard Only):** All ranging evaluation metrics are conditioned on successfully detected True Positive bounding boxes (detector recall 0.8443 for YOLO11s, 0.8281 for YOLOv8s, 0.8325 for YOLOv5su) matched greedily at IoU $\ge 0.5$. Because our evaluation isolates ranging error conditional on true positive localization, detection mAP@0.7 is not evaluated; open-world operational risk under false negative misses (where miss rates reach 26.2%–29.3% in the distant 30–50 m regime and 72.7%–93.9% beyond 50 m on Split T) exceeds reported conditional ranging errors. Evaluation is strictly confined to the Car Hard category.
3. **Sparse Distant Sample Support (>50 m):** Ground Truth Car Hard instances beyond 50 m comprise only 19 objects in B, 0 in C, and 33 in T. True Positives on T drop to $\le 9$ instances, precluding statistically robust empirical anchoring in ultra-long-range regimes.
4. **Drive Concentration and Cluster Correlation:** Vehicle occurrences are heavily clustered: Split V contains only 6 car drives (top drive = 41%), Split B contains 12 car drives, Split C has 10, and Split T has 10. With $k \le 12$ clusters (below the recommended threshold of 20), bootstrap confidence intervals must be interpreted as coarse cluster bounds.
5. **Detector Training Checkpoint Reproducibility and Single-Seed Training:** Training logs record `git_dirty: true` due to untracked runtime artifacts during background execution. Model integrity is independently secured via exact SHA-256 checkpoint verification. All 2D YOLO detectors were optimized under a single fixed random seed (seed 42), precluding multi-seed detector variance estimation.
6. **KITTI Neighbor Class Matching Protocols and Restricted Training Data Scale (Split A ~50% Data):** Unlike official KITTI benchmarks, Van and Truck are not treated as neighbor ignore classes. Occasional false positive assignments slightly depress reported Precision without impacting ranging on verified Car TP instances. Furthermore, detector training is constrained to Split A (3,740 frames, 35 drives; $\approx 50.0\%$ data) to preserve disciplined drive disjointness for B, C, and T.
7. **Indirect Qualitative Design Leakage and Flat Ground-Plane Assumption:** Ten driving sequences repartitioned during preliminary dataset balancing informed exploratory image boundary masking heuristics before protocol freezing. Additionally, the ground contact cue $Z_g$ relies on a planar road geometry assumption ($H_{\text{cam}}, \delta$) that is vulnerable to road slopes and vehicle pitch dynamics.
8. **Empirical Indistinguishability of Residual and Direct Regression & Vehicle Viewing Angle:** Residual Model (f) and Direct Model (e) exhibit overlapping confidence bounds on Split T. The hybrid methodology is justified by physical interpretability and structured fallback pathways rather than numerical accuracy superiority. Moreover, fixed physical priors assume frontal/rear orientation; oblique vehicle viewing angle $\theta$ distorts 2D projections relative to the closest vehicle surface (bumper).
9. **Post-Hoc Verification Transparency:** The initial Split T run exhibited inflated errors due to an XGBoost `base_score` serialization string parsing anomaly in v1. We transparently disclose that the primary author observed the initial unstandardized run metrics (AbsRel ~0.62) before diagnosing and resolving this bug post-hoc via JSON scalar standardization in v1.1 without model retraining, hyperparameter alteration, or violation of the locked held-out test split protocol.
10. **Exchangeability Shift and Conservative Over-Coverage:** Differences in drive composition between calibration Split C and Split T violate strict exchangeability, converting expected marginal coverage into conservative over-coverage (0.9639).
11. **Optimism of 10-Cluster Bootstrap CIs:** Bootstrap intervals conditioned on fixed calibration threshold $\hat{Q}$ reflect intra-split variation across 10 drives and underestimate total partition sensitivity ($\sigma \approx 7\text{--}9\%$).
12. **Masked Local Under-Coverage:** Pooled 96%–97% coverage conceals local vulnerabilities: truncated vehicles (88.0%–91.5%) and edge fallback cases (77.8%–86.1%) remain below nominal 90% coverage.
13. **Mondrian Interval Width Inflation and Distance Conditioning:** Mondrian CQR restores coverage in the near 0–10 m distance band (94.1% vs 91.8% for Standard CQR), with interval width ratio in that band averaging 1.43–1.47× (compared to pooled average width of 1.29–1.32×).
14. **Hardware and Latency Benchmark Constraints:** GPU latency benchmarking utilizes static $640 \times 640$ square padding rather than dynamic rectangular letterboxing ($640 \times 224$), increasing pixel processing volume by $\approx 2.9\times$. All workstation latency numbers carry the `PRELIMINARY-v2` designation.

---

## References
1. **Ni, Z., Shi, J., Li, L., & Ni, C.** (2026). Real-time Vehicle Detection and Distance Estimation: Soft-sensor Approach Using Optimized YOLOv5 and Perspective Geometry. *Sensors and Materials*, 38(1).
2. **Vajgl, M., Hurtik, P., & Nejezchleba, T.** (2022). Dist-YOLO: Fast Object Detection with Distance Estimation. *Applied Sciences*, 12(3), 1354.
3. **Haseeb, M. A., Guan, J., Ristić-Durrant, D., & Gräser, A.** (2018). DisNet: A Novel Method for Distance Estimation from Monocular Camera. In *Workshop on Planning, Perception and Navigation for Intelligent Vehicles (PPNIV), IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*.
4. **Shahzad, M. Z., Hanif, M. A., & Shafique, M.** (2024). DECADE: Towards Designing Efficient-yet-Accurate Distance Estimation Modules for Collision Avoidance in Mobile ADAS. *arXiv preprint arXiv:2410.19336*.
5. **Christanto, R., & Miaou, S.-G.** (2026). Lightweight Monocular Distance Estimation via Anisotropic Geometry Loss for Low-Light Driving Environments. *Sensors*, 26(1).
6. **Bertoni, L., Kreiss, S., & Alahi, A.** (2019). MonoLoco: Monocular 3D Pedestrian Localization and Uncertainty Estimation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 6861–6871.
7. **Romano, Y., Patterson, E., & Candès, E.** (2019). Conformalized Quantile Regression. In *Advances in Neural Information Processing Systems (NeurIPS)*, Vol. 32, pp. 3543–3553.
8. **Bhatt, D., Mani, K., Bansal, D., Murthy, K., Lee, H., & Paull, L.** (2021). f-Cal: Calibrated Aleatoric Uncertainty Estimation from Neural Networks for Robot Perception. *arXiv preprint arXiv:2109.13913*.
9. **Dagan, E., Mano, O., Stein, G. P., & Shashua, A.** (2004). Forward Collision Warning with a Single Camera. In *IEEE Intelligent Vehicles Symposium*, pp. 37–42.
10. **Geiger, A., Lenz, P., & Urtasun, R.** (2012). Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 3354–3361.
11. **Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.** (2010). Shrinkage Algorithms for MMSE Covariance Estimation. *IEEE Transactions on Signal Processing*, 58(10), 5016–5029.
12. **Chen, T., & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794.
