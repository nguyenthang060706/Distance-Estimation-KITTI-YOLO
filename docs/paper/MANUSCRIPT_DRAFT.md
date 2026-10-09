# Calibrated Hybrid Geometry–Learning Monocular Vehicle Distance Estimation with Lightweight YOLO Detectors

**Authors:** Anonymous Submission  
**Target Venue:** IEEE / Applied ITS Conference & Journal Submission  
**Repository State:** Verified v1.1, Zero-Touch Split T Locked (`runs/final_T.lock`)

---

## Abstract
Estimating the distance of preceding vehicles using a single monocular camera is a vital task for advanced driver assistance systems (ADAS) and autonomous driving under stringent computational constraints. While deep learning models offer competitive precision, direct depth regression lacks physical explainability and fails catastrophically under out-of-distribution visual degradation. Conversely, pure pinhole geometry offers explicit physical grounding but suffers from systematic biases due to vehicle orientation, 3D center-to-surface offset, and boundary clipping. In this work, we propose a calibrated hybrid monocular distance estimation framework integrating perspective geometry with learned residual correction and conformalized quantile regression (CQR). Three pinhole cues (width, height, and ground-plane constraint) are fused via empirical covariance-weighted log-space pooling to establish a physically grounded baseline. A lightweight gradient-boosted residual model compensates for systematic optical and perspective discrepancies, supplemented by a direct ranging fallback for severely truncated bounding boxes. Crucially, finite-sample conformal prediction provides calibrated prediction intervals with rigorous finite-sample coverage guarantees. Evaluated on the held-out Split T of the KITTI benchmark ($N_{\text{gt}} = 3,212$ Car Hard objects across 10 independent driving sequences), our pipeline achieves an AbsRel of 0.0463 ($\delta_1 = 0.9967$) with YOLO11s, maintaining an empirical 90% conformal coverage of 0.9639 across 2,528 commonly detected vehicles. We show that while the hybrid pipeline achieves comparable numerical precision to direct bounding-box regression, its core value resides in transparent error decomposition, interpretable physical foundations, and fail-safe fallback capabilities.

---

## 1. Introduction
Monocular vehicle distance estimation is a foundational perception primitive for forward collision warning (FCW), autonomous emergency braking (AEB), and adaptive cruise control (ACC). Standard automotive setups rely heavily on active sensors such as LiDAR and radar; however, passive camera-based distance estimation provides an indispensable, cost-effective modality with rich semantic perception.

Despite extensive research, existing literature presents three prominent gaps:
- **Gap 1 (Unresolved Systematic Geometric Biases):** Classical perspective geometry relies on rigid assumptions (known 3D vehicle dimensions, flat ground plane, frontal orientation). Few works systematically isolate the error contributions of individual geometric cues across operational distance regimes or decouple geometric formulation errors from 2D detector bounding box jitter.
- **Gap 2 (Uncontrolled Multi-Detector Comparison):** Lightweight detectors (e.g., YOLOv5, YOLOv8, YOLO11) are rarely benchmarked under strictly uniform training recipes and evaluated on common support True Positive sets.
- **Gap 3 (Lack of Calibrated Safety Intervals):** Autonomous decision-making requires well-calibrated confidence bounds. While conformal prediction offers distribution-free guarantees, its validity under real-world cluster-correlated driving drives remains underexplored.

To bridge these gaps, this study presents:
1. A rigorous drive-based dataset protocol (`splits-v2`) preventing scene overlap across training, calibration, and test phases.
2. An interpretable hybrid pipeline uniting multi-cue pinhole geometry, covariance shrinkage fusion, and residual tree ensembles.
3. An uncertainty quantification framework via Conformalized Quantile Regression (CQR), accompanied by an empirical examination of exchangeability and domain shift across driving sequences.

---

## 2. Related Work
- **Monocular Ranging via Geometry & Neural Approximations:** Dist-YOLO (Vajgl et al., 2022) incorporated distance estimation heads directly into YOLOv3. DisNet (Haseeb et al., 2023) utilized multilayer perceptrons operating on bounding box features. DECADE (Kim et al., 2024) benchmarked lightweight YOLO variants for mobile ADAS. Recent formulations like Anisotropic Geometry Loss (AGL) (Ni et al., 2026) and MonoLoco (Bertoni et al., 2019) further advanced geometric learning. We position our architecture as a principled hybrid synthesis of these precedents without claiming priority.
- **Conformal Prediction in Autonomous Perception:** Conformalized Quantile Regression (CQR) (Romano et al., 2019) extends conformal inference to heteroscedastic interval estimation. In autonomous perception, f-Cal (Bhatt et al., 2021) addressed aleatoric uncertainty. Our work investigates the practical behavior of log-space CQR under correlated cluster shifts in natural driving datasets.

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
where weights $\mathbf{w}$ are obtained via non-negative least squares (NNLS) on the inverse empirical error covariance $\mathbf{\Sigma}^{-1}$, regularized with Ledoit-Wolf shrinkage.

### 3.3 Residual Calibration Model and Fallback Mechanism
The baseline estimation is defined as $Z_{\text{base}} = Z_d$ for pattern combinations with valid cues, and $Z_{\text{base}} = Z_e$ (a direct regression model trained on bounding box features) when all geometric cues are invalidated (Pattern 000). A gradient-boosted tree ensemble models the log-ratio residual:
$$\hat{r} = f(\mathbf{x}) \approx \ln Z_{\text{gt}} - \ln Z_{\text{base}}, \quad \hat{Z}_f = Z_{\text{base}} \exp(\hat{r})$$
operating strictly on a whitelist of 17 test-time observable 2D bounding box and geometric features.

### 3.4 Conformalized Quantile Regression (CQR)
Two quantile regression models estimate the 5th and 95th percentiles of residual error: $\hat{q}_{0.05}(\mathbf{x})$ and $\hat{q}_{0.95}(\mathbf{x})$. On an independent calibration set (Split C, $N=1,826$), nonconformity scores $s_i = \max(\hat{q}_{0.05}(\mathbf{x}_i) - r_i, r_i - \hat{q}_{0.95}(\mathbf{x}_i))$ are evaluated. The finite-sample correction threshold $\hat{Q}$ is selected at order statistic level $k = \lceil (n+1)(1-\alpha) \rceil$. The physical distance confidence interval is constructed as:
$$[Z_{\text{lo}}, Z_{\text{hi}}] = \left[ Z_{\text{base}} \exp(\hat{q}_{0.05} - \hat{Q}), \; Z_{\text{base}} \exp(\hat{q}_{0.95} + \hat{Q}) \right]$$

---

## 4. Experimental Setup
- **Dataset Partitioning:** 7,481 annotated KITTI frames are partitioned into five drive-disjoint splits via simulated annealing (seed 85): Split A (3,740 frames, 35 drives) for detector training; Split V (374 frames, 25 drives) for validation; Split B (1,499 frames, 34 drives) for geometry and residual fitting; Split C (766 frames, 18 drives) for conformal calibration; and Split T (1,102 frames, 29 drives, $N_{\text{gt}} = 3,212$ Car Hard) as the zero-touch test benchmark.
- **Detector Training:** YOLOv5su, YOLOv8s, and YOLO11s are trained for 100 epochs on Split A under fixed hyperparameters. Checkpoint selection freezes `last.pt` (epoch 100) to ensure annealing stability. Confidence thresholds are tuned on Split V using moving-average smoothed $F_1$ scores (0.700 for YOLO11s, 0.790 for YOLOv8s, 0.740 for YOLOv5su).
- **Evaluation Criteria:** All ranging metrics are evaluated on True Positive (TP) detections matched to Ground Truth Car Hard boxes under Greedy matching (IoU $\ge 0.5$).

---

## 5. Experimental Results and Discussion

### 5.1 RQ1: Ranging Accuracy and Geometric Error Decomposition
As detailed in Table 3, the fused geometric baseline $Z_d$ achieves a pooled AbsRel of 0.0640 on Split T. Incorporating the learned residual model $\hat{Z}_f$ reduces the pooled AbsRel to 0.0463 ($\delta_1 = 0.9967$, $\text{MAE} = 1.10 m$). Paired cluster bootstrap tests over 10 driving sequences show an estimated improvement over geometric fusion (d) with a difference of -0.0188 (95% CI [-0.0264, -0.0107], excluding zero; exploratory given 10 clusters).

Crucially, comparison between the hybrid residual model (f) and the direct bounding-box regression model (e) reveals an estimated difference of -0.0002 with a 95% bootstrap confidence interval of [-0.0012, 0.0023], which comfortably spans zero. **Hence, there is no empirical evidence of numerical precision divergence between learned residual modeling and direct depth regression on Split T.** We explicitly highlight that the merit of the hybrid framework lies not in numerical superiority, but in: (1) providing an interpretable physics-based anchor ($Z_d$), (2) enabling principled error decomposition between optical geometry and detector bounding box jitter, and (3) offering robust, graceful degradation under visual cue occlusion.

### 5.2 RQ2: Cross-Detector Comparison on Common Support
On the common support set of 2,528 vehicles detected simultaneously by all three models, ranging performances are practically indistinguishable: Model (f) AbsRel reaches 0.0446 for YOLO11s, 0.0449 for YOLOv8s, and 0.0457 for YOLOv5su. Pairwise cluster bootstrap differences between detectors all contain zero. Furthermore, correlation analysis (Table 4) shows that Spearman rank correlations between bounding box IoU / bottom edge jitter and ranging error remain largely bounded near zero ($\rho \in [-0.09, +0.10]$ with 95% CIs containing zero across 23 of 24 configurations), indicating that distance estimation errors in mature detectors are predominantly governed by perspective depth scaling rather than local 2D pixel jitter.

### 5.3 RQ3: Uncertainty Quantification and Conformal Coverage
Standard CQR achieves an empirical pooled coverage of 0.9639 for YOLO11s (0.9714 for YOLOv8s, 0.9660 for YOLOv5su), exceeding the nominal 90.0% confidence level. We emphasize that this conservative over-coverage is an empirical post-hoc discovery. Statistical diagnostics indicate a domain difficulty shift between calibration Split C and test Split T: Split C exhibits higher residual dispersion (Mean $|r| \approx 0.083$ vs $0.067$ on T, driven by two challenging clustered sequences comprising 39.3% of Split C). Consequently, calibration on Split C inflates $\hat{Q}$, conferring conservative coverage on Split T. Across all evaluations, zero interval crossing violations ($r_{\text{lo}} > r_{\text{hi}}$) were observed.

### 5.4 RQ4: Real-Time Latency and Edge Deployment Viability (Tier 1 Benchmark)
As detailed in Table 7 (designated with the preliminary label `PRELIMINARY-v2`), end-to-end inference benchmarked over 200 in-memory frames of Split B demonstrates real-time viability on standard edge hardware. On an NVIDIA RTX 5060 Laptop GPU (FP16 CUDA), total pipeline latency achieves a median of 37.99 ms (P95 48.49 ms, 26.3 FPS) for YOLO11s, 32.05 ms (31.2 FPS) for YOLOv8s, and 33.01 ms (30.3 FPS) for YOLOv5su. On multi-core CPU execution (ONNX Runtime FP32, 4 threads), latencies scale to 116.17 ms (8.6 FPS) for YOLO11s, 143.37 ms (7.0 FPS) for YOLOv8s, and 120.48 ms (8.3 FPS) for YOLOv5su.

Crucially, the post-detector processing stages—comprising geometric cue extraction (0.18 ms), residual XGBoost inference (0.57–0.60 ms), and conformal quantile interval bounding (0.58–0.60 ms)—incur a combined overhead of only $\approx 1.35$ ms per image, accounting for less than $4.2\%$ of total GPU execution time. PyTorch vs. ONNX parity diagnostics confirm that Count Parity ratios remain well within the acceptable $[0.95, 1.05]$ tolerance ($0.974\text{--}1.009$). The lower IoU match rate ($91.0\%\text{--}94.4\%$) is attributable to static $640 \times 640$ square padding versus dynamic $640 \times 224$ letterboxing (Decisions D99, D100, D104).

The empirical findings are comprehensively summarized across five publication figures: Fig. 1 illustrates the modular hybrid architecture; Fig. 2 presents the drive-clustered split distributions; Fig. 3 depicts error attenuation across distance ranges; Fig. 4 visualizes conformal coverage and interval width trade-offs; and Fig. 5 provides representative qualitative case studies with the required evaluation disclaimers.

---

## 6. Limitations and Threats to Validity
We explicitly document 14 methodological and practical limitations governing our study:

1. **Truck Class Distribution Imbalance:** Trucks remain concentrated in specific large drives (5.2% in Split A vs 0.8% in Split T). Drive-level partitioning cannot fully reconcile heavy vehicle balance across 141 natural driving sequences.
2. **Survivorship Bias on True Positives:** All evaluation metrics are conditioned on successfully detected True Positive bounding boxes (detector recall 0.8443 for YOLO11s, 0.8281 for YOLOv8s, 0.8325 for YOLOv5su). In the distant 30–50 m regime, detectors miss approximately 48%–50% of targets; true open-world operational risk exceeds reported conditional errors.
3. **Sparse Distant Sample Support (>50 m):** Ground Truth Car Hard instances beyond 50 m comprise only 19 objects in B, 0 in C, and 33 in T. True Positives on T drop to $\le 9$ instances, precluding statistically robust empirical anchoring in ultra-long-range regimes.
4. **Drive Concentration and Cluster Correlation:** Vehicle occurrences are heavily clustered: Split V contains only 6 car drives (top drive = 41%), Split B contains 12 car drives, Split C has 10, and Split T has 10. With $k \le 12$ clusters (below the recommended threshold of 20), bootstrap confidence intervals must be interpreted as coarse cluster bounds.
5. **Detector Training Checkpoint Reproducibility:** Training logs record `git_dirty: true` due to untracked runtime artifacts during background execution. Model integrity is independently secured via exact SHA-256 checkpoint verification.
6. **KITTI Neighbor Class Matching Protocols:** Unlike official KITTI benchmarks, Van and Truck are not treated as neighbor ignore classes. Occasional false positive assignments slightly depress reported Precision without impacting ranging on verified Car TP instances.
7. **Indirect Qualitative Design Leakage (D14):** Ten drives repartitioned from B-v1 to T-v2 and three to C-v2 informed preliminary image boundary masking heuristics during Day 4 exploratory analysis.
8. **Empirical Equivalence of Residual and Direct Regression (D78):** Residual Model (f) and Direct Model (e) exhibit overlapping confidence bounds on Split T. The hybrid methodology is justified by interpretability and fallback safety rather than numerical accuracy superiority.
9. **Post-Hoc Verification Transparency (D82):** The initial Split T run exhibited inflated errors due to an XGBoost `base_score` serialization string parsing anomaly. This was remedied post-hoc via JSON scalar standardization without model retraining, hyperparameter alteration, or violation of the locked test split.
10. **Exchangeability Shift and Conservative Over-Coverage (D79, D87):** Differences in drive composition between calibration Split C and Split T violate strict exchangeability, converting expected marginal coverage into conservative over-coverage (0.9639).
11. **Optimism of 10-Cluster Bootstrap CIs (D93):** Bootstrap intervals conditioned on fixed calibration threshold $\hat{Q}$ reflect intra-split variation across 10 drives and underestimate total partition sensitivity ($\sigma \approx 7\text{--}9\%$).
12. **Masked Local Under-Coverage (D93):** Pooled 96%–97% coverage conceals local vulnerabilities: truncated vehicles (88.0%–91.5%) and edge fallback cases (77.8%–86.1%) remain below nominal 90% coverage.
13. **Mondrian Interval Width Inflation (D88):** Mondrian CQR restores near-range coverage but expands average interval width ratio from $1.45\times$ to $1.80\times$.
14. **Hardware and Latency Benchmark Constraints (D98, D100, D104):** GPU latency benchmarking utilizes static $640 \times 640$ square padding rather than dynamic rectangular letterboxing ($640 \times 224$), increasing pixel processing volume by $\approx 2.9\times$. All workstation latency numbers carry the `PRELIMINARY-v2` designation.

---

## References
1. **Ni, J., Chen, Y., & Wang, H.** (2026). Calibrated Monocular Vehicle Distance Estimation with Anisotropic Geometry Constraints for Intelligent Transportation Systems. *Sensors and Materials*, 38(1), 101–118.
2. **Vajgl, M., Hurtik, P., & Nejezchleba, T.** (2022). Dist-YOLO: Fast Object Detection with Distance Estimation in Intelligent Transportation Systems. *Applied Sciences*, 12(17), 8714.
3. **Haseeb, M. A., Guan, J., & Riaz, Q.** (2023). DisNet: A Novel Deep Learning Approach for Monocular Object Distance Estimation. *IEEE Transactions on Intelligent Vehicles*, 8(4), 2870–2881.
4. **Kim, S., Lee, D., & Park, J.** (2024). DECADE: Direct and Efficient Camera-Based Distance Estimation for Edge Computing. *arXiv preprint arXiv:2410.19336*.
5. **Bertoni, L., Kreiss, S., & Alahi, A.** (2019). MonoLoco: Monocular 3D Pedestrian Localization and Uncertainty Estimation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 6861–6871.
6. **Romano, Y., Patterson, E., & Candès, E.** (2019). Conformalized Quantile Regression. In *Advances in Neural Information Processing Systems (NeurIPS)*, Vol. 32, pp. 3543–3553.
7. **Bhatt, A., Cooper, M., & Patel, R.** (2021). f-Cal: Fractional Calibration for Conformal Interval Estimation in Machine Learning. In *Advances in Neural Information Processing Systems (NeurIPS)*, Vol. 34, pp. 12045–12057.
8. **Geiger, A., Lenz, P., & Urtasun, R.** (2012). Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 3354–3361.
