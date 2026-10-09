# Calibrated Hybrid Geometry–Learning Monocular Vehicle Distance Estimation with Lightweight YOLO Detectors

**Authors:** Anonymous Submission  
**Target Venue:** IEEE / Applied ITS Conference & Journal Submission  

---

## Abstract
Estimating the distance of preceding vehicles using a single monocular camera is a vital task for advanced driver assistance systems (ADAS) and autonomous driving under stringent computational constraints. While deep learning models offer competitive precision, direct depth regression lacks physical explainability and can become unpredictable under visual corruption or domain shifts. Conversely, pure pinhole geometry offers explicit physical grounding but suffers from systematic biases due to vehicle orientation, 3D center-to-surface offset, and boundary clipping. In this work, we propose a calibrated hybrid monocular distance estimation framework integrating perspective geometry with learned residual correction and conformalized quantile regression (CQR). Three pinhole cues (width, height, and ground-plane constraint) are fused via empirical covariance-weighted log-space pooling to establish a physically grounded baseline. A lightweight gradient-boosted residual model compensates for systematic optical and perspective discrepancies, supplemented by a direct ranging fallback for severely truncated bounding boxes. Furthermore, conformal prediction provides prediction intervals targeting nominal coverage under explicit calibration protocols. Evaluated on the held-out Split T of the KITTI benchmark ($N_{\text{gt}} = {{num:split_t_car_hard_count}}$ Car Hard objects across 10 independent driving sequences), our pipeline achieves an AbsRel of {{num:yolo11s_absrel_f}} ($\delta_1 = {{num:yolo11s_delta1_f}}$) with YOLO11s, maintaining an empirical 90% conformal coverage of {{num:yolo11s_coverage_cqr}} on True Positive detections (exceeding nominal coverage due to cross-drive distribution shifts between calibration and test drives). Across the common support of {{num:common_support_count}} vehicles detected simultaneously by all models, the three detectors exhibit comparable distance estimation precision. While the hybrid pipeline achieves comparable numerical precision to direct bounding-box regression, its core value resides in transparent error decomposition, interpretable physical foundations, and systematic fallback mechanisms under geometric invalidation.

---

## 1. Introduction
Monocular vehicle distance estimation is a foundational perception primitive for forward collision warning (FCW), autonomous emergency braking (AEB), and adaptive cruise control (ACC). Standard automotive setups rely heavily on active sensors such as LiDAR and radar; however, passive camera-based distance estimation provides an indispensable, cost-effective modality with rich semantic perception.

Despite extensive research, existing literature presents three prominent gaps:
- **Gap 1 (Unresolved Systematic Geometric Biases):** Classical perspective geometry relies on rigid assumptions (known 3D vehicle dimensions, flat ground plane, frontal orientation). Few works systematically isolate the error contributions of individual geometric cues across operational distance regimes or decouple geometric formulation errors from 2D detector bounding box jitter.
- **Gap 2 (Uncontrolled Multi-Detector Comparison):** Lightweight detectors (e.g., YOLOv5, YOLOv8, YOLO11) are rarely benchmarked under strictly uniform training recipes and evaluated on common support True Positive sets.
- **Gap 3 (Lack of Calibrated Safety Intervals):** Autonomous decision-making requires well-calibrated confidence bounds. While conformal prediction offers distribution-free guarantees under exchangeability, its empirical behavior under real-world cluster-correlated driving drives remains underexplored.

To bridge these gaps, this study presents:
1. A disciplined drive-based dataset protocol (`splits-v2`) preventing scene overlap across training, calibration, and test phases.
2. An interpretable hybrid pipeline uniting multi-cue pinhole geometry, covariance shrinkage fusion, and residual tree ensembles.
3. An uncertainty quantification framework via Conformalized Quantile Regression (CQR), accompanied by an empirical examination of exchangeability and domain shift across driving sequences.

---

## 2. Related Work
- **Monocular Ranging via Geometry & Neural Approximations:** Dist-YOLO (Vajgl et al., 2022) incorporated distance estimation heads directly into YOLOv3. DisNet (Haseeb et al., 2018) utilized multilayer perceptrons operating on bounding box features. DECADE (Shahzad et al., 2024) benchmarked lightweight YOLO variants for mobile ADAS. Formulations such as Anisotropic Geometry Loss (AGL) (Christanto & Miaou, 2026), soft-sensor perspective geometry with optimized YOLOv5 (Ni et al., 2026), MonoLoco (Bertoni et al., 2019), and classical monocular visual ranging foundations (Dagan et al., 2004; Geiger et al., 2012) established geometric baselines. We position our architecture as a principled hybrid synthesis of these precedents without claiming priority.
- **Conformal Prediction in Autonomous Perception:** Conformalized Quantile Regression (CQR) (Romano et al., 2019) extends conformal inference to heteroscedastic interval estimation. In autonomous perception, f-Cal (Bhatt et al., 2021) addressed aleatoric uncertainty calibration. Covariance shrinkage methods such as Oracle Approximating Shrinkage (OAS) (Chen et al., 2010) and scalable tree ensembles (Chen & Guestrin, 2016) provide stable regression backbones. Our work investigates the practical behavior of log-space CQR under correlated cluster shifts in natural driving datasets.

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
where weights $\mathbf{w}$ are obtained by solving the minimum-variance portfolio problem $\min_{\mathbf{w}} \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$ subject to $\sum w_k = 1, w_k \ge 0$ via an active-set non-negative weight procedure on the Oracle Approximating Shrinkage (OAS) covariance matrix $\mathbf{\Sigma}$ (Chen et al., 2010).

### 3.3 Residual Calibration Model and Fallback Mechanism
The baseline estimation is defined as $Z_{\text{base}} = Z_d$ for pattern combinations with valid cues, and $Z_{\text{base}} = Z_e$ (a direct regression model trained on bounding box features) when all geometric cues are invalidated (Pattern 000). A gradient-boosted tree ensemble (Chen & Guestrin, 2016) models the log-ratio residual:
$$\hat{r} = f(\mathbf{x}) \approx \ln Z_{\text{gt}} - \ln Z_{\text{base}}, \quad \hat{Z}_f = Z_{\text{base}} \exp(\hat{r})$$
operating strictly on a whitelist of 17 test-time observable 2D bounding box and geometric features.

### 3.4 Conformalized Quantile Regression (CQR)
Two quantile regression models estimate the 5th and 95th percentiles of residual error: $\hat{q}_{0.05}(\mathbf{x})$ and $\hat{q}_{0.95}(\mathbf{x})$. On an independent calibration set (Split C, $N_{\text{gt}}=1,826$ Car Hard objects, yielding $n=1,489$ calibration True Positives), nonconformity scores $s_i = \max(\hat{q}_{0.05}(\mathbf{x}_i) - r_i, r_i - \hat{q}_{0.95}(\mathbf{x}_i))$ are evaluated. The calibration threshold $\hat{Q}$ is selected at order statistic level $k = \lceil (n+1)(1-\alpha) \rceil$. The physical distance confidence interval is constructed as:
$$[Z_{\text{lo}}, Z_{\text{hi}}] = \left[ Z_{\text{base}} \exp(\hat{q}_{0.05} - \hat{Q}), \; Z_{\text{base}} \exp(\hat{q}_{0.95} + \hat{Q}) \right]$$

---

## 4. Experimental Setup
- **Dataset Partitioning:** 7,481 annotated KITTI frames are partitioned into five drive-disjoint splits, with Split A and V established first and Splits B, C, T repartitioned via simulated annealing (seed 85): Split A ({{num:split_a_frames}} frames, 35 drives) for detector training; Split V (374 frames, 25 drives) for validation; Split B (1,499 frames, 34 drives) for geometry and residual fitting; Split C (766 frames, 18 drives) for conformal calibration; and Split T ({{num:split_t_frames}} frames, 29 drives, $N_{\text{gt}} = {{num:split_t_car_hard_count}}$ Car Hard) as the zero-touch test benchmark.
- **Detector Training:** YOLOv5su, YOLOv8s, and YOLO11s are trained for 100 epochs on Split A under fixed hyperparameters. Checkpoint selection freezes `last.pt` (epoch 100) to ensure annealing stability. Confidence thresholds are tuned on Split V using moving-average smoothed $F_1$ scores (0.700 for YOLO11s, 0.790 for YOLOv8s, 0.740 for YOLOv5su).
- **Evaluation Criteria:** All ranging metrics are evaluated on True Positive (TP) detections matched to Ground Truth Car Hard boxes under Greedy matching (IoU $\ge 0.5$).

---

## 5. Experimental Results and Discussion

### 5.1 Error Decomposition and Geometric Baseline Analysis (Split B OOF and Split T)
As summarized in Table 2, evaluating individual perspective cues on detector bounding boxes across Split B out-of-fold predictions isolates fundamental optical limitations:
1. **Width Cue ($Z_w$):** Yields the highest error (Pooled AbsRel {{num:yolo11s_b_absrel_zw}}, MAE {{num:yolo11s_b_mae_zw}}, $\delta_1 = {{num:yolo11s_b_delta1_zw}}$), heavily compromised by vehicle orientation and aspect ratio distortion under oblique angles.
2. **Ground Contact Cue ($Z_g$):** Suffers from pitch variations and non-flat road profiles (Pooled AbsRel {{num:yolo11s_b_absrel_zg}}, MAE {{num:yolo11s_b_mae_zg}}, $\delta_1 = {{num:yolo11s_b_delta1_zg}}$).
3. **Height Cue ($Z_h$):** Represents the most reliable single cue (Pooled AbsRel {{num:yolo11s_b_absrel_zh}}, MAE {{num:yolo11s_b_mae_zh}}, $\delta_1 = {{num:yolo11s_b_delta1_zh}}$), benefitting from relatively stable 3D vehicle heights across passenger car classes.

Optimal covariance shrinkage fusion ($Z_d$) effectively combines $Z_h$ and $Z_g$, reducing the pooled AbsRel to {{num:yolo11s_b_absrel_zd}} (MAE {{num:yolo11s_b_mae_zd}}). Ablation experiments on Split B further confirm that dropping $Z_h$ incurs a measurable degradation ($\Delta_{\text{pooled}} = +{{num:yolo11s_abl_drop_zh_delta_pooled}}$, 95% bootstrap CI {{num:yolo11s_abl_drop_zh_delta_pooled:display_ci}}, strictly excluding zero), isolating height as the dominant physical anchor. Dropping the bounding box geometry feature group degrades performance ($\Delta_{\text{macro}} = +{{num:yolo11s_abl_drop_bbox_delta_macro}}$, CI {{num:yolo11s_abl_drop_bbox_delta_macro:display_ci}}), while removing validity flags produces a marginal shift ($\Delta_{\text{pooled}} = +{{num:yolo11s_abl_drop_validity_delta_pooled}}$, CI {{num:yolo11s_abl_drop_validity_delta_pooled:display_ci}}).

Transferring this grounded geometric base to the locked held-out Split T (Table 3), the fused geometric baseline $Z_d$ achieves a pooled AbsRel of {{num:yolo11s_absrel_d}} on YOLO11s. Incorporating learned residual correction $\hat{Z}_f$ reduces the pooled AbsRel to {{num:yolo11s_absrel_f}} ($\delta_1 = {{num:yolo11s_delta1_f}}$, $\text{MAE} = {{num:yolo11s_mae_f}}$). Paired cluster bootstrap tests over 10 driving sequences show an estimated improvement over geometric fusion (d) with a difference of {{num:yolo11s_diff_f_minus_d}} (95% CI {{num:yolo11s_diff_f_minus_d:display_ci}}, excluding zero; exploratory given 10 clusters).

Crucially, comparison between the hybrid residual model (f) and the direct bounding-box regression model (e) reveals an estimated difference of {{num:yolo11s_diff_f_minus_e}} with a 95% bootstrap confidence interval of {{num:yolo11s_diff_f_minus_e:display_ci}}, which spans zero at the 10-cluster level. **Hence, there is no empirical evidence of numerical precision divergence between learned residual modeling and direct depth regression on Split T.** We explicitly highlight that the merit of the hybrid framework lies not in numerical superiority, but in: (1) providing an interpretable physics-based anchor ($Z_d$), (2) enabling principled error decomposition between optical geometry and detector bounding box jitter, and (3) offering robust, graceful degradation under visual cue occlusion.

### 5.2 RQ2: Cross-Detector Comparison on Common Support
On the common support set of {{num:common_support_count}} vehicles detected simultaneously by all three models, ranging performances are practically indistinguishable: Model (f) AbsRel reaches {{num:yolo11s_common_absrel_f}} for YOLO11s, {{num:yolov8s_common_absrel_f}} for YOLOv8s, and {{num:yolov5su_common_absrel_f}} for YOLOv5su. Pairwise cluster bootstrap differences between detectors all contain zero. Furthermore, correlation analysis (Table 4) shows that Spearman rank correlations between bounding box IoU / bottom edge jitter and ranging error remain largely bounded near zero ($\rho \in [-0.09, +0.10]$ with 95% CIs containing zero across 23 of 24 configurations), indicating that distance estimation errors in mature detectors are predominantly governed by perspective depth scaling rather than local 2D pixel jitter.

### 5.3 RQ3: Uncertainty Quantification and Conformal Coverage
Standard CQR achieves an empirical pooled coverage of {{num:yolo11s_coverage_cqr}} for YOLO11s ({{num:yolov8s_coverage_cqr}} for YOLOv8s, {{num:yolov5su_coverage_cqr}} for YOLOv5su), exceeding the nominal 90.0% confidence level. We emphasize that this conservative over-coverage is an empirical post-hoc discovery. Statistical diagnostics indicate a domain difficulty shift between calibration Split C and test Split T: Split C exhibits higher residual dispersion (Mean $|r| \approx 0.083$ vs $0.067$ on T, driven by two challenging clustered sequences comprising 39.3% of Split C). Consequently, calibration on Split C inflates $\hat{Q}$, conferring conservative coverage on Split T. Across all evaluations, zero interval crossing violations ($r_{\text{lo}} > r_{\text{hi}}$) were observed.

**Split-Stability Across 20 Resplits:** To test whether over-coverage is an artifact of the specific Split C / Split T partition, we evaluated 20 drive-disjoint resplits of B, C, and T under the identical simulated annealing protocol (Task T08). Across these 20 resplits, empirical pooled coverage averages {{num:yolo11s_resplit20_mean_cov}} ($\pm {{num:yolo11s_resplit20_std_cov}}$, range [{{num:yolo11s_resplit20_min_cov}}, {{num:yolo11s_resplit20_max_cov}}]) for YOLO11s, {{num:yolov8s_resplit20_mean_cov}} ($\pm {{num:yolov8s_resplit20_std_cov}}$, range [{{num:yolov8s_resplit20_min_cov}}, {{num:yolov8s_resplit20_max_cov}}]) for YOLOv8s, and {{num:yolov5su_resplit20_mean_cov}} ($\pm {{num:yolov5su_resplit20_std_cov}}$, range [{{num:yolov5su_resplit20_min_cov}}, {{num:yolov5su_resplit20_max_cov}}]) for YOLOv5su. The observed cross-resplit standard deviation ($\sigma \approx 8.4\%\text{--}8.7\%$) demonstrates substantial natural drive-to-drive heterogeneity across KITTI sequences, confirming our post-hoc hypothesis that exchangeability between small drive clusters is imperfect and that the 96%–97% coverage observed on Split T represents an expected consequence of sample drive selection rather than an invariant property of the method.

### 5.4 RQ4: Hardware Latency Benchmark and Deployment Viability (Tier 1 Benchmark)
As detailed in Table 7 (designated with the preliminary label `PRELIMINARY-v2`), end-to-end inference benchmarked over 200 in-memory frames of Split B indicates real-time feasibility on workstation laptop hardware. On an NVIDIA RTX 5060 Laptop GPU (FP16 CUDA), total pipeline latency achieves a median of {{num:yolo11s_latency_gpu_median}} (P95 {{num:yolo11s_latency_gpu_p95}}, {{num:yolo11s_latency_gpu_fps}}) for YOLO11s, {{num:yolov8s_latency_gpu_median}} ({{num:yolov8s_latency_gpu_fps}}) for YOLOv8s, and {{num:yolov5su_latency_gpu_median}} ({{num:yolov5su_latency_gpu_fps}}) for YOLOv5su. On multi-core CPU execution (ONNX Runtime FP32, 4 threads), latencies scale to {{num:yolo11s_latency_cpu_median}} ({{num:yolo11s_latency_cpu_fps}}) for YOLO11s, {{num:yolov8s_latency_cpu_median}} ({{num:yolov8s_latency_cpu_fps}}) for YOLOv8s, and {{num:yolov5su_latency_cpu_median}} ({{num:yolov5su_latency_cpu_fps}}) for YOLOv5su.

Crucially, the post-detector processing stages—comprising geometric cue extraction (0.18 ms), residual XGBoost inference (0.57–0.60 ms), and conformal quantile interval bounding (0.58–0.60 ms)—incur a combined overhead of only $\approx 1.35$ ms per image, accounting for less than $4.2\%$ of total GPU execution time. PyTorch vs. ONNX parity diagnostics confirm that Count Parity ratios remain well within the acceptable $[0.95, 1.05]$ tolerance ($0.974\text{--}1.009$). We hypothesize that the lower IoU match rate ($91.0\%\text{--}94.4\%$) arises from static $640 \times 640$ square padding versus dynamic $640 \times 224$ letterboxing.

The empirical findings are comprehensively summarized across five publication figures: Fig. 1 illustrates the modular hybrid architecture; Fig. 2 presents the drive-clustered split distributions; Fig. 3 depicts error attenuation across distance ranges; Fig. 4 visualizes conformal coverage and interval width trade-offs; and Fig. 5 provides representative qualitative case studies with the required evaluation disclaimers.

---

## 6. Limitations and Threats to Validity
We explicitly document 14 methodological and practical limitations governing our study:

1. **Truck Class Distribution Imbalance:** Trucks remain concentrated in specific large drives (5.2% in Split A vs 0.8% in Split T). Drive-level partitioning cannot fully reconcile heavy vehicle balance across 141 natural driving sequences.
2. **Survivorship Bias on True Positives and Target Matching Scope:** All ranging evaluation metrics are conditioned on successfully detected True Positive bounding boxes (detector recall {{num:yolo11s_recall}} for YOLO11s, {{num:yolov8s_recall}} for YOLOv8s, {{num:yolov5su_recall}} for YOLOv5su) matched greedily at IoU $\ge 0.5$. Because our evaluation isolates ranging error conditional on true positive localization, detection mAP@0.7 is not evaluated; open-world operational risk under false negative misses (which reach ~48%–50% in the distant 30–50 m regime) exceeds reported conditional ranging errors.
3. **Sparse Distant Sample Support (>50 m):** Ground Truth Car Hard instances beyond 50 m comprise only 19 objects in B, 0 in C, and 33 in T. True Positives on T drop to $\le 9$ instances, precluding statistically robust empirical anchoring in ultra-long-range regimes.
4. **Drive Concentration and Cluster Correlation:** Vehicle occurrences are heavily clustered: Split V contains only 6 car drives (top drive = 41%), Split B contains 12 car drives, Split C has 10, and Split T has 10. With $k \le 12$ clusters (below the recommended threshold of 20), bootstrap confidence intervals must be interpreted as coarse cluster bounds.
5. **Detector Training Checkpoint Reproducibility:** Training logs record `git_dirty: true` due to untracked runtime artifacts during background execution. Model integrity is independently secured via exact SHA-256 checkpoint verification.
6. **KITTI Neighbor Class Matching Protocols:** Unlike official KITTI benchmarks, Van and Truck are not treated as neighbor ignore classes. Occasional false positive assignments slightly depress reported Precision without impacting ranging on verified Car TP instances.
7. **Indirect Qualitative Design Leakage:** Ten drives repartitioned from B-v1 to T-v2 and three to C-v2 informed preliminary image boundary masking heuristics during Day 4 exploratory analysis.
8. **Empirical Indistinguishability of Residual and Direct Regression:** Residual Model (f) and Direct Model (e) exhibit overlapping confidence bounds on Split T. The hybrid methodology is justified by physical interpretability and fallback safety rather than numerical accuracy superiority.
9. **Post-Hoc Verification Transparency:** The initial Split T run exhibited inflated errors due to an XGBoost `base_score` serialization string parsing anomaly in v1. This was remedied post-hoc via JSON scalar standardization in v2 without model retraining, hyperparameter alteration, or violation of the locked held-out test split protocol.
10. **Exchangeability Shift and Conservative Over-Coverage:** Differences in drive composition between calibration Split C and Split T violate strict exchangeability, converting expected marginal coverage into conservative over-coverage ({{num:yolo11s_coverage_cqr}}).
11. **Optimism of 10-Cluster Bootstrap CIs:** Bootstrap intervals conditioned on fixed calibration threshold $\hat{Q}$ reflect intra-split variation across 10 drives and underestimate total partition sensitivity ($\sigma \approx 7\text{--}9\%$).
12. **Masked Local Under-Coverage:** Pooled 96%–97% coverage conceals local vulnerabilities: truncated vehicles (88.0%–91.5%) and edge fallback cases (77.8%–86.1%) remain below nominal 90% coverage.
13. **Mondrian Interval Width Inflation:** Mondrian CQR restores coverage in the near 0–10 m distance band, but expands average interval width ratio from $1.45\times$ to $1.80\times$.
14. **Hardware and Latency Benchmark Constraints:** GPU latency benchmarking utilizes static $640 \times 640$ square padding rather than dynamic rectangular letterboxing ($640 \times 224$), increasing pixel processing volume by $\approx 2.9\times$. All workstation latency numbers carry the `PRELIMINARY-v2` designation.

---

## References
1. **Ni, Z., Shi, J., Li, L., & Ni, C.** (2026). Real-time Vehicle Detection and Distance Estimation: Soft-sensor Approach Using Optimized YOLOv5 and Perspective Geometry. *Sensors and Materials*, 38(1).
2. **Vajgl, M., Hurtik, P., & Nejezchleba, T.** (2022). Dist-YOLO: Fast Object Detection with Distance Estimation in Intelligent Transportation Systems. *Applied Sciences*, 12(3), 1354.
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
