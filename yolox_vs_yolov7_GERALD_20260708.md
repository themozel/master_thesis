# YOLOX vs YOLOv7 on GERALD (2026-07-08)

## Evaluation Artifacts Used
- data/GERALD/results_yolox_test/metrics_report.json
- data/GERALD/results_yolox_test/analysis_summary.json
- inference_results/yolov7_20260708_GERALD/metrics_report.json
- inference_results/yolov7_20260708_GERALD/summary.txt
- inference_results/yolov7_20260708_GERALD/detailed_results.csv
- runs/train/20260705_GERALD/confusion_matrix.png

## Headline
YOLOX is still stronger than YOLOv7 on the GERALD test split.

This rerun now includes YOLOv7 detection AP metrics, so the core comparison is much more direct than before.

## Side-by-Side Metrics
| Metric | YOLOX | YOLOv7 |
|---|---:|---:|
| Images processed | 500 | 500 |
| Total GT objects | 3554 | 3554 |
| Total predictions | 3459 | 3114 |
| mAP@0.50:0.95 | 0.3324 | 0.1233 |
| mAP@0.50 | 0.6023 | 0.2317 |
| TP / FP / FN | 2606 / 853 / 948 | not available at object level |
| Micro precision (object-level) | 0.7534 | not available |
| Micro recall (object-level) | 0.7333 | not available |
| Micro F1 (object-level) | 0.7432 | not available |
| Reported precision | not directly in summary file | 1.0000 |
| Reported recall | not directly in summary file | 0.9840 |
| Reported F1 | not directly in summary file | 0.9919 |
| Average latency (ms) | not present in these YOLOX outputs | 40.93 |

## Per-Class AP@0.50 Comparison
YOLOv7 per-class AP@0.50 comes from inference_results/yolov7_20260708_GERALD/metrics_report.json.

YOLOX per-class AP@0.50 was computed from data/GERALD/results_yolox_test/predictions_coco.json against data/GERALD/annotations/instances_test2017.json using IoU=0.50 and 101-point interpolation.

| Class | YOLOX AP@0.50 | YOLOv7 AP@0.50 | Gap (YOLOX-YOLOv7) |
|---|---:|---:|---:|
| El_6 | 0.2871 | 0.0000 | 0.2871 |
| Hectometer_Sign | 0.6505 | 0.6048 | 0.0456 |
| Hp_0_HV | 0.8092 | 0.4179 | 0.3913 |
| Hp_0_Ks | 0.8606 | 0.2631 | 0.5975 |
| Hp_0_Sh | 0.5695 | 0.2086 | 0.3608 |
| Hp_1 | 0.8403 | 0.4339 | 0.4064 |
| Hp_2 | 0.8602 | 0.3231 | 0.5371 |
| ICE | N/A | N/A | N/A |
| Ks_1 | 0.8877 | 0.6760 | 0.2118 |
| Ks_2 | 0.7883 | 0.2015 | 0.5867 |
| LZB | N/A | N/A | N/A |
| Lf_2 | N/A | N/A | N/A |
| Lf_3 | 0.0000 | 0.0000 | 0.0000 |
| Lf_6 | 0.7129 | 0.2948 | 0.4180 |
| Lf_7 | 0.3788 | 0.0000 | 0.3788 |
| Mast_Sign_WRW | 0.8394 | 0.7278 | 0.1116 |
| Mast_Sign_WYWYW | 0.7723 | 0.0000 | 0.7723 |
| Mast_Sign_Y_Triangle | 0.8368 | 0.7227 | 0.1141 |
| Ne_1 | 0.7327 | 0.0000 | 0.7327 |
| Ne_2 | 0.8502 | 0.8246 | 0.0255 |
| Ne_3_1 | 0.6277 | 0.2059 | 0.4217 |
| Ne_3_2 | 0.6478 | 0.0000 | 0.6478 |
| Ne_3_3 | 0.5050 | 0.0000 | 0.5050 |
| Ne_3_4 | N/A | N/A | N/A |
| Ne_3_5 | N/A | N/A | N/A |
| Ne_4 | 0.7129 | 0.0000 | 0.7129 |
| Ne_5 | 0.3579 | 0.0000 | 0.3579 |
| Ne_6 | N/A | N/A | N/A |
| Ne_7a | 0.5050 | 0.0000 | 0.5050 |
| Ne_7b | N/A | N/A | N/A |
| Platform_Display | 0.6293 | 0.3582 | 0.2710 |
| Platform_Text_Sign | 0.3913 | 0.1559 | 0.2354 |
| Platform_Track_Sign | 0.3449 | 0.2391 | 0.1058 |
| Platform_Warn_Sign | 0.7174 | 0.7429 | -0.0254 |
| Ra_10 | N/A | N/A | N/A |
| Ride_Indicator_1 | N/A | N/A | N/A |
| Ride_Indicator_Off | 0.0000 | 0.0000 | 0.0000 |
| Sh_0 | 0.6517 | 0.0000 | 0.6517 |
| Sh_1 | 0.5050 | 0.0000 | 0.5050 |
| Sh_2 | 0.4713 | 0.0000 | 0.4713 |
| Sign_Back | 0.4111 | 0.2862 | 0.1249 |
| Signal_Back | 0.5810 | 0.3542 | 0.2268 |
| Signal_Identifier_Sign | 0.6586 | 0.6276 | 0.0310 |
| Signal_Invalid | 0.6634 | 0.0000 | 0.6634 |
| Signal_Off | 0.6200 | 0.3780 | 0.2420 |
| So_20_Left | 0.5545 | 0.0000 | 0.5545 |
| So_20_Right | 0.3861 | 0.0000 | 0.3861 |
| Traffic_Light | 0.2931 | 0.0000 | 0.2931 |
| Traffic_Sign | 0.4613 | 0.0297 | 0.4316 |
| Vr_0 | 0.7671 | 0.5084 | 0.2588 |
| Vr_1 | 0.8403 | 0.6077 | 0.2327 |
| Vr_2 | 0.9381 | 0.1742 | 0.7639 |
| Wn_1 | 1.0000 | 0.0000 | 1.0000 |
| Wn_2 | N/A | N/A | N/A |
| Zs_2 | 0.4158 | 0.0000 | 0.4158 |
| Zs_2v | 0.5363 | 0.0000 | 0.5363 |
| Zs_3 | 0.7207 | 0.5001 | 0.2207 |
| Zs_3v | 0.7495 | 0.0000 | 0.7495 |
| Zs_6 | 0.0000 | 0.0000 | 0.0000 |
| Zs_Off | 0.7766 | 0.7183 | 0.0583 |

## Per-Class AP@0.50 vs Paper (Main Classes)
Paper source used: Table 6 (AP50 row) from /mnt/c/Amid/Uni/Master Thesis/Papers/Paper 7a — GERALD.pdf.

Class mapping from paper to this project:
- Hp 0 (H/V) -> Hp_0_HV
- Hp 0 (Ks) -> Hp_0_Ks

| Class | YOLOX AP@0.50 | YOLOv7 AP@0.50 | Paper AP50 (YOLOv4) | YOLOX-Paper | YOLOv7-Paper |
|---|---:|---:|---:|---:|---:|
| Hp_0_HV | 0.8092 | 0.4179 | 0.5800 | 0.2292 | -0.1621 |
| Hp_1 | 0.8403 | 0.4339 | 0.7200 | 0.1203 | -0.2861 |
| Hp_2 | 0.8602 | 0.3231 | 0.7300 | 0.1302 | -0.4069 |
| Vr_0 | 0.7671 | 0.5084 | 0.7600 | 0.0071 | -0.2516 |
| Vr_1 | 0.8403 | 0.6077 | 0.7700 | 0.0703 | -0.1623 |
| Vr_2 | 0.9381 | 0.1742 | 0.7600 | 0.1781 | -0.5858 |
| Hp_0_Ks | 0.8606 | 0.2631 | 0.5400 | 0.3206 | -0.2769 |
| Ks_1 | 0.8877 | 0.6760 | 0.7900 | 0.0977 | -0.1140 |
| Ks_2 | 0.7883 | 0.2015 | 0.7800 | 0.0083 | -0.5785 |

Paper mAP@0.50 over those main classes is 0.7200.

## AP Gap (YOLOX - YOLOv7)
- mAP@0.50 gap: 0.3706
- mAP@0.50:0.95 gap: 0.2091
- Relative AP retained by YOLOv7 at AP@0.50: 38.47% of YOLOX
- Relative AP retained by YOLOv7 at AP@0.50:0.95: 37.09% of YOLOX

## YOLOv7 Output Sanity Check
From detailed_results.csv in this rerun:
- Mean predictions per image: 6.228
- Max predictions in an image: 29
- Images with zero predictions: 8/500

This confirms the rerun is producing multi-object outputs.

## YOLOv7 Confusion Matrix Findings
Confusion matrix used: [runs/train/20260705_GERALD/confusion_matrix.png](runs/train/20260705_GERALD/confusion_matrix.png)

Main observations from the matrix:
- The diagonal is present for several frequent classes, which means YOLOv7 is learning class identity for many signs.
- The bottom row (predicted as background FN) is strong across many true classes, indicating missed detections are a major error mode.
- Hp and Ks-related classes show nearby off-diagonal leakage.
- Platform_Display, Platform_Text_Sign, Platform_Track_Sign, and Platform_Warn_Sign show mutual confusion.
- Vr_0, Vr_1, and Vr_2 form a confusion block.
- Sign_Back and Signal_Back are occasionally mixed.
- The rightmost background FP column is present but visually weaker than the background FN row, suggesting recall loss is more dominant than background false alarms.

## Interpretation
The comparison is now substantially more apples-to-apples because YOLOv7 includes detection AP metrics.

YOLOX remains better on both AP measures by a wide margin, while YOLOv7 now shows strong image-level presence recall and higher prediction coverage than the previous run.

The YOLOv7 confusion matrix supports the AP gap diagnosis: performance is constrained more by missed objects and within-family class confusions than by pure background false positives.

## Why YOLOv7 Is Worse Here
Most likely causes in this setup are:
- Optimization and recipe mismatch: YOLOX and YOLOv7 use different training schedules, augmentations, and assignment logic, and one recipe can fit GERALD better than the other.
- Class imbalance sensitivity: several rare classes collapse to AP50 near zero in YOLOv7, which usually indicates imbalance and hard-positive scarcity.
- Confusion among visually similar classes: the confusion matrix shows leakage in Hp/Ks, Platform_*, Vr_*, and Sign_Back vs Signal_Back groups.
- Recall bottleneck: the strong background-FN pattern suggests missed objects are a larger issue than background false alarms.

Answer to the anchor-free question:
- This is not primarily because of anchor-free vs anchor-based design by itself.
- YOLOX being anchor-free can help with simpler assignment and better generalization on some datasets, but architecture family alone rarely explains a 0.37 AP50 gap.
- The bigger drivers are usually data distribution fit, label quality, class balance, augmentation policy, and hyperparameter tuning per model.

## How To Improve YOLOX Further
Recommended order (highest expected gain first):
- Improve hard-class data: add and oversample classes with largest residual errors (for example Vr_2 edge cases, Platform_*, Hp_0_* confusions).
- Run targeted copy-paste/mosaic with constraints: place confusing classes in varied backgrounds while preserving realistic scale and aspect.
- Tune confidence and NMS IoU on validation: sweep confidence thresholds and NMS IoU, then lock per-run settings before test export.
- Increase effective resolution for small/distant signals: train/eval at a larger img size if GPU allows and compare AP50 for small classes.
- Use class-balanced sampling or reweighting: reduce dominance of frequent classes and stabilize learning for long-tail signs.
- Extend training with early-stop on AP50 and AP50:95: keep best checkpoint by validation AP, not only loss.
- Audit labels for top confusion pairs: correct borderline/ambiguous labels in Hp/Ks and Sign_Back/Signal_Back groups.
- Use TTA at inference for final benchmark: horizontal-scale/flip test-time augmentation can improve AP at modest latency cost.

Suggested validation protocol for fair iteration:
- Keep the same test split and evaluator.
- Change one factor at a time.
- Track global AP50/AP50:95 plus per-class AP50 deltas for the nine paper main classes.

## Conclusion
For detector quality on this evaluation setup, YOLOX is still the stronger model.

Using the YOLOv7 confusion matrix, the main improvement targets are clear:
- Raise recall for classes that frequently fall into background FN.
- Reduce confusion among closely related families (Hp/Ks, Platform_*, Vr_*, Sign_Back vs Signal_Back).