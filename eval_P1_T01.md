USN1,USN2,USN3

# Evaluation - P1 - T01
Images evaluated: 100 (low: `data/LOL-v2/Real_captured/Test/Low`, reference: `data/LOL-v2/Real_captured/Test/Normal`), resized to max side 600px.

## 1. PSNR / SSIM / LPIPS / NIQE / latency / noise / edge preservation (mean over images)
| method                        |   latency_ms |   noise_rel |   edge_pres_vs_input |   overexposed_pct |   color_cast |
|:------------------------------|-------------:|------------:|---------------------:|------------------:|-------------:|
| Classical only                |      155.212 |      0.058  |               0.838  |            0      |       5.4261 |
| Deep learning only (Zero-DCE) |      825.516 |      0.1218 |               0.5914 |            0.0944 |       6.9468 |
| Hybrid (proposed)             |      547.938 |      0.0485 |               0.8369 |            0      |       4.2997 |

## 2. Ablation study (hybrid pipeline, one component removed at a time)
| method                     |   latency_ms |   noise_rel |   edge_pres_vs_input |   overexposed_pct |   color_cast |
|:---------------------------|-------------:|------------:|---------------------:|------------------:|-------------:|
| Full hybrid                |      550.473 |      0.0485 |               0.8369 |            0      |       4.2997 |
| No adaptive params (fixed) |      549.919 |      0.0208 |               0.8517 |            0      |       4.1351 |
| No denoise                 |     1104.72  |      0.1506 |               0.744  |            0      |       4.1588 |
| No Retinex                 |      554.42  |      0.0373 |               0.8925 |            0      |       1.8764 |
| No CLAHE                   |     1066.45  |      0.0372 |               0.8666 |            0      |       3.9129 |
| No deep branch             |      130.798 |      0.058  |               0.838  |            0      |       5.4261 |
| No spatial fusion          |      557.073 |      0.0443 |               0.8207 |            0      |       3.8497 |
| No quality gate            |      530.733 |      0.0465 |               0.8367 |            0      |       3.9236 |
| No colour correction       |      549.074 |      0.0472 |               0.839  |            0      |       9.2202 |
| No overexposure guard      |      546.564 |      0.0485 |               0.8381 |            0.0288 |       4.2993 |

Metric notes: PSNR/SSIM/LPIPS/Delta-E need ground truth; `noise_rel` = Immerkaer sigma / mean luma (lower = better noise suppression); `edge_pres_*` = gradient-magnitude correlation (higher = better); `overexposed_pct` = % pixels with max-RGB>0.98; `color_cast` = Lab (a,b) offset from neutral. NIQE needs `pip install pyiqa`; LPIPS needs `pip install lpips`; missing metrics are omitted.