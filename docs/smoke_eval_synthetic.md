> NOTE: generated with the synthetic-bootstrap weights (weights/zerodce_synthetic_bootstrap.pth), before swapping in the official Zero-DCE weights. Re-run scripts/evaluate.py for current numbers.

1AB23CS001,1AB23CS002

# Evaluation - P1 - T01
Images evaluated: 4 (low: `samples/low`, reference: `samples/high`), resized to max side 600px.

## 1. PSNR / SSIM / LPIPS / NIQE / latency / noise / edge preservation (mean over images)
| method                        |    psnr |   ssim |   latency_ms |   noise_rel |   edge_pres_vs_gt |   edge_pres_vs_input |   overexposed_pct |   color_cast |   delta_e |
|:------------------------------|--------:|-------:|-------------:|------------:|------------------:|---------------------:|------------------:|-------------:|----------:|
| Classical only                | 10.967  | 0.5061 |      67.2938 |      0.07   |            0.7491 |               0.8639 |            0      |       7.7519 |   31.3043 |
| Deep learning only (Zero-DCE) | 14.0684 | 0.2589 |     555.318  |      0.1957 |            0.6561 |               0.5771 |            0.0084 |       8.8441 |   37.9692 |
| Hybrid (proposed)             | 16.1917 | 0.6623 |     618.422  |      0.0525 |            0.8013 |               0.8145 |            0      |       2.4055 |   24.1344 |

## 2. Ablation study (hybrid pipeline, one component removed at a time)
| method                     |    psnr |   ssim |   latency_ms |   noise_rel |   edge_pres_vs_gt |   edge_pres_vs_input |   overexposed_pct |   color_cast |   delta_e |
|:---------------------------|--------:|-------:|-------------:|------------:|------------------:|---------------------:|------------------:|-------------:|----------:|
| Full hybrid                | 16.1917 | 0.6623 |     593.356  |      0.0525 |            0.8013 |               0.8145 |            0      |       2.4055 |   24.1344 |
| No adaptive params (fixed) | 16.3083 | 0.6646 |     642.235  |      0.0183 |            0.741  |               0.5789 |            0      |       4.6646 |   23.155  |
| No denoise                 | 14.0101 | 0.2822 |     562.504  |      0.2011 |            0.6918 |               0.7579 |            0      |       3.9005 |   36.7338 |
| No Retinex                 | 15.9186 | 0.6324 |     581.755  |      0.0504 |            0.7809 |               0.7922 |            0      |       1.7879 |   25.3236 |
| No CLAHE                   | 16.2619 | 0.6493 |     465.23   |      0.0499 |            0.796  |               0.8027 |            0      |       2.2989 |   24.3701 |
| No deep branch             | 10.967  | 0.5061 |      60.4328 |      0.07   |            0.7491 |               0.8639 |            0      |       7.7519 |   31.3043 |
| No spatial fusion          | 16.6298 | 0.6494 |     632.787  |      0.0513 |            0.8032 |               0.7801 |            0      |       2.3725 |   24.0966 |
| No quality gate            | 14.8251 | 0.6591 |     579.779  |      0.0555 |            0.7996 |               0.8402 |            0      |       2.4105 |   25.7733 |
| No colour correction       | 17.4254 | 0.6794 |     423.976  |      0.0501 |            0.8013 |               0.818  |            0      |       8.8492 |   20.1557 |
| No overexposure guard      | 16.1916 | 0.6623 |     580.227  |      0.0525 |            0.8012 |               0.8145 |            0.0013 |       2.4052 |   24.135  |

Metric notes: PSNR/SSIM/LPIPS/Delta-E need ground truth; `noise_rel` = Immerkaer sigma / mean luma (lower = better noise suppression); `edge_pres_*` = gradient-magnitude correlation (higher = better); `overexposed_pct` = % pixels with max-RGB>0.98; `color_cast` = Lab (a,b) offset from neutral. NIQE needs `pip install pyiqa`; LPIPS needs `pip install lpips`; missing metrics are omitted.