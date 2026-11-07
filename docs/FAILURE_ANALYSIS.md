# Failure analysis
Fill in with *your* LOL-v2 results (attach input/output crops from `eval_P1_*_per_image.csv` worst-PSNR rows).

| # | Case | Observed symptom | Root cause | Mitigation in code | Remaining gap |
|---|---|---|---|---|---|
| 1 | Near-black frames (mean luma < 0.01) | Blotchy chroma noise | Information lost at capture; Retinex gain Le/L very large | gain-aware post-denoise, `post_dn` up to 0.9 | Cannot recover lost detail |
| 2 | Point light sources in dark scene | Faint halo / bloom near lamps | Illumination smoothing across a very bright, small region | guided filter (edge aware), soft-knee guard | Clipped lamp cores unrecoverable |
| 3 | Coloured light (neon) | Colour partially neutralised | Gray-world assumption violated | strength `cc` scales with measured cast and is capped at 0.9 | No scene-illuminant classifier |
| 4 | Fine texture at high noise | Over-smoothing | Guided denoise eps tied to noise sigma | noise-adaptive CLAHE/denoise | Texture-vs-noise ambiguity |
