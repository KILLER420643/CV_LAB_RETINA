---
title: Adaptive Low-Light Restoration
sdk: gradio
app_file: app.py
python_version: "3.11"
---
# Adaptive Low-Light Image Restoration (P1)
Hybrid classical (illumination decomposition, Retinex, CLAHE, guided denoise, colour correction) + Zero-DCE, with adaptive parameters and quality-aware fusion. See `docs/DESIGN.md` (architecture diagram, design), `docs/FAILURE_ANALYSIS.md`.

## Run
```bash
pip install -r requirements.txt && python app.py          # http://localhost:7860
# or: docker build -t lowlight . && docker run -p 7860:7860 lowlight
```
## Reproduce results
```bash
# 1. data: LOL (our485 / eval15), LOL-v2 -> data/
python scripts/train.py --low data/LOL/our485/low --epochs 100          # unsupervised Zero-DCE
python scripts/train.py --low data/LOL/our485/low --high data/LOL/our485/high --init weights/zerodce.pth --epochs 50 --lr 5e-5   # paired fine-tune
# 2. evaluate on UNSEEN LOL-v2 (writes eval_P1_T01.md; first line = USNs)
python scripts/evaluate.py --low data/LOL-v2/Real_captured/Test/Low --high data/LOL-v2/Real_captured/Test/Normal --team T01 --usns USN1,USN2,USN3
pip install pyiqa   # optional: enables NIQE
pytest tests/       # smoke tests (modes, tiny/huge/black/white inputs)
```
`python scripts/make_samples.py` creates synthetic pairs (gamma + vignette + colour cast + Poisson-Gaussian noise) used as UI examples and for a smoke evaluation. The shipped `weights/zerodce.pth` is a bootstrap trained on these only.
## Deploy
Push to a Hugging Face Space (Gradio SDK) or Render/Railway with the Dockerfile. No secrets are used.

## Weights
`weights/zerodce.pth` = official Zero-DCE pretrained weights (Li-Chongyi/Zero-DCE, `Epoch99.pth`, key-renamed to this repo's `DCENet`). `weights/zerodce_synthetic_bootstrap.pth` = our small synthetic-data bootstrap. Fine-tune on LOL with `scripts/train.py --init weights/zerodce.pth` and cite the Zero-DCE authors in your report.
