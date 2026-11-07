"""Comparative + ablation evaluation -> eval_<ProbID>_<TeamID>.md (+ per-image CSV).
python scripts/evaluate.py --low data/LOL-v2/Real_captured/Test/Low --high .../Normal --team T01 --usns 1XX23CS001,1XX23CS002"""
import argparse, glob, os, sys, time, cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lowlight.pipeline import run, ABLATIONS
from lowlight.metrics import evaluate
ap = argparse.ArgumentParser(); ap.add_argument('--low', default='samples/low'); ap.add_argument('--high', default='samples/high')
ap.add_argument('--team', default='T01'); ap.add_argument('--usns', default='USN1,USN2'); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--max-side', type=int, default=600); ap.add_argument('--no-lpips', action='store_true'); ap.add_argument('--no-niqe', action='store_true')
a = ap.parse_args()
rd = lambda f: cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32)/255
files = sorted(glob.glob(a.low+'/*'))[:a.limit or None]
METHODS = {'Classical only': dict(mode='classical'), 'Deep learning only (Zero-DCE)': dict(mode='dl'), 'Hybrid (proposed)': dict(mode='hybrid')}
rows = []
def go(group, name, kw):
    for f in files:
        lo = rd(f); hf = os.path.join(a.high, os.path.basename(f)); hi = rd(hf) if os.path.exists(hf) else None
        r = run(lo, max_side=a.max_side, **kw)
        if hi is not None and hi.shape != r['output'].shape: hi = cv2.resize(hi, r['output'].shape[1::-1], interpolation=cv2.INTER_AREA)
        m = evaluate(r['output'], r['input'], hi, not a.no_lpips, not a.no_niqe); m.update(group=group, method=name, latency_ms=r['latency_ms'], image=os.path.basename(f)); rows.append(m)
        print(group, name, os.path.basename(f), flush=True)
run(rd(files[0]), max_side=a.max_side)  # warm-up (model load) so latency is fair
for n, kw in METHODS.items(): go('compare', n, kw)
for n, kw in ABLATIONS.items(): go('ablation', n, dict(mode='hybrid', **kw))
df = pd.DataFrame(rows); df.to_csv(f'eval_P1_{a.team}_per_image.csv', index=False)
cols = ['psnr', 'ssim', 'lpips', 'niqe', 'latency_ms', 'noise_rel', 'edge_pres_vs_gt', 'edge_pres_vs_input', 'overexposed_pct', 'color_cast', 'delta_e']
def table(g):
    t = df[df.group == g].groupby('method', sort=False)[[c for c in cols if c in df and df[c].notna().any()]].mean().round(4)
    return t.to_markdown() if hasattr(t, 'to_markdown') and _has_tab else t.to_string()
try: import tabulate; _has_tab = True
except ImportError: _has_tab = False
md = [a.usns, '', f'# Evaluation - P1 - {a.team}', f'Images evaluated: {len(files)} (low: `{a.low}`, reference: `{a.high}`), resized to max side {a.max_side}px.', '',
      '## 1. PSNR / SSIM / LPIPS / NIQE / latency / noise / edge preservation (mean over images)', table('compare'), '',
      '## 2. Ablation study (hybrid pipeline, one component removed at a time)', table('ablation'), '',
      'Metric notes: PSNR/SSIM/LPIPS/Delta-E need ground truth; `noise_rel` = Immerkaer sigma / mean luma (lower = better noise suppression); '
      '`edge_pres_*` = gradient-magnitude correlation (higher = better); `overexposed_pct` = % pixels with max-RGB>0.98; `color_cast` = Lab (a,b) offset from neutral. '
      'NIQE needs `pip install pyiqa`; LPIPS needs `pip install lpips`; missing metrics are omitted.']
open(f'eval_P1_{a.team}.md', 'w').write('\n'.join(md)); print('\n'.join(md))
