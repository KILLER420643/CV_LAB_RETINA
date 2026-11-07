"""Full-reference, no-reference and artifact metrics."""
import cv2, numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr, structural_similarity as _ssim
from .classical import gray, noise_sigma
_lp = {}
def lpips_score(a, b):
    try:
        import lpips, torch
        if 'm' not in _lp: _lp['m'] = lpips.LPIPS(net='alex', verbose=False).eval()
        t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None]*2-1
        with torch.no_grad(): return float(_lp['m'](t(a), t(b)))
    except Exception: return None
def niqe_score(a):
    try:
        import pyiqa, torch
        if 'n' not in _lp: _lp['n'] = pyiqa.create_metric('niqe')
        return float(_lp['n'](torch.from_numpy(a).permute(2, 0, 1)[None]))
    except Exception: return None
def _grad(g): return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
def edge_preservation(out, ref):
    """Pearson correlation of gradient magnitudes (1 = edges fully preserved)."""
    a, b = _grad(gray(out)).ravel(), _grad(gray(ref)).ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.
def lab(x): return cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
def evaluate(out, inp, ref=None, with_lpips=True, with_niqe=True):
    g = gray(out); m = lab(out).reshape(-1, 3).mean(0)
    r = dict(noise_sigma=noise_sigma(g), noise_rel=noise_sigma(g)/(g.mean()+1e-3),
             edge_pres_vs_input=edge_preservation(out, inp),
             overexposed_pct=float((out.max(2) > .98).mean()*100),
             color_cast=float(np.hypot(m[1], m[2])), mean_luma=float(g.mean()))
    if with_niqe: r['niqe'] = niqe_score(out)
    if ref is not None:
        r.update(psnr=float(_psnr(ref, out, data_range=1)), ssim=float(_ssim(ref, out, channel_axis=2, data_range=1)),
                 delta_e=float(np.sqrt(((lab(out)-lab(ref))**2).sum(2)).mean()), edge_pres_vs_gt=edge_preservation(out, ref))
        if with_lpips: r['lpips'] = lpips_score(out, ref)
    return r
