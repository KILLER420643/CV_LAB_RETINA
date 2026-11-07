"""Hybrid pipeline: classical illumination/Retinex + Zero-DCE, fused by image-quality-aware + illumination-aware weights."""
import time, cv2, numpy as np
from . import classical as C
from .zerodce import DLEnhancer
_dl = {}
def get_dl():
    if 'm' not in _dl: _dl['m'] = DLEnhancer()
    return _dl['m']

def quality_score(x):
    """No-reference quality proxy (higher=better): entropy, contrast, low noise, no saturation, sane exposure."""
    g = C.gray(x); h = np.histogram(g, 64, (0, 1))[0]/g.size; ent = -(h[h > 0]*np.log2(h[h > 0])).sum()/6
    return float(ent + .5*g.std() - .9*C.noise_sigma(g)/(g.mean()+.05) - 2*(x.max(2) > .98).mean() - abs(g.mean()-.45))

def run(img, mode='hybrid', adaptive=True, denoise=True, retinex=True, clahe=True, dl=True,
        spatial=True, quality_gate=True, color=True, guard=True, max_side=1280):
    """img: float32 RGB [0,1]. mode in {classical, dl, hybrid}. Flags enable ablation."""
    t0 = time.perf_counter(); h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side/max(h, w); img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    st = C.analyze(img); p = C.adaptive_params(st, adaptive); inter = {}; wts = {}
    if mode == 'classical':
        out, inter, L = C.enhance(img, p, denoise, retinex, clahe, color, guard)
    elif mode == 'dl':
        out = get_dl()(img); L = C.illumination(img)
    else:
        pre = C.denoise(img, p) if denoise else img                      # classical -> DL (stops DL amplifying noise)
        Cout, inter, L = C.enhance(img, p, denoise, retinex, clahe, False, False)
        if dl:
            D = get_dl()(pre); inter['deep (Zero-DCE)'] = D; gD = .5
            if quality_gate:                                              # image-quality-aware candidate weighting
                sc = np.array([quality_score(D), quality_score(Cout)]); e = np.exp(6*(sc-sc.max())); gD = float(e[0]/e.sum())
                wts = dict(score_dl=float(sc[0]), score_classical=float(sc[1]))
            Ln = (L/(L.max()+1e-6)).astype(np.float32)
            wm = np.clip(gD*(.4+1.2*Ln), 0, 1) if spatial else np.full_like(L, gD)  # dark areas lean Retinex, bright lean DL
            wts['global_dl_weight'] = gD; inter['fusion weight map (DL)'] = wm
            out = wm[..., None]*D + (1-wm[..., None])*Cout
        else: out = Cout
        if color: out = C.color_fix(out, p)
        if guard: out = C.overexposure_guard(out)
    out = np.clip(out, 0, 1).astype(np.float32)
    return dict(output=out, input=img, intermediates=inter, illumination=L, stats=st, params=p, fusion=wts,
                latency_ms=(time.perf_counter()-t0)*1e3)

ABLATIONS = {'Full hybrid': {}, 'No adaptive params (fixed)': dict(adaptive=False), 'No denoise': dict(denoise=False),
             'No Retinex': dict(retinex=False), 'No CLAHE': dict(clahe=False), 'No deep branch': dict(dl=False),
             'No spatial fusion': dict(spatial=False), 'No quality gate': dict(quality_gate=False),
             'No colour correction': dict(color=False), 'No overexposure guard': dict(guard=False)}
