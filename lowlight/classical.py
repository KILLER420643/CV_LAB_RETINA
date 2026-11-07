"""Classical CV branch: analysis, illumination decomposition, Retinex, guided denoise, CLAHE, colour/contrast."""
import cv2, numpy as np

def gray(x): return cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2GRAY)
def box(x, r): return cv2.boxFilter(x, -1, (2*r+1, 2*r+1), normalize=True, borderType=cv2.BORDER_REFLECT)

def guided_filter(I, p, r, eps):
    """He et al. guided filter (edge-preserving, halo-free)."""
    mI, mp = box(I, r), box(p, r)
    a = (box(I*p, r) - mI*mp) / (box(I*I, r) - mI*mI + eps)
    b = mp - a*mI
    return box(a, r)*I + box(b, r)

def noise_sigma(g):
    """Immerkaer fast noise-sigma estimate."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = g.shape
    return float(np.abs(cv2.filter2D(g, -1, k)[1:-1, 1:-1]).sum() * np.sqrt(0.5*np.pi) / (6*(h-2)*(w-2)))

def analyze(img):
    """Image characteristics that drive every adaptive decision."""
    g = gray(img); mean = float(g.mean()); p1, p99 = np.percentile(g, [1, 99])
    sig = noise_sigma(g)
    blur = cv2.GaussianBlur(g, (0, 0), max(g.shape)/20)
    m = img.reshape(-1, 3).mean(0)
    return dict(mean_luminance=mean, dark_ratio=float((g < 0.1).mean()), dynamic_range=float(p99-p1),
                noise_sigma=sig, noise_level=float(np.clip(sig/max(mean, .02)*.45/.10, 0, 1)),
                nonuniformity=float(np.clip(blur.std()/(blur.mean()+1e-3)/1.5, 0, 1)),
                severity=float(np.clip((.40-mean)/.35, 0, 1)),
                color_cast=float(m.std()/(m.mean()+1e-3)))

FIXED = dict(gamma=.5, clip=2.5, tile=8, dn_eps=1e-3, dn_r=4, clahe_mix=.5, cc=.5, sat=1.1, post_dn=.5)

def adaptive_params(s, adaptive=True):
    if not adaptive: return dict(FIXED)
    sv, nl, nu = s['severity'], s['noise_level'], s['nonuniformity']
    return dict(gamma=.85-.45*sv, clip=float(np.clip(1.5+2*sv-1.5*nl, 1, 3.5)), tile=8 if nu < .5 else 12,
                dn_eps=float(np.clip((2.5*s['noise_sigma'])**2, 1e-5, 1e-2)), dn_r=int(3+4*nl),
                clahe_mix=.3+.4*(1-nl), cc=float(np.clip(s['color_cast']*3, .2, .9)), sat=1+.15*sv, post_dn=.3+.6*nl)

def denoise(x, p):
    g = gray(x)
    return np.clip(np.stack([guided_filter(g, x[..., c], p['dn_r'], p['dn_eps']) for c in range(3)], -1), 0, 1)

def illumination(x):
    """Illumination map: max-RGB prior refined with a guided filter (edge-aware => no halos)."""
    h, w = x.shape[:2]
    L = guided_filter(gray(x), x.max(2), max(8, min(h, w)//30), 1e-3)
    return np.clip(np.maximum(L, 0.6*x.max(2)), .01, 1)

def clahe_lab(x, clip, tile, mix):
    lab = cv2.cvtColor((np.clip(x, 0, 1)*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    lab[..., 0] = (mix*l2 + (1-mix)*lab[..., 0]).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)/255

def color_fix(x, p):
    """Partial gray-world white balance + saturation recovery."""
    m = x.reshape(-1, 3).mean(0) + 1e-4
    gain = np.clip(1 + p['cc']*(m.mean()/m - 1), .7, 1.4)
    hsv = cv2.cvtColor(np.clip(x*gain, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1]*p['sat'], 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def overexposure_guard(x, knee=.85):
    """Soft shoulder: compresses highlights instead of clipping."""
    return np.where(x > knee, knee + (1-knee)*np.tanh((x-knee)/(1-knee)), x).astype(np.float32)

def enhance(img, p, denoise_on=True, retinex=True, clahe=True, color=True, guard=True):
    inter, x = {}, img.copy()
    if denoise_on: x = denoise(x, p); inter['denoised'] = x
    L = illumination(x); inter['illumination'] = L
    if retinex:
        R = np.clip(x/L[..., None], 0, 1.5); inter['reflectance'] = np.clip(R, 0, 1)
        Le = L**p['gamma']; y = np.clip(R*Le[..., None], 0, 1)
        gain = Le/L   # noise amplification map -> extra smoothing where gain is large
        if denoise_on:
            m = (p['post_dn']*np.clip((gain-1)/4, 0, 1))[..., None]
            y = m*denoise(y, p) + (1-m)*y
        inter['retinex'] = y
    else: y = x
    if clahe: y = clahe_lab(y, p['clip'], p['tile'], p['clahe_mix']); inter['clahe'] = y
    if color: y = color_fix(y, p)
    if guard: y = overexposure_guard(y)
    return np.clip(y, 0, 1).astype(np.float32), inter, L
