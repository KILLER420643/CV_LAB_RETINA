"""Create synthetic low/high pairs (gamma + gain + non-uniform vignette + Poisson-Gaussian noise) from bundled skimage images."""
import os, numpy as np, cv2
from skimage import data
rng = np.random.default_rng(0)
def synth(hi, level):
    h, w = hi.shape[:2]; yy, xx = np.mgrid[:h, :w]
    vig = np.exp(-(((xx-w*.3)/(w*.7))**2 + ((yy-h*.4)/(h*.7))**2))[..., None]*.8+.2   # non-uniform lighting
    lo = (hi**rng.uniform(1.8, 2.6))*level*vig*np.array([1, .95, .8])                # dark + colour cast
    lo = rng.poisson(lo*400)/400 + rng.normal(0, .004, lo.shape)                      # shot + read noise
    return np.clip(lo, 0, 1).astype(np.float32)
def pairs(n_levels=(.15, .35)):
    for name in ['astronaut', 'coffee', 'chelsea', 'rocket', 'cat']:
        try: im = getattr(data, name)()
        except Exception: continue
        hi = cv2.resize(im, (480, 360)).astype(np.float32)/255
        for lv in n_levels: yield f'{name}_{lv}', synth(hi, lv), hi
if __name__ == '__main__':
    for d in ('samples/low', 'samples/high'): os.makedirs(d, exist_ok=True)
    for n, lo, hi in pairs():
        cv2.imwrite(f'samples/low/{n}.png', cv2.cvtColor((lo*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'samples/high/{n}.png', cv2.cvtColor((hi*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print('samples written')
