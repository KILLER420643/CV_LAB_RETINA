"""Gradio UI: adaptive low-light restoration with intermediate visualisations, 3-way comparison, ablation, failure notes."""
import cv2, numpy as np, pandas as pd, gradio as gr
from lowlight.pipeline import run, ABLATIONS, get_dl
from lowlight.metrics import evaluate
MAXS = 1024
def load(img):
    if img is None: raise gr.Error('Please upload an image first.')
    x = np.asarray(img)
    if x.ndim == 2: x = np.stack([x]*3, -1)
    x = x[..., :3]
    if x.dtype != np.uint8: x = np.clip(x, 0, 255).astype(np.uint8)
    return x.astype(np.float32)/255
def u8(x): return (np.clip(x, 0, 1)*255).astype(np.uint8)
def heat(m): return cv2.cvtColor(cv2.applyColorMap(u8(m), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
def hist(x):
    g = cv2.cvtColor(u8(x), cv2.COLOR_RGB2GRAY); h = cv2.calcHist([g], [0], None, [64], [0, 256]).ravel(); h = h/(h.max()+1e-9)
    im = np.full((120, 256, 3), 245, np.uint8)
    for i, v in enumerate(h): cv2.rectangle(im, (i*4, 119), (i*4+3, int(119-v*110)), (60, 90, 200), -1)
    return im
def guard_dl():
    if not get_dl().ok: gr.Warning('weights/zerodce.pth not found - deep branch is untrained. Run scripts/train.py.')

def enhance(img, mode, adaptive):
    x = load(img); guard_dl()
    try: r = run(x, mode=mode.lower(), adaptive=adaptive, max_side=MAXS)
    except Exception as e: raise gr.Error(f'Processing failed: {e}')
    g = [(u8(r['input']), 'input'), (heat(r['illumination']), 'illumination map')]
    for k, v in r['intermediates'].items(): g.append((heat(v) if v.ndim == 2 else u8(v), k))
    g += [(hist(r['input']), 'input histogram'), (hist(r['output']), 'output histogram')]
    info = {'image_analysis': {k: round(v, 4) for k, v in r['stats'].items()}, 'adaptive_params': {k: round(float(v), 3) for k, v in r['params'].items()},
            'fusion': {k: round(v, 3) for k, v in r['fusion'].items()}, 'latency_ms': round(r['latency_ms'], 1)}
    return [(u8(r['input']), 'before'), (u8(r['output']), 'after')], g, info

def compare(img, gt):
    x = load(img); ref = None; guard_dl()
    out, rows = [], []
    for n, m in (('Classical', 'classical'), ('Deep learning', 'dl'), ('Hybrid', 'hybrid')):
        r = run(x, mode=m, max_side=MAXS)
        if gt is not None and ref is None:
            ref = cv2.resize(load(gt), r['output'].shape[1::-1], interpolation=cv2.INTER_AREA)
        e = evaluate(r['output'], r['input'], ref, with_niqe=False)
        rows.append({'method': n, 'latency_ms': round(r['latency_ms'], 1), **{k: round(v, 4) for k, v in e.items() if v is not None}}); out.append((u8(r['output']), n))
    return [(u8(r['input']), 'input')] + out, pd.DataFrame(rows)

def ablate(img, gt):
    x = load(img); ref = None; rows = []; guard_dl()
    for n, kw in ABLATIONS.items():
        r = run(x, mode='hybrid', max_side=640, **kw)
        if gt is not None and ref is None: ref = cv2.resize(load(gt), r['output'].shape[1::-1], interpolation=cv2.INTER_AREA)
        e = evaluate(r['output'], r['input'], ref, with_lpips=False, with_niqe=False)
        rows.append({'variant': n, 'latency_ms': round(r['latency_ms'], 1), **{k: round(v, 4) for k, v in e.items() if k in ('psnr', 'ssim', 'delta_e', 'noise_rel', 'edge_pres_vs_input', 'overexposed_pct', 'color_cast')}})
    return pd.DataFrame(rows)

FAIL = """### Known failure cases (see docs/FAILURE_ANALYSIS.md)
1. **Near-black input (mean luma < 0.01):** information is destroyed by quantisation; Retinex gain explodes noise -> blotchy chroma noise.
2. **Bright point lights (street lamps) in dark scene:** the illumination map is guided-filter smoothed, so a faint halo may remain around very bright sources; the overexposure guard compresses but cannot recover clipped pixels.
3. **Strong coloured lighting (neon signs):** partial gray-world correction can neutralise intentional colour; strength is capped (`cc<=0.9`) and scales with measured cast.
4. **Heavy sensor noise + texture:** guided denoising trades fine texture for smoothness; edge-preservation score drops."""

with gr.Blocks(title='Adaptive Low-Light Restoration') as demo:
    gr.Markdown('# 🌙 Adaptive Low-Light Image Restoration\nClassical Retinex/CLAHE/guided filtering **+** Zero-DCE, fused by an image-quality-aware, illumination-aware adaptive mechanism.')
    with gr.Tab('Enhance'):
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label='Low-light image', type='numpy', sources=['upload', 'clipboard'])
                mode = gr.Radio(['Hybrid', 'Classical', 'DL'], value='Hybrid', label='Method')
                ad = gr.Checkbox(True, label='Adaptive parameters (off = fixed baseline)')
                b = gr.Button('Enhance', variant='primary')
                gr.Examples(['samples/low/astronaut_0.15.png', 'samples/low/coffee_0.35.png', 'samples/low/chelsea_0.15.png'], inp)
            with gr.Column(scale=2):
                cmp = gr.Gallery(label='Before / After', columns=2, height='auto')
        gal = gr.Gallery(label='Intermediate stages (illumination, denoised, reflectance, Retinex, CLAHE, deep output, fusion map, histograms)', columns=4, height='auto')
        js = gr.JSON(label='Image analysis, adaptive parameters, fusion weights, latency')
        b.click(enhance, [inp, mode, ad], [cmp, gal, js], api_name='enhance')
    with gr.Tab('Compare 3 approaches'):
        with gr.Row(): i2 = gr.Image(label='Low-light image', type='numpy'); g2 = gr.Image(label='Ground truth (optional - enables PSNR/SSIM/LPIPS/ΔE)', type='numpy')
        b2 = gr.Button('Run comparison', variant='primary'); o2 = gr.Gallery(columns=4, height='auto'); t2 = gr.Dataframe(label='Metrics')
        b2.click(compare, [i2, g2], [o2, t2])
    with gr.Tab('Ablation'):
        with gr.Row(): i3 = gr.Image(label='Low-light image', type='numpy'); g3 = gr.Image(label='Ground truth (optional)', type='numpy')
        b3 = gr.Button('Run ablation', variant='primary'); t3 = gr.Dataframe(label='Ablation (one component removed per row)')
        b3.click(ablate, [i3, g3], t3)
    with gr.Tab('Failure analysis'): gr.Markdown(FAIL)
if __name__ == '__main__':
    demo.queue(max_size=8).launch(server_name='0.0.0.0', server_port=7860, theme=gr.themes.Soft())
