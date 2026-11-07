"""Zero-DCE (Guo et al., CVPR'20) DCE-Net + non-reference losses; optional paired L1 fine-tuning."""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights', 'zerodce.pth')

class DCENet(nn.Module):
    def __init__(s, nf=32):
        super().__init__(); c = lambda i, o: nn.Conv2d(i, o, 3, 1, 1)
        s.c1, s.c2, s.c3, s.c4, s.c5, s.c6, s.c7 = c(3, nf), c(nf, nf), c(nf, nf), c(nf, nf), c(2*nf, nf), c(2*nf, nf), c(2*nf, 24)
    def forward(s, x):
        r = F.relu; x1 = r(s.c1(x)); x2 = r(s.c2(x1)); x3 = r(s.c3(x2)); x4 = r(s.c4(x3))
        x5 = r(s.c5(torch.cat([x3, x4], 1))); x6 = r(s.c6(torch.cat([x2, x5], 1)))
        A = torch.tanh(s.c7(torch.cat([x1, x6], 1))); e = x
        for a in torch.split(A, 3, 1): e = e + a*(e*e - e)   # LE-curve, 8 iterations
        return e, A

def l_spa(org, enh):
    o, e = F.avg_pool2d(org.mean(1, keepdim=True), 4), F.avg_pool2d(enh.mean(1, keepdim=True), 4)
    ks = [torch.tensor(k, dtype=torch.float32).view(1, 1, 3, 3) for k in
          ([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, -1], [0, 0, 0]], [[0, -1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, -1, 0]])]
    return sum(((F.conv2d(o, k, padding=1) - F.conv2d(e, k, padding=1))**2).mean() for k in [k.to(o.device) for k in ks])
def l_exp(enh, E=.6): return ((F.avg_pool2d(enh.mean(1, keepdim=True), 16) - E)**2).mean()
def l_col(enh):
    r, g, b = enh.mean((2, 3)).chunk(3, 1); return ((r-g)**2 + (r-b)**2 + (g-b)**2).mean()
def l_tv(A): return ((A[:, :, 1:] - A[:, :, :-1])**2).mean() + ((A[:, :, :, 1:] - A[:, :, :, :-1])**2).mean()
def total_loss(x, enh, A, gt=None):
    l = l_spa(x, enh) + 10*l_exp(enh) + 5*l_col(enh) + 200*l_tv(A)
    return l + (5*F.l1_loss(enh, gt) if gt is not None else 0)

class DLEnhancer:
    def __init__(s, path=WEIGHTS, max_side=1024):
        s.net, s.max_side, s.ok = DCENet().eval(), max_side, False
        if os.path.exists(path): s.net.load_state_dict(torch.load(path, map_location='cpu')); s.ok = True
    @torch.no_grad()
    def __call__(s, img):
        h, w = img.shape[:2]; sc = min(1, s.max_side/max(h, w))
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None]
        if sc < 1: x = F.interpolate(x, scale_factor=sc, mode='bilinear', align_corners=False)
        y, _ = s.net(x)
        if sc < 1: y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
        return y[0].permute(1, 2, 0).clamp(0, 1).numpy().astype(np.float32)
