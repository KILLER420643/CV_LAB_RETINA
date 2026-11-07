"""Train / fine-tune Zero-DCE.
  python scripts/train.py --low data/LOL/our485/low                    # unsupervised (Zero-DCE losses)
  python scripts/train.py --low .../low --high .../high --init weights/zerodce.pth   # + paired L1 fine-tune
  python scripts/train.py                                              # bootstrap on synthetic pairs"""
import argparse, glob, os, sys, random, cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lowlight.zerodce import DCENet, total_loss
from scripts.make_samples import pairs
ap = argparse.ArgumentParser(); ap.add_argument('--low'); ap.add_argument('--high'); ap.add_argument('--init')
ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--steps', type=int, default=40); ap.add_argument('--bs', type=int, default=8)
ap.add_argument('--crop', type=int, default=128); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--out', default='weights/zerodce.pth')
a = ap.parse_args(); random.seed(0); torch.manual_seed(0)
rd = lambda f: cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32)/255
if a.low:
    fs = sorted(glob.glob(a.low+'/*')); data = [(rd(f), rd(os.path.join(a.high, os.path.basename(f))) if a.high else None) for f in fs]
    n = int(len(data)*.9); train, val = data[:n], data[n:]     # 90/10 train/validation split
else:
    d = [(l, h) for _, l, h in pairs((.1, .2, .35, .5))]; train, val = d[:-2], d[-2:]
def batch():
    X, Y = [], []
    for _ in range(a.bs):
        lo, hi = random.choice(train); H, W = lo.shape[:2]; c = a.crop; y, x = random.randint(0, H-c), random.randint(0, W-c)
        l = lo[y:y+c, x:x+c]; h = hi[y:y+c, x:x+c] if hi is not None else None
        if random.random() < .5: l, h = l[:, ::-1], (h[:, ::-1] if h is not None else None)
        X.append(l.copy()); Y.append(h.copy() if h is not None else None)
    t = lambda z: torch.from_numpy(np.stack(z)).permute(0, 3, 1, 2)
    return t(X), (t(Y) if Y[0] is not None else None)
net = DCENet()
if a.init and os.path.exists(a.init): net.load_state_dict(torch.load(a.init))
opt = torch.optim.Adam(net.parameters(), a.lr, weight_decay=1e-4)
for ep in range(a.epochs):
    for _ in range(a.steps):
        x, y = batch(); e, A = net(x); loss = total_loss(x, e, A, y); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
    with torch.no_grad():
        vl = np.mean([float(total_loss(*(lambda t: (t, *net(t)))(torch.from_numpy(l).permute(2, 0, 1)[None]), None)) for l, _ in val])
    print(f'epoch {ep+1}/{a.epochs} train_loss {loss.item():.4f} val_loss {vl:.4f}', flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True); torch.save(net.state_dict(), a.out); print('saved', a.out)
