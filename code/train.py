"""Train one condition/seed on cached stage-6 EfficientNet-B0 features.
Trainable part: MBConv block 15 (stage 7) + 1x1 conv head (both ImageNet-initialised) + task head.
Conditions: B1 (GAP + CE), P (lesion-query attention + focal + BCE + diversity), A2 (P with plain CE), A3 (P at 448 px),
            GATE (GAP + 3-class EyeQ quality).
Usage: python train.py COND SEED
"""
import sys, os, json, copy, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import cohen_kappa_score, roc_auc_score
torch.set_num_threads(int(os.environ.get('NT', 2)))
COND, SEED = sys.argv[1], int(sys.argv[2])
RES = 448 if COND == 'A3' else 224
EPOCHS = int(os.environ.get('EPOCHS', 12)); BS = 64
FE = '/home/claude/feats'; OUT = f'/home/claude/runs/{COND}_s{SEED}'; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
meta = pd.read_csv('/home/claude/prep/meta.csv', dtype={'id': str})

def load(ds):
    ids = open(f'{FE}/{ds}_ids.txt').read().split('\n')
    return ids, np.load(f'{FE}/{RES}_{ds}.npy', mmap_mode='r')

ep_ids, ep_X = load('eyepacs')
em = meta[meta.ds == 'eyepacs'].set_index('id').loc[ep_ids]
idx = {s: np.where(em.split.values == s)[0] for s in ['train', 'val', 'test']}
test_all = idx['test'].copy()
if COND == 'GATE':
    q = em.eyeq.values
    for s in idx: idx[s] = idx[s][~np.isnan(q[idx[s]])]
    Y = np.nan_to_num(q).astype(int)
else:
    Y = em.grade.values.astype(int)
Xtr = torch.from_numpy(np.asarray(ep_X[idx['train']], dtype=np.float32))
Ytr = torch.from_numpy(Y[idx['train']])
K = 3 if COND == 'GATE' else 5

class Net(nn.Module):
    def __init__(s, head):
        super().__init__()
        b = EfficientNet.from_name('efficientnet-b0')
        b.load_state_dict(torch.load('/home/claude/weights/efficientnet-b0-355c32eb.pth', map_location='cpu'))
        s.blk, s.conv, s.bn, s.act = b._blocks[15], b._conv_head, b._bn1, b._swish
        s.head = head; d = 1280
        if head == 'gap':
            s.drop = nn.Dropout(0.2); s.fc = nn.Linear(d, K)
        else:
            s.Q, dk = 8, 256; s.dk = dk
            s.ln = nn.LayerNorm(d); s.Wk = nn.Linear(d, dk); s.Wv = nn.Linear(d, dk)
            s.P = nn.Parameter(torch.randn(s.Q, dk) * 0.02)
            s.ln2 = nn.LayerNorm(dk); s.fc = nn.Linear(dk, K); s.fc_any = nn.Linear(dk, 1)
    def feats(s, x):
        return s.act(s.bn(s.conv(s.blk(x))))           # (B,1280,g,g)
    def forward(s, x, ret_maps=False):
        f = s.feats(x)
        if s.head == 'gap':
            return dict(logits=s.fc(s.drop(f.mean((2, 3)))), fmap=f)
        B, C, H, W = f.shape
        X = s.ln(f.flatten(2).transpose(1, 2))           # (B,N,d)
        Kx, V = s.Wk(X), s.Wv(X)
        A = torch.softmax(torch.einsum('qk,bnk->bqn', s.P, Kx) / s.dk ** 0.5, dim=-1)   # Eq. (1)
        F = s.ln2(torch.einsum('bqn,bnk->bqk', A, V).mean(1))
        return dict(logits=s.fc(F), any=s.fc_any(F).squeeze(1), A=A.view(B, s.Q, H, W))

head = 'gap' if COND in ('B1', 'GATE') else 'lat'
net = Net(head)
freq = np.bincount(Y[idx['train']], minlength=K).astype(float)
alpha = torch.tensor((1 / freq) / (1 / freq).mean(), dtype=torch.float32)

def loss_fn(o, y):
    if COND in ('B1', 'GATE'):
        return Fn.cross_entropy(o['logits'], y)
    logp = Fn.log_softmax(o['logits'], 1)
    if COND == 'A2':
        Lg = Fn.nll_loss(logp, y)
    else:   # focal loss, Eq. (3), gamma = 2, class weights inverse to frequency
        lp = logp.gather(1, y[:, None]).squeeze(1); p = lp.exp()
        Lg = (-alpha[y] * (1 - p) ** 2 * lp).mean()
    Lb = Fn.binary_cross_entropy_with_logits(o['any'], (y > 0).float())
    A = o['A'].flatten(2); An = Fn.normalize(A, dim=-1)
    S = torch.einsum('bqn,bpn->bqp', An, An); Q = A.shape[1]
    Ld = (S.sum((1, 2)) - S.diagonal(dim1=1, dim2=2).sum(1)).mean() / (Q * (Q - 1))   # Eq. (4)
    return Lg + 1.0 * Lb + 0.1 * Ld

bb = [p for n, p in net.named_parameters() if n.split('.')[0] in ('blk', 'conv', 'bn')]
hd = [p for n, p in net.named_parameters() if n.split('.')[0] not in ('blk', 'conv', 'bn')]
opt = torch.optim.AdamW([{'params': bb, 'lr': 3e-4}, {'params': hd, 'lr': 1e-3}], weight_decay=0.05)
steps = EPOCHS * int(np.ceil(len(Ytr) / BS))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-4, 1e-3], total_steps=steps, pct_start=0.15)

@torch.no_grad()
def predict(X, ids=None, maps=False):
    net.eval(); P, S, M = [], [], []
    for i in range(0, len(X), 128):
        o = net(torch.from_numpy(np.asarray(X[i:i + 128], dtype=np.float32)))
        pr = torch.softmax(o['logits'], 1); P.append(pr.numpy())
        S.append((torch.sigmoid(o['any']) if 'any' in o else 1 - pr[:, 0]).numpy())
        if maps and 'A' in o: M.append(o['A'].numpy().astype(np.float16))
    return np.concatenate(P), np.concatenate(S), (np.concatenate(M) if M else None)

def val_metric(P, y):
    if COND == 'GATE': return (P.argmax(1) == y).mean()
    return cohen_kappa_score(y, P.argmax(1), weights='quadratic')

Xva = ep_X[idx['val']]; yva = Y[idx['val']]
best, best_state, log = -9, None, []
t0 = time.time()
for ep in range(EPOCHS):
    net.train(); perm = torch.randperm(len(Ytr)); tl = 0
    for i in range(0, len(perm), BS):
        b = perm[i:i + BS]; x = Xtr[b]
        if np.random.rand() < 0.5: x = x.flip(3)            # feature-map flips as cheap augmentation
        if np.random.rand() < 0.5: x = x.flip(2)
        o = net(x); L = loss_fn(o, Ytr[b])
        opt.zero_grad(); L.backward(); opt.step(); sched.step(); tl += L.item() * len(b)
    Pv, Sv, _ = predict(Xva); m = val_metric(Pv, yva)
    log.append(dict(epoch=ep, loss=tl / len(Ytr), val=float(m), t=time.time() - t0)); print(log[-1], flush=True)
    if m > best: best, best_state = m, copy.deepcopy(net.state_dict())
net.load_state_dict(best_state)
torch.save(best_state, f'{OUT}/model.pt')
json.dump(log, open(f'{OUT}/log.json', 'w'))

# threshold: highest threshold with validation sensitivity >= 0.90 (any DR)
Pv, Sv, _ = predict(Xva)
res = {}
if COND != 'GATE':
    pos = np.sort(Sv[yva > 0])[::-1]; thr = float(pos[int(np.ceil(0.90 * len(pos))) - 1])
    res['thr'] = thr
np.savez(f'{OUT}/pred_val.npz', P=Pv, S=Sv, y=yva, ids=np.array(ep_ids)[idx['val']])
Pt, St, _ = predict(ep_X[test_all])
np.savez(f'{OUT}/pred_internal.npz', P=Pt, S=St, y=Y[test_all], ids=np.array(ep_ids)[test_all])
for ds in ['aptos', 'idrid', 'ddr']:
    ids, X = load(ds)
    P_, S_, M_ = predict(X, maps=(ds == 'ddr'))
    d = dict(P=P_, S=S_, ids=np.array(ids))
    if M_ is not None: d['A'] = M_
    if COND == 'B1' and ds == 'ddr':      # Grad-CAM on the conv-head activations for the any-DR score log(1-p0)
        cams = []
        net.eval()
        for i in range(0, len(X), 32):
            x = torch.from_numpy(np.asarray(X[i:i + 32], dtype=np.float32))
            f = net.feats(x); f.retain_grad()
            lg = net.fc(f.mean((2, 3))); sc = torch.log1p(-torch.softmax(lg, 1)[:, 0] + 1e-6).sum()
            net.zero_grad(); sc.backward()
            w = f.grad.mean((2, 3), keepdim=True); cams.append(torch.relu((w * f).sum(1)).detach().numpy().astype(np.float16))
        d['cam'] = np.concatenate(cams)
    np.savez(f'{OUT}/pred_{ds}.npz', **d)
res.update(best_val=float(best), time=time.time() - t0)
json.dump(res, open(f'{OUT}/res.json', 'w')); print('DONE', COND, SEED, res, flush=True)
