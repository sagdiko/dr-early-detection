"""Evaluate all runs: metrics per seed, seed-averaged metrics with patient-level bootstrap CIs, DeLong and McNemar tests,
quality gate, pointing game. Writes results/*.json and results/*.csv."""
import glob, json, os, numpy as np, pandas as pd
from scipy import stats
from scipy.ndimage import binary_dilation, zoom
from sklearn.metrics import roc_auc_score, cohen_kappa_score, f1_score
R = '/home/claude/runs'; OUT = '/home/claude/results'; os.makedirs(OUT, exist_ok=True)
meta = pd.read_csv('/home/claude/prep/meta.csv', dtype={'id': str})
SEEDS = sorted({int(p.split('_s')[-1]) for p in glob.glob(f'{R}/P_s*') if os.path.exists(p + '/res.json') and os.path.exists(p.replace('/P_s', '/B1_s') + '/res.json')})
json.dump(SEEDS, open(f'{OUT}/seeds.json', 'w')); print('SEEDS', SEEDS)
CONDS = [c for c in ['B1', 'P', 'A2', 'A3'] if all(os.path.exists(f'{R}/{c}_s{s}/res.json') for s in SEEDS)]
NB = 1000; rng = np.random.RandomState(12345)

# ---------------- quality gate ----------------
gate = {}
for ds in ['internal', 'aptos', 'idrid', 'ddr']:
    g = np.load(f'{R}/GATE_s0/pred_{ds}.npz')
    gate[ds] = dict(ids=g['ids'], reject=g['P'].argmax(1) == 2)
gv = np.load(f'{R}/GATE_s0/pred_internal.npz')

def gt(ds, ids):
    if ds == 'internal': d = meta[meta.ds == 'eyepacs']
    else: d = meta[meta.ds == ds]
    d = d.set_index('id').loc[ids]
    return d.grade.values, d.pid.values

def load(cond, seed, ds):
    z = np.load(f'{R}/{cond}_s{seed}/pred_{ds}.npz', allow_pickle=True)
    thr = json.load(open(f'{R}/{cond}_s{seed}/res.json'))['thr']
    return z, thr

def metrics(y, S, P, thr):
    yb = (y > 0).astype(int); pred = (S >= thr).astype(int)
    out = {}
    if yb.min() != yb.max(): out['auc'] = roc_auc_score(yb, S)
    pos, neg = yb == 1, yb == 0
    out['se'] = pred[pos].mean() if pos.any() else np.nan
    out['sp'] = 1 - pred[neg].mean() if neg.any() else np.nan
    se, sp = out['se'], out['sp']; pi = 0.10
    out['ppv10'] = se * pi / (se * pi + (1 - sp) * (1 - pi)) if neg.any() else np.nan
    out['g1rec'] = pred[y == 1].mean() if (y == 1).any() else np.nan
    if P is not None and len(np.unique(y)) > 2:
        out['kw'] = cohen_kappa_score(y, P.argmax(1), weights='quadratic')
        out['mf1'] = f1_score(y, P.argmax(1), average='macro')
    return out

def delong(y, s1, s2):
    """Fast DeLong test for paired AUCs (Sun & Xu 2014)."""
    pos, neg = s1[y == 1], s1[y == 0]; m, n = len(pos), len(neg)
    def v(sp, sn):
        allv = np.concatenate([sp, sn]); r = stats.rankdata(allv)
        rp, rn = stats.rankdata(sp), stats.rankdata(sn)
        auc = (r[:m].sum() - m * (m + 1) / 2) / (m * n)
        v01 = (r[:m] - rp) / n; v10 = 1 - (r[m:] - rn) / m
        return auc, v01, v10
    a1, v01a, v10a = v(s1[y == 1], s1[y == 0]); a2, v01b, v10b = v(s2[y == 1], s2[y == 0])
    s01 = np.cov(np.vstack([v01a, v01b])); s10 = np.cov(np.vstack([v10a, v10b]))
    S = s01 / m + s10 / n; var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var); return a1 - a2, 2 * stats.norm.sf(abs(z))

def mcnemar(a, b):
    """exact McNemar on paired binary decisions"""
    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
    if n01 + n10 == 0: return 1.0
    return float(min(1, 2 * stats.binom.cdf(min(n01, n10), n01 + n10, 0.5)))

def boot_idx(groups):
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(ug))]
    for _ in range(NB):
        pick = rng.randint(0, len(ug), len(ug))
        yield np.concatenate([members[k] for k in pick])

def gated(ds, ids):
    rj = dict(zip(gate[ds]['ids'], gate[ds]['reject']))
    return np.array([not rj[i] for i in ids])

# ---------------- main loop ----------------
rows, summary, preds = [], {}, {}
for ds in ['internal', 'aptos', 'idrid', 'ddr']:
    for cond in CONDS + (['A1'] if 'P' in CONDS else []):
        src = 'P' if cond == 'A1' else cond
        per_seed, Ss, Ps, Ds = [], [], [], []
        for s in SEEDS:
            z, thr = load(src, s, ds)
            ids = z['ids'].astype(str)
            y, pid = (gt(ds, ids) if ds != 'ddr' else (np.full(len(ids), -1), ids))
            keep = np.ones(len(ids), bool) if cond in ('B1', 'A1') else gated(ds, ids)
            if ds != 'ddr':
                ex = meta[meta.ds == ('eyepacs' if ds == 'internal' else ds)].set_index('id').loc[ids].exclude.fillna('').values
                keep = keep & (ex == '')
            if ds == 'ddr':
                yy = np.ones(keep.sum(), int) * 2      # all DDR lesion images contain lesions (any DR)
                m = dict(se=float((z['S'][keep] >= thr).mean()))
            else:
                yy = y[keep]; m = metrics(yy, z['S'][keep], z['P'][keep], thr)
            m.update(cond=cond, seed=s, ds=ds, n=int(keep.sum())); per_seed.append(m)
            Ss.append(z['S']); Ps.append(z['P']); Ds.append((z['S'] >= thr).astype(int))
        rows += per_seed
        preds[(ds, cond)] = dict(ids=ids, y=y, pid=pid, keep=keep, S=np.array(Ss), P=np.array(Ps), D=np.array(Ds),
                                 thr=[load(src, s, ds)[1] for s in SEEDS])
pd.DataFrame(rows).to_csv(f'{OUT}/per_seed.csv', index=False)

def seedavg(ds, cond, idx=None):
    p = preds[(ds, cond)]; keep = p['keep']
    sel = np.where(keep)[0] if idx is None else idx
    ms = []
    for k in range(len(SEEDS)):
        if ds == 'ddr':
            ms.append(dict(se=p['D'][k][sel].mean()))
        else:
            ms.append(metrics(p['y'][sel], p['S'][k][sel], p['P'][k][sel], p['thr'][k]))
    return {key: float(np.nanmean([m.get(key, np.nan) for m in ms])) for key in ms[0]}

table = {}
for ds in ['internal', 'aptos', 'idrid', 'ddr']:
    for cond in CONDS + ['A1']:
        p = preds[(ds, cond)]; sel = np.where(p['keep'])[0]
        point = seedavg(ds, cond)
        bs = []
        for bi in boot_idx(p['pid'][sel]):
            bs.append(seedavg(ds, cond, sel[bi]))
        ci = {k: [float(np.nanpercentile([b[k] for b in bs], 2.5)), float(np.nanpercentile([b[k] for b in bs], 97.5))] for k in point}
        sd = pd.DataFrame([r for r in rows if r['cond'] == cond and r['ds'] == ds]).std(numeric_only=True).to_dict()
        table[f'{ds}|{cond}'] = dict(point=point, ci=ci, sd={k: float(v) for k, v in sd.items() if k in point}, n=int(len(sel)),
                                     prev=float((p['y'][sel] > 0).mean()) if ds != 'ddr' else 1.0)
        print(ds, cond, len(sel), {k: round(v, 3) for k, v in point.items()}, flush=True)
json.dump(table, open(f'{OUT}/table.json', 'w'), indent=1)

# ---------------- paired comparisons (on images graded by both) ----------------
comp = {}
pairs = [('P', 'B1'), ('A1', 'B1')]
for ds in ['internal', 'aptos', 'idrid']:
    for a, b in pairs:
        if (ds, a) not in preds or (ds, b) not in preds: continue
        pa, pb = preds[(ds, a)], preds[(ds, b)]
        common = np.where(pa['keep'] & pb['keep'])[0]
        y = (pa['y'][common] > 0).astype(int)
        sa, sb = pa['S'][:, common].mean(0), pb['S'][:, common].mean(0)
        dauc, p_delong = delong(y, sa, sb)
        da = (pa['D'][:, common].mean(0) > 0.5).astype(int); db = (pb['D'][:, common].mean(0) > 0.5).astype(int)
        p_se = mcnemar(da[y == 1], db[y == 1]); p_sp = mcnemar(da[y == 0], db[y == 0])
        diffs = {k: [] for k in ['auc', 'se', 'sp', 'g1rec', 'kw']}
        for bi in boot_idx(pa['pid'][common]):
            ma = seedavg(ds, a, common[bi]); mb = seedavg(ds, b, common[bi])
            for k in diffs: diffs[k].append(ma.get(k, np.nan) - mb.get(k, np.nan))
        ma = seedavg(ds, a, common); mb = seedavg(ds, b, common)
        comp[f'{ds}|{a}-{b}'] = dict(n=int(len(common)), dauc_ens=float(dauc), p_delong=float(p_delong), p_mcnemar_se=p_se, p_mcnemar_sp=p_sp,
                                     diff={k: float(ma.get(k, np.nan) - mb.get(k, np.nan)) for k in diffs},
                                     ci={k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in diffs.items()})
        print(ds, a, '-', b, json.dumps(comp[f'{ds}|{a}-{b}'])[:300], flush=True)
json.dump(comp, open(f'{OUT}/comparisons.json', 'w'), indent=1)

# ---------------- quality gate summary ----------------
gs = {}
q = meta[meta.ds == 'eyepacs'].set_index('id').loc[gv['ids'].astype(str)].eyeq.values
ok = ~np.isnan(q)
gs['internal_eyeq_acc'] = float((gv['P'][ok].argmax(1) == q[ok]).mean()); gs['internal_eyeq_n'] = int(ok.sum())
rj = q[ok] == 2; pr = gv['P'][ok].argmax(1) == 2
gs['reject_recall'] = float(pr[rj].mean()); gs['reject_precision'] = float(rj[pr].mean())
for ds in ['internal', 'aptos', 'idrid', 'ddr']: gs[f'rejrate_{ds}'] = float(gate[ds]['reject'].mean())
y_int, _ = gt('internal', gate['internal']['ids'].astype(str))
gs['rejrate_internal_by_grade'] = {int(g): float(gate['internal']['reject'][y_int == g].mean()) for g in range(5)}
json.dump(gs, open(f'{OUT}/gate.json', 'w'), indent=1); print('gate', gs)

# ---------------- pointing game on DDR ----------------
mk = np.load('/home/claude/prep/ddr_masks512.npz'); mids = list(mk['ids']); masks = mk['masks']
fov = {}
from PIL import Image
r = 512 * 16 / 224   # one 16-px patch at 224 px, expressed in the 512-px frame
yy, xx = np.mgrid[-int(r):int(r) + 1, -int(r):int(r) + 1]; disk = (yy ** 2 + xx ** 2) <= r ** 2

def point(m):
    up = zoom(m.astype(np.float32), 512 / m.shape[0], order=1)
    return np.unravel_index(up.argmax(), up.shape)

ddr_ids = preds[('ddr', 'P')]['ids']
order = [mids.index(i) for i in ddr_ids]
dil_all, dil_ma, has_ma, rand_all, rand_ma = [], [], [], [], []
for k in order:
    m = masks[k]; f = np.asarray(Image.open(f'/home/claude/prep/ddr/{mids[k]}.jpg').convert('L')) > 20
    from scipy.ndimage import distance_transform_edt as edt
    da = edt(~(m > 0)) <= r; dm = edt(~((m & 1) > 0)) <= r
    dil_all.append(da); dil_ma.append(dm); has_ma.append(bool((m & 1).any()))
    rand_all.append(da[f].mean()); rand_ma.append(dm[f].mean())
has_ma = np.array(has_ma); has_any = np.array([d.any() for d in dil_all])
pg = {'n_images': int(len(order)), 'n_with_lesion': int(has_any.sum()), 'n_with_ma': int(has_ma.sum()),
      'random_all': float(np.mean(np.array(rand_all)[has_any])), 'random_ma': float(np.mean(np.array(rand_ma)[has_ma]))}
hits = {}
for cond in [c for c in CONDS if c != 'B1'] + ['B1']:
    H_all, H_ma = [], []
    for si, s in enumerate(SEEDS):
        z = np.load(f'{R}/{cond}_s{s}/pred_ddr.npz')
        maps = z['cam'].astype(np.float32) if cond == 'B1' else z['A'].astype(np.float32).max(1)
        ha, hm = [], []
        for j in range(len(order)):
            py, px = point(maps[j]); ha.append(dil_all[j][py, px]); hm.append(dil_ma[j][py, px])
        H_all.append(ha); H_ma.append(hm)
    H_all, H_ma = np.array(H_all, float), np.array(H_ma, float)
    hits[cond] = (H_all, H_ma)
    a = H_all[:, has_any].mean(1); mm = H_ma[:, has_ma].mean(1)
    ba = [H_all[:, has_any][:, b].mean() for b in [rng.randint(0, has_any.sum(), has_any.sum()) for _ in range(NB)]]
    bm = [H_ma[:, has_ma][:, b].mean() for b in [rng.randint(0, has_ma.sum(), has_ma.sum()) for _ in range(NB)]]
    pg[cond] = dict(all=float(a.mean()), all_sd=float(a.std(ddof=1)), all_ci=[float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))],
                    ma=float(mm.mean()), ma_sd=float(mm.std(ddof=1)), ma_ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))])
for c in [c for c in hits if c != 'B1']:
    d_all = hits[c][0][:, has_any].mean(0) - hits['B1'][0][:, has_any].mean(0)
    d_ma = hits[c][1][:, has_ma].mean(0) - hits['B1'][1][:, has_ma].mean(0)
    bd = [d_all[rng.randint(0, len(d_all), len(d_all))].mean() for _ in range(NB)]
    bdm = [d_ma[rng.randint(0, len(d_ma), len(d_ma))].mean() for _ in range(NB)]
    pg[f'{c}-B1'] = dict(all=float(d_all.mean()), all_ci=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                         ma=float(d_ma.mean()), ma_ci=[float(np.percentile(bdm, 2.5)), float(np.percentile(bdm, 97.5))])
json.dump(pg, open(f'{OUT}/pointing.json', 'w'), indent=1); print('pointing', json.dumps(pg)[:800])
