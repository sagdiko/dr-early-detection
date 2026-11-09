"""Build metadata, near-duplicate analysis (pHash) and the patient-level 70/15/15 split of the EyePACS development subset."""
import numpy as np, pandas as pd, json
P = '/home/claude/prep'; D = '/home/claude/data'
T = int(__import__('sys').argv[1]) if len(__import__('sys').argv) > 1 else 10   # Hamming threshold (of 255 bits)
pr = pd.concat([pd.read_csv(f'{P}/props_{d}.csv', dtype={'id': str}) for d in ['eyepacs', 'aptos', 'idrid', 'ddr']])
pr = pr[pr.ok == 1].reset_index(drop=True)
pr['phash'] = pr.phash.apply(lambda v: np.uint64(int(float(v))) if pd.notna(v) else np.uint64(0))
ep = pd.read_csv(f'{D}/Eyepacs.jsu-ysj.tree/trainLabels.csv').rename(columns={'image': 'id', 'level': 'grade'})
ap = pd.read_csv(f'{D}/aptos2019-blindness-detection.Abdelrahmanrezk.tree/train.csv').rename(columns={'id_code': 'id', 'diagnosis': 'grade'})
g = f'{D}/Disease-Grading-IDRiD.tree/data/original/2. Groundtruths/'
it = pd.concat([pd.read_csv(g + 'a. IDRiD_Disease Grading_Training Labels.csv').iloc[:, :2].assign(pre='tr_'),
                pd.read_csv(g + 'b. IDRiD_Disease Grading_Testing Labels.csv').iloc[:, :2].assign(pre='te_')])
it.columns = ['id', 'grade', 'pre']; it['id'] = it.pre + it.id; it = it[['id', 'grade']]
eq = pd.read_csv(f'{D}/eyeq/Label_EyeQ_train.csv'); eq['id'] = eq.image.str.replace('.jpeg', '', regex=False)
lab = pd.concat([ep.assign(ds='eyepacs'), ap.assign(ds='aptos'), it.assign(ds='idrid')])
m = pr.merge(lab, on=['ds', 'id'], how='left')
m = m.merge(eq[['id', 'quality']].rename(columns={'quality': 'eyeq'}).assign(ds='eyepacs'), on=['ds', 'id'], how='left')
m['pid'] = np.where(m.ds == 'eyepacs', 'ep' + m.id.str.split('_').str[0], m.ds + '_' + m.id)

# ---- pairwise distances of the 256-bit vessel-pattern hash (hash2.py) ----
z = np.load(f'{P}/hash2.npz'); key = {(a, b): k for k, (a, b) in enumerate(zip(z['ds'], z['ids']))}
bits = np.unpackbits(z['h'], axis=1)[:, :255]
B = np.packbits(np.pad(bits, ((0, 0), (0, 1))), axis=1).view('>u8').astype(np.uint64)
order = np.array([key[(a, b)] for a, b in zip(m.ds, m.id)]); B = B[order]
pairs = []
for i in range(0, len(B), 1000):
    d = sum(np.bitwise_count(B[i:i + 1000, k][:, None] ^ B[None, :, k]) for k in range(4)).astype(np.int32)
    r, c = np.where(d <= 20); k = (r + i) < c
    pairs.append(np.stack([r[k] + i, c[k], d[r[k], c[k]]], 1))
pairs = np.concatenate(pairs)
pdf = pd.DataFrame(pairs, columns=['i', 'j', 'd'])
pdf['ds_i'] = m.ds.values[pdf.i]; pdf['ds_j'] = m.ds.values[pdf.j]
pdf['g_i'] = m.grade.values[pdf.i]; pdf['g_j'] = m.grade.values[pdf.j]
pdf['same_pid'] = m.pid.values[pdf.i] == m.pid.values[pdf.j]
pdf['id_i'] = m.id.values[pdf.i]; pdf['id_j'] = m.id.values[pdf.j]
pdf.to_csv(f'{P}/dup_pairs.csv', index=False)
# external sets: duplicate groups -> keep one image if all grades agree, drop the whole group otherwise
m['exclude'] = ''
dupx = pdf[(pdf.d <= T) & (pdf.ds_i == pdf.ds_j) & (pdf.ds_i != 'eyepacs')]
par2 = {}
def f2(x):
    par2.setdefault(x, x)
    while par2[x] != x: par2[x] = par2[par2[x]]; x = par2[x]
    return x
for a, b in zip(dupx.i, dupx.j): par2[f2(a)] = f2(b)
groups = {}
for x in list(par2): groups.setdefault(f2(x), []).append(x)
for gg in groups.values():
    gr = m.grade.values[gg]
    if len(set(gr)) > 1: m.loc[m.index[gg], 'exclude'] = 'duplicate, conflicting grades'
    else: m.loc[m.index[sorted(gg)[1:]], 'exclude'] = 'duplicate copy'
# ---- union-find: patients + near-duplicates (d <= T) inside EyePACS ----
idx = np.where(m.ds.values == 'eyepacs')[0]
par = {i: i for i in idx}
def f(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
def u(a, b): par[f(a)] = f(b)
first = {}
for i in idx:
    p = m.pid.values[i]
    if p in first: u(i, first[p])
    else: first[p] = i
dup = pdf[(pdf.ds_i == 'eyepacs') & (pdf.ds_j == 'eyepacs') & (pdf.d <= T)]
for a, b in zip(dup.i, dup.j): u(a, b)
grp = np.array([f(i) for i in idx]); ug = np.unique(grp)
rng = np.random.RandomState(0); rng.shuffle(ug)
n = len(ug); cut = {gg: ('train' if k < 0.70 * n else 'val' if k < 0.85 * n else 'test') for k, gg in enumerate(ug)}
m['split'] = ''
m.loc[m.index[idx], 'split'] = [cut[gg] for gg in grp]
m.loc[m.ds != 'eyepacs', 'split'] = 'external'
m.drop(columns=['phash']).to_csv(f'{P}/meta.csv', index=False)
s = dict(T=T, n_groups=int(n), merged_dup_pairs_eyepacs=int((~dup.same_pid).sum()),
         split_counts=m[m.ds == 'eyepacs'].split.value_counts().to_dict())
print(json.dumps(s, indent=1))
for (a, b), gdf in pdf[pdf.d <= T].groupby(['ds_i', 'ds_j']):
    print(a, b, 'pairs', len(gdf), 'same_pid', int(gdf.same_pid.sum()), 'grade_conflict', int((gdf.g_i != gdf.g_j).sum()))
print('exclusions', m.groupby(['ds', 'exclude']).size().to_dict())
cross = pdf[(pdf.d <= T) & (pdf.ds_i != pdf.ds_j)]; print('cross-dataset duplicates', len(cross))
print('idrid split-crossing pairs', int(((pdf.d <= T) & (pdf.ds_i == 'idrid') & (pdf.id_i.str[:2] != pdf.id_j.str[:2])).sum()))
