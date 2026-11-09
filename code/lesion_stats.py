"""DDR lesion masks: component statistics at native resolution and masks mapped to the 512-px preprocessed frame."""
import glob, os, numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage
from multiprocessing import Pool
R = '/home/claude/data/ddr_dataset_lesion_segmentation_detection.tree/lesion_segmentation'
TYPES = ['MA', 'HE', 'EX', 'SE']

def one(p):
    split = p.split('/')[-3]; name = os.path.basename(p)[:-4]; iid = f'{split}_{name}'
    im = np.asarray(Image.open(p).convert('RGB')); H, W = im.shape[:2]
    m = im.max(2) > 20
    rows = np.where(m.mean(1) > 0.02)[0]; cols = np.where(m.mean(0) > 0.02)[0]
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    side = max(y1 - y0, x1 - x0)
    lab_dir = f'{R}/{split}/' + ('segmentation label' if split == 'valid' else 'label')
    comps, packed = [], np.zeros((512, 512), np.uint8)
    for k, t in enumerate(TYPES):
        f = f'{lab_dir}/{t}/{name}.tif'
        a = np.asarray(Image.open(f)) > 0 if os.path.exists(f) else np.zeros((H, W), bool)
        lab, n = ndimage.label(a, structure=np.ones((3, 3)))
        if n:
            areas = ndimage.sum(a, lab, index=np.arange(1, n + 1))
            for ar in areas: comps.append(dict(id=iid, type=t, area=float(ar)))
        c = a[y0:y1, x0:x1]; h, w = c.shape
        sq = np.zeros((side, side), bool); sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = c
        small = np.asarray(Image.fromarray(sq.astype(np.uint8) * 255).resize((512, 512), Image.BOX)) > 0
        packed |= (small.astype(np.uint8) << k)
    return iid, side, comps, packed

if __name__ == '__main__':
    files = sorted(glob.glob(f'{R}/*/image/*.jpg'))
    with Pool(2) as pool: out = pool.map(one, files, chunksize=8)
    ids = [o[0] for o in out]
    np.savez_compressed('/home/claude/prep/ddr_masks512.npz', ids=np.array(ids), masks=np.stack([o[3] for o in out]))
    pd.DataFrame([dict(id=o[0], side=o[1]) for o in out]).to_csv('/home/claude/prep/ddr_sides.csv', index=False)
    pd.DataFrame([c for o in out for c in o[2]]).to_csv('/home/claude/prep/ddr_components.csv', index=False)
    print('done', len(ids))
