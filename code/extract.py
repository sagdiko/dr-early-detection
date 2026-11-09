"""Extract frozen EfficientNet-B0 (ImageNet) features up to stage 6 (block 14) for every preprocessed image.
Output: feats/{res}_{ds}.npy  (N,192,h,w) float16, and feats/{ds}_ids.txt
Usage: python extract.py RES DS [DS ...]
"""
import sys, os, numpy as np, pandas as pd, torch, time
from PIL import Image
from efficientnet_pytorch import EfficientNet
torch.set_num_threads(int(os.environ.get('NT', 2))); torch.set_grad_enabled(False)
RES = int(sys.argv[1]); DSS = sys.argv[2:]
P = '/home/claude/prep'; O = '/home/claude/feats'; os.makedirs(O, exist_ok=True)
m = EfficientNet.from_name('efficientnet-b0')
m.load_state_dict(torch.load('/home/claude/weights/efficientnet-b0-355c32eb.pth', map_location='cpu')); m.eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def front(x):
    h = m._swish(m._bn0(m._conv_stem(x)))
    for b in m._blocks[:15]:
        h = b(h)
    return h

for ds in DSS:
    props = pd.read_csv(f'{P}/props_{ds}.csv'); props = props[props.ok == 1]
    ids = props.id.astype(str).tolist()
    with open(f'{O}/{ds}_ids.txt', 'w') as f: f.write('\n'.join(ids))
    g = RES // 32
    out = np.lib.format.open_memmap(f'{O}/{RES}_{ds}.npy', mode='w+', dtype=np.float16, shape=(len(ids), 192, g, g))
    t = time.time(); B = 32
    for i in range(0, len(ids), B):
        ims = [np.asarray(Image.open(f'{P}/{ds}/{k}.jpg').resize((RES, RES), Image.BILINEAR), dtype=np.float32) / 255. for k in ids[i:i + B]]
        x = (torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2) - mean) / std
        out[i:i + len(ims)] = front(x).numpy().astype(np.float16)
        if (i // B) % 50 == 0: print(ds, RES, i, len(ids), f'{time.time() - t:.0f}s', flush=True)
    out.flush(); del out
    print('done', ds, RES, f'{time.time() - t:.0f}s', flush=True)
