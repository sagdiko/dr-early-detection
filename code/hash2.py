"""Vessel-pattern perceptual hash (256 bit) on high-pass filtered green channel; used for near-duplicate search."""
import numpy as np, pandas as pd, os
from PIL import Image
from scipy.fft import dct
from scipy import ndimage
from multiprocessing import Pool
P='/home/claude/prep'
def h(args):
    ds,i=args
    g=np.asarray(Image.open(f'{P}/{ds}/{i}.jpg').resize((256,256),Image.BILINEAR))[...,1].astype(np.float32)
    f=ndimage.binary_erosion(g>8,iterations=8)
    hp=g-ndimage.gaussian_filter(g,6); hp[~f]=0
    a=np.asarray(Image.fromarray(hp).resize((64,64),Image.BILINEAR),dtype=np.float64)
    d=dct(dct(a,axis=0,norm='ortho'),axis=1,norm='ortho')[:16,:16].flatten()[1:]
    bits=(d>np.median(d)).astype(np.uint8)
    return np.packbits(bits)
if __name__=='__main__':
    J=[]
    for ds in ['eyepacs','aptos','idrid','ddr']:
        p=pd.read_csv(f'{P}/props_{ds}.csv',dtype={'id':str}); p=p[p.ok==1]
        J+=[(ds,i) for i in p.id]
    with Pool(2) as pool: H=pool.map(h,J,chunksize=64)
    np.savez(f'{P}/hash2.npz',ds=np.array([j[0] for j in J]),ids=np.array([j[1] for j in J]),h=np.stack(H))
    print('done',len(J))
