"""Preprocess fundus images: FOV crop, pad to square, resize to 512; compute image properties and pHash."""
import sys, os, numpy as np, pandas as pd, glob
from PIL import Image
from scipy.fft import dct
from scipy import ndimage
from multiprocessing import Pool
OUT='/home/claude/prep'

def phash(gray):  # gray: PIL L image
    a=np.asarray(gray.resize((32,32),Image.LANCZOS),dtype=np.float64)
    d=dct(dct(a,axis=0,norm='ortho'),axis=1,norm='ortho')[:8,:8].flatten()
    med=np.median(d[1:]); bits=(d>med).astype(np.uint8)
    return int(''.join(map(str,bits)),2)

def proc(args):
    ds,iid,path=args
    try:
        im=Image.open(path); W0,H0=im.size
        if im.format=='JPEG': im.draft('RGB',(W0//2,H0//2) if W0>2000 else (W0,H0))
        if max(im.size)>1600:
            sc=1600/max(im.size); im=im.convert('RGB').resize((round(im.size[0]*sc),round(im.size[1]*sc)),Image.BILINEAR)
        im=im.convert('RGB')
    except Exception as e:
        return dict(ds=ds,id=iid,ok=0,err=str(e))
    W,H=W0,H0
    a=np.asarray(im)
    m=a.max(axis=2)>20
    rows=np.where(m.mean(1)>0.02)[0]; cols=np.where(m.mean(0)>0.02)[0]
    if len(rows)<10 or len(cols)<10:
        return dict(ds=ds,id=iid,ok=0,err='nofov',W=W,H=H)
    y0,y1,x0,x1=rows[0],rows[-1]+1,cols[0],cols[-1]+1
    c=im.crop((x0,y0,x1,y1)); w,h=c.size; s=max(w,h)
    sq=Image.new('RGB',(s,s)); sq.paste(c,((s-w)//2,(s-h)//2))
    r=sq.resize((512,512),Image.LANCZOS)
    os.makedirs(f'{OUT}/{ds}',exist_ok=True); r.save(f'{OUT}/{ds}/{iid}.jpg',quality=95)
    ra=np.asarray(r).astype(np.float32); g=ra.mean(2)
    fm=ra.max(2)>20
    lap=ndimage.laplace(g)
    er=ndimage.binary_erosion(fm,iterations=6)
    return dict(ds=ds,id=iid,ok=1,W=W,H=H,fov_frac=float(m.mean()),
        bright=float(g[fm].mean()),contrast=float(g[fm].std()),
        sharp=float(lap[er].var()),red=float(ra[...,0][fm].mean()),green=float(ra[...,1][fm].mean()),blue=float(ra[...,2][fm].mean()),
        phash=phash(Image.fromarray(g.astype(np.uint8))))

def jobs():
    D='/home/claude/data'; J=[]
    ep=pd.read_csv(f'{D}/Eyepacs.jsu-ysj.tree/trainLabels.csv')
    ep['pid']=ep.image.str.split('_').str[0]
    rng=np.random.RandomState(0); pids=np.array(sorted(ep.pid.unique())); sel=set(rng.choice(pids,6000,replace=False))
    ep=ep[ep.pid.isin(sel)]
    J+= [('eyepacs',i,f'{D}/Eyepacs.jsu-ysj.tree/resized_train/{i}.jpeg') for i in ep.image]
    ap=pd.read_csv(f'{D}/aptos2019-blindness-detection.Abdelrahmanrezk.tree/train.csv')
    files={os.path.basename(p).rsplit('.',1)[0]:p for p in glob.glob(f'{D}/aptos2019-blindness-detection.Abdelrahmanrezk.tree/train_*/*')}
    J+= [('aptos',i,files[i]) for i in ap.id_code if i in files]
    g=f'{D}/Disease-Grading-IDRiD.tree/data/original/1. Original Images/'
    J+= [('idrid',('tr_' if 'Training' in p else 'te_')+os.path.basename(p)[:-4],p) for p in sorted(glob.glob(g+'*/*.jpg'))]
    J+= [('ddr',p.split('/')[-3]+'_'+os.path.basename(p)[:-4],p) for p in sorted(glob.glob(f'{D}/ddr_dataset_lesion_segmentation_detection.tree/lesion_segmentation/*/image/*.jpg'))]
    return J,ap,files

if __name__=='__main__':
    J,ap,files=jobs()
    print('aptos labels',len(ap),'files',len(files),'matched',ap.id_code.isin(files).sum(),flush=True)
    order={'idrid':0,'ddr':1,'aptos':2,'eyepacs':3}
    J.sort(key=lambda j:order[j[0]])
    with Pool(2) as p:
        for ds in ['idrid','ddr','aptos','eyepacs']:
            JJ=[j for j in J if j[0]==ds]
            if os.path.exists(f'/home/claude/prep/props_{ds}.csv'): continue
            res=list(p.imap(proc,JJ,chunksize=4))
            pd.DataFrame(res).to_csv(f'/home/claude/prep/props_{ds}.csv',index=False); print(ds,len(res),flush=True)
    print('done')
