import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage
plt.rcParams.update({'font.size':8,'font.family':'serif'})
R='/home/claude/data/ddr_dataset_lesion_segmentation_detection.tree/lesion_segmentation'
c=pd.read_csv('/home/claude/prep/ddr_components.csv')
# DDR image with many microaneurysms of typical size
ma=c[c.type=='MA'].groupby('id').size().sort_values(ascending=False)
cand=ma[(ma>=15)&(ma<=40)].index.tolist()
iid=[i for i in cand if i.startswith('test_')][0]
split,name=iid.split('_',1)
im=np.asarray(Image.open(f'{R}/{split}/image/{name}.jpg').convert('RGB'))
lab=f'{R}/{split}/'+('segmentation label' if split=='valid' else 'label')
M={t:np.asarray(Image.open(f'{lab}/{t}/{name}.tif'))>0 for t in ['MA','HE','EX']}
# crop window centred on densest MA cluster
dens=ndimage.uniform_filter(M['MA'].astype(float),300)
cy,cx=np.unravel_index(dens.argmax(),dens.shape); h=150
y0,x0=max(cy-h,0),max(cx-h,0); crop=im[y0:y0+2*h,x0:x0+2*h]
side=max(im.shape[:2]); s224=224/side
small=np.asarray(Image.fromarray(crop).resize((max(1,round(2*h*s224)),)*2,Image.BOX).resize((2*h,2*h),Image.NEAREST))
ids={'(a) Grade 0\nno DR':'tr_IDRiD_141','(b) Grade 1\nmild NPDR':'tr_IDRiD_105','(c) Grade 2\nmoderate NPDR':'tr_IDRiD_003','(d) Grade 4\nPDR':'tr_IDRiD_008'}
fig,ax=plt.subplots(1,6,figsize=(7.2,1.75),gridspec_kw=dict(wspace=0.04))
for a,(t,i) in zip(ax[:4],ids.items()):
    a.imshow(Image.open(f'/home/claude/prep/idrid/{i}.jpg')); a.set_title(t,fontsize=6.5); a.axis('off')
ax[4].imshow(crop)
cols={'MA':'#ff2b2b','HE':'#2b8cff','EX':'#ffe14d'}
for t,col in cols.items():
    mm=M[t][y0:y0+2*h,x0:x0+2*h]
    if mm.any(): ax[4].contour(ndimage.binary_dilation(mm,iterations=2),levels=[0.5],colors=col,linewidths=0.6)
ax[4].set_title('(e) DDR crop, native,\nMA outlined',fontsize=6.5); ax[4].axis('off')
ax[5].imshow(small); ax[5].set_title('(f) Same crop as\nseen at 224 px input',fontsize=6.5); ax[5].axis('off')
fig.savefig('/home/claude/figs/fig1_examples.pdf',bbox_inches='tight',dpi=300); fig.savefig('/home/claude/tmp/fig1.png',bbox_inches='tight',dpi=150)
print(iid, cy,cx, 'n MA in crop', ndimage.label(M['MA'][y0:y0+2*h,x0:x0+2*h])[1], 'crop px at 224:', round(2*h*s224,1))
