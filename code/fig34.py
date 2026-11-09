import json, glob, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.ndimage import zoom
from PIL import Image
plt.rcParams.update({'font.size':7,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False})
R='/home/claude/runs'; meta=pd.read_csv('/home/claude/prep/meta.csv',dtype={'id':str})
T=json.load(open('/home/claude/results/table.json'))
seeds=json.load(open('/home/claude/results/seeds.json'))
fig,ax=plt.subplots(1,3,figsize=(7.2,2.2),gridspec_kw=dict(wspace=0.35))
for a,(ds,nm) in zip(ax,[('internal','EyePACS internal'),('aptos','APTOS (external)'),('idrid','IDRiD (external)')]):
    for c,col,lb in [('B1','#888888','B1 baseline'),('P','#c0392b','P proposed (gated)')]:
        S=[];ids=None
        for s in seeds:
            z=np.load(f'{R}/{c}_s{s}/pred_{ds}.npz'); S.append(z['S']); ids=z['ids'].astype(str)
        d=meta[meta.ds==('eyepacs' if ds=='internal' else ds)].set_index('id').loc[ids]
        keep=(d.exclude.fillna('')=='').values.copy()
        if c=='P':
            g=np.load(f'{R}/GATE_s0/pred_{ds}.npz'); keep&=g['P'].argmax(1)!=2
        y=(d.grade.values[keep]>0); s=np.mean(S,0)[keep]
        fpr,tpr,_=roc_curve(y,s); a.plot(fpr,tpr,color=col,lw=1.1,label=f"{lb}, AUC {T[f'{ds}|{c}']['point']['auc']:.3f}")
        pt=T[f'{ds}|{c}']['point']; a.plot(1-pt['sp'],pt['se'],'o',color=col,ms=4)
    a.add_patch(plt.Rectangle((0,0.9),0.1,0.1,color='#2e86c1',alpha=0.15,lw=0))
    a.plot([0,1],[0,1],':',color='k',lw=0.5); a.set_xlabel('1 − specificity'); a.set_ylabel('Sensitivity'); a.set_title(nm,fontsize=7.5)
    a.legend(fontsize=5.5,frameon=False,loc='lower right')
fig.savefig('/home/claude/figs/fig3_roc.pdf',bbox_inches='tight'); fig.savefig('/home/claude/tmp/fig3.png',dpi=160,bbox_inches='tight')
# fig 4 heatmaps
mk=np.load('/home/claude/prep/ddr_masks512.npz'); mids=list(mk['ids'])
zb=np.load(f'{R}/B1_s0/pred_ddr.npz'); zp=np.load(f'{R}/P_s0/pred_ddr.npz'); ids=list(zp['ids'].astype(str))
cmp=pd.read_csv('/home/claude/prep/ddr_components.csv'); cnt=cmp.groupby('id').size()
choice=[i for i in ids if i.startswith('test_') and 8<=cnt.get(i,0)<=60][:3]
fig,ax=plt.subplots(3,3,figsize=(5.2,5.3),gridspec_kw=dict(wspace=0.03,hspace=0.08))
for r,i in enumerate(choice):
    k=ids.index(i); im=np.asarray(Image.open(f'/home/claude/prep/ddr/{i}.jpg')); m=mk['masks'][mids.index(i)]
    cam=zoom(zb['cam'][k].astype(float),512/7,order=1); att=zoom(zp['A'][k].astype(float).max(0),512/7,order=1)
    ax[r,0].imshow(im); ax[r,0].contour(m>0,levels=[0.5],colors='cyan',linewidths=0.5)
    for j,(hm,t) in enumerate([(cam,'B1 Grad-CAM'),(att,'P lesion attention')],1):
        ax[r,j].imshow(im); ax[r,j].imshow(hm,cmap='jet',alpha=0.45); py,px=np.unravel_index(hm.argmax(),hm.shape)
        ax[r,j].contour(m>0,levels=[0.5],colors='white',linewidths=0.4); ax[r,j].plot(px,py,'w+',ms=9,mew=1.5)
        if r==0: ax[r,j].set_title(t,fontsize=7.5)
    if r==0: ax[r,0].set_title('DDR image + lesion masks',fontsize=7.5)
    for a in ax[r]: a.axis('off')
fig.savefig('/home/claude/figs/fig4_heatmaps.pdf',bbox_inches='tight',dpi=200); fig.savefig('/home/claude/tmp/fig4.png',dpi=120,bbox_inches='tight')
print(choice)
