import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':7,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False})
D=json.load(open('/home/claude/results/data_analysis.json'))
m=pd.read_csv('/home/claude/prep/meta.csv',dtype={'id':str}); m=m[m.exclude.fillna('')=='']
GC=['#4c72b0','#dd4444','#e8a33d','#8c6bb1','#3a3a3a']
fig,ax=plt.subplots(1,4,figsize=(7.2,1.9),gridspec_kw=dict(wspace=0.45,width_ratios=[1.15,1.1,1,1]))
# (a) grade distribution
names=['EyePACS\nall','EyePACS\nsubset','APTOS','IDRiD']
G=[D['eyepacs_full']['grades'],D['eyepacs']['grades'],D['aptos']['grades'],D['idrid']['grades']]
G=np.array(G,float); G=G/G.sum(1,keepdims=True)*100
b=np.zeros(4)
for k in range(5):
    ax[0].bar(range(4),G[:,k],bottom=b,color=GC[k],width=0.7,label=f'Grade {k}'); b+=G[:,k]
for i in range(4): ax[0].text(i,G[i,0]+G[i,1]/2,f'{G[i,1]:.1f}',ha='center',va='center',fontsize=5.5,color='white')
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(names,fontsize=5.5,rotation=35); ax[0].set_ylabel('Share of images (%)')
ax[0].legend(fontsize=5,frameon=False,ncol=3,loc='lower left',bbox_to_anchor=(-0.08,0.99),handlelength=0.8,columnspacing=0.6)
ax[0].set_title('(a) ICDR grades',fontsize=7,pad=22)
# (b) contrast by dataset
dsn=['eyepacs','aptos','idrid','ddr']; lab=['EyePACS','APTOS','IDRiD','DDR']
data=[m[m.ds==d].contrast.values for d in dsn]
bp=ax[1].boxplot(data,showfliers=False,widths=0.6,patch_artist=True)
for p,c in zip(bp['boxes'],['#9ecae1','#fdae6b','#a1d99b','#bcbddc']): p.set_facecolor(c)
for med in bp['medians']: med.set_color('k')
ax[1].set_xticks(range(1,5)); ax[1].set_xticklabels(lab,fontsize=6,rotation=30); ax[1].set_ylabel('RMS contrast (grey levels)')
ax[1].set_title('(b) Acquisition shift',fontsize=7,pad=22)
# (c) MA diameter at model input
c=pd.read_csv('/home/claude/prep/ddr_components.csv').merge(pd.read_csv('/home/claude/prep/ddr_sides.csv'),on='id')
ma=c[c.type=='MA']; d=2*np.sqrt(ma.area/np.pi)
bins=np.linspace(0,6,31)
ax[2].hist(d*224/ma.side,bins=bins,alpha=0.75,color='#dd4444',label='224 px')
ax[2].hist(d*448/ma.side,bins=bins,alpha=0.55,color='#4c72b0',label='448 px')
ax[2].axvline(1,ls='--',lw=0.7,color='k'); ax[2].set_xlabel('MA diameter at input (px)'); ax[2].set_ylabel('Microaneurysms')
ax[2].legend(fontsize=5.5,frameon=False); ax[2].set_title(f'(c) MA size, DDR (n={len(ma):,})',fontsize=7,pad=22)
# (d) EyeQ quality by grade
t=np.array(D['eyeq']['table'],float); t=t/t.sum(1,keepdims=True)*100
bb=np.zeros(5)
for k,(cl,lb) in enumerate(zip(['#74c476','#fdd0a2','#e6550d'],['Good','Usable','Reject'])):
    ax[3].bar(range(5),t[:,k],bottom=bb,color=cl,width=0.7,label=lb); bb+=t[:,k]
ax[3].set_xticks(range(5)); ax[3].set_xlabel('ICDR grade'); ax[3].set_ylabel('Share of images (%)')
ax[3].legend(fontsize=5,frameon=False,loc='lower left',bbox_to_anchor=(-0.1,0.99),ncol=3,handlelength=0.8)
ax[3].set_title('(d) EyeQ quality by grade',fontsize=7,pad=22)
fig.savefig('/home/claude/figs/fig2_data.pdf',bbox_inches='tight'); fig.savefig('/home/claude/tmp/fig2.png',bbox_inches='tight',dpi=170)
