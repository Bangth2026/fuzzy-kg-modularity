import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0,'/home/claude/kt/src'); import analyse as A
OUT='/home/claude/kt/paper'
plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
NAMES={'assist09':'AS09','assist12':'AS12','assist15':'AS15','assist17':'AS17','junyi':'Junyi','algebra05':'Alg05','eedi':'Eedi','ednet':'EdNet'}

def fig_pipeline():
    fig,ax=plt.subplots(figsize=(7.4,2.6)); ax.axis('off'); ax.set_xlim(0,12.8); ax.set_ylim(-0.45,3.2)
    W,H=2.1,1.05
    B={'logs':(0.1,1.05,'Training logs\n(70% of learners;\n10% in sparse regime)'),
       'kg':(2.7,2.0,'Fuzzy prerequisite KG\n(order-preserving,\nco-practice or\nDKT-influence)'),
       'exp':(2.7,0.1,'Expert structure\n(where published)'),
       'comp':(5.3,2.0,'Rank coding and\nproduct composition\n(shortest paths,\ng = -ln x)'),
       'audit':(5.3,0.1,'Pre-training audit\nrank-coded Q_F drop,\nProposition 1\ncertificate'),
       'model':(7.9,1.05,'FKG-KT\ngraph-refined\nembeddings,\nprerequisite-biased\nattention, GRU state'),
       'out':(10.5,1.05,'P(correct)\non the next attempt')}
    for k,(x,y,t) in B.items():
        ax.add_patch(FancyBboxPatch((x,y),W,H,boxstyle='round,pad=0.03',fc='#eef2f7',ec='#3b5b8a',lw=0.8))
        ax.text(x+W/2,y+H/2,t,ha='center',va='center',fontsize=5.3)
    def L(a,sa,b,sb,cs='arc3'):
        xa,ya,_=B[a]; xb,yb,_=B[b]
        pa={'r':(xa+W,ya+H/2),'b':(xa+W/2,ya),'t':(xa+W/2,ya+H)}[sa]; pb={'l':(xb,yb+H/2),'t':(xb+W/2,yb+H),'b':(xb+W/2,yb)}[sb]
        ax.annotate('',xy=pb,xytext=pa,arrowprops=dict(arrowstyle='->',lw=0.7,color='#333',connectionstyle=cs))
    L('logs','r','kg','l'); L('logs','r','exp','l'); L('kg','r','comp','l'); L('kg','b','audit','t')
    L('comp','r','model','l');  L('audit','r','model','l'); L('model','r','out','l')
    ax.plot([3.75,3.75,8.95],[0.07,-0.3,-0.3],lw=0.7,color='#333'); ax.annotate('',xy=(8.95,1.02),xytext=(8.95,-0.3),arrowprops=dict(arrowstyle='->',lw=0.7,color='#333'))
    fig.savefig(f'{OUT}/fig1.png',dpi=300,bbox_inches='tight'); plt.close(fig)

def fig_gain(stats):
    """Gain of the graph (FKG-KT/transition:product3 minus FKG-KT/none) vs training interactions per KC."""
    df=A.df; fig,axs=plt.subplots(1,2,figsize=(7.2,2.6))
    for ax,metric,title in zip(axs,['test_auc','test_auc_newkc'],['all predictions','first attempt on a KC']):
        for fr,mk in ((0.1,'o'),(1.0,'s')):
            xs=[];ys=[];es=[]
            for d in NAMES:
                a=df[(df.data==d)&(df.frac==fr)&(df.cfg=='FKG-KT/transition:product3')][metric]
                b=df[(df.data==d)&(df.frac==fr)&(df.cfg=='FKG-KT/none')][metric]
                if len(a)==0 or len(b)==0: continue
                x=stats.loc[d,'train_int_per_kc']*(0.1 if fr<1 else 1)
                xs.append(x); ys.append(a.mean()-b.mean()); ax.annotate(NAMES[d],(x,a.mean()-b.mean()),fontsize=5.5,xytext=(2,2),textcoords='offset points')
            ax.scatter(xs,ys,marker=mk,s=18,label='sparse (10%)' if fr<1 else 'full',color='#c0392b' if fr<1 else '#2c3e50')
        ax.axhline(0,lw=0.6,color='gray'); ax.set_xscale('log'); ax.set_xlabel('training interactions per KC'); ax.set_ylabel('AUC gain of the KG'); ax.set_title(title,fontsize=8)
    axs[0].legend(frameon=False,fontsize=6.5); fig.tight_layout(); fig.savefig(f'{OUT}/fig2.png',dpi=300); plt.close(fig)

def fig_audit(v):
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.6))
    order=['support','permuted','cut25','cut50','cut75']
    for i,g in enumerate(order):
        y=v[v.graph==g].loss.values; axs[0].scatter(np.full(len(y),i)+np.random.default_rng(i).uniform(-.15,.15,len(y)),y,s=10,alpha=.7,color='#3b5b8a')
    axs[0].axhline(0,lw=0.6,color='gray'); axs[0].set_xticks(range(5)); axs[0].set_xticklabels(order); axs[0].set_ylabel('AUC loss vs rank-coded KG')
    axs[0].set_title('candidate graphs, sparse regime',fontsize=8)
    for c,col in (('transition','#c0392b'),('copractice','#2c3e50')):
        s=v[(v.construction==c)&v.graph.str.startswith('cut')]; axs[1].scatter(s.rank_drop,s.loss,s=12,color=col,label=c)
    axs[1].axhline(0,lw=0.6,color='gray'); axs[1].set_xlabel('rank-coded Q_F drop of the cut'); axs[1].set_ylabel('AUC loss'); axs[1].legend(frameon=False,fontsize=6.5)
    axs[1].set_title('audit vs downstream loss',fontsize=8); fig.tight_layout(); fig.savefig(f'{OUT}/fig3.png',dpi=300); plt.close(fig)

if __name__=='__main__':
    stats=pd.read_csv(f'{OUT}/table1.csv').set_index('data')
    fig_pipeline(); fig_gain(stats); fig_audit(A.audit_validity(0.1)); print('figures ok')
