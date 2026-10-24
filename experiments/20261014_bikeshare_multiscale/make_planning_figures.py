"""Static research figures drawn from the saved planning tables, not hand counts."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from prepare import ROOT
report=json.loads((ROOT/'app/data/report.json').read_text())
out=ROOT/'output/planning'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
for ax,scale in zip(axes,[0,200]):
    data=sorted([r for r in report['benefits'] if r['city']=='washington' and r['workload']=='controlled' and r['policy']=='no_redistribution' and r['comparison']=='investment' and r['scale_m']==scale],key=lambda r:-r['additional_completed'])
    for i,r in enumerate(data):
        y=2-i;ax.barh(y,r['additional_completed'],height=.42,color='#0b7a70' if r['additional_completed']>=0 else '#c44b3c')
        ax.plot([r['additional_completed_p10'],r['additional_completed_p90']],[y,y],color='#263c43',lw=1.3,marker='|')
    ax.set_yticks([2,1,0],[r['intervention'] for r in data]);ax.axvline(0,color='gray',lw=.7);ax.set_xlim(-4.2,5.2);ax.set_xlabel('Additional completed trips per matched run')
    ax.set_title('Detailed network' if not scale else '200 m reduction');ax.grid(axis='x',alpha=.15)
fig.suptitle('Infrastructure interventions ranked by additional completed bike trips',fontsize=14)
fig.text(.02,.005,'Washington controlled demand; no redistribution; n=12 matched seeds. Whiskers: p10-p90, not confidence intervals.',fontsize=9)
for ext in ['png','svg','pdf']:fig.savefig(out/f'investment_rankings.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
for city,color in [('washington','#0b7a70'),('kaliningrad','#3376bd')]:
    rows=[r for r in report['network_audit']['scales'] if r['city']==city];ax.plot([r['scale_m'] for r in rows],[r['mean_route_error_pct'] for r in rows],'-o',label=city.title(),color=color)
ax.set_xlabel('Routing reduction scale (m)');ax.set_ylabel('Mean route-length error (%)');ax.legend();ax.grid(alpha=.2);ax.set_title('Averaging route errors does not certify planning-rank stability')
for ext in ['png','svg']:fig.savefig(out/f'routing_scale_error.{ext}',dpi=180)
print('Planning figures saved.')
