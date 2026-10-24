"""Local map inspection of retained OSM geometry; no new downloads."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from routing import network, stations
from prepare import ROOT, C

for city in C['cities']:
    net = network(city)
    fig, ax = plt.subplots(figsize=(12, 10))
    coords = {n: v['xy'] for n, v in net['nodes'].items()}
    for w in net['water']:
        p = w['xy']
        if w['kind'] == 'polygon':
            ax.fill(*zip(*p), color='#cce4ed', alpha=.7)
        else:
            ax.plot(*zip(*p), color='#88b7cc', lw=1)
    ax.add_collection(LineCollection([[coords[str(e['u'])], coords[str(e['v'])]] for e in net['edges']], colors='#b5bbc0', linewidths=.4))
    for b in net['crossings']:
        ax.text(*b['xy'], b['id'], color='red', fontsize=10)
    ss = stations(city)
    for s in ss:
        ax.scatter(*s['xy'], s=12, color='black')
        ax.text(*s['xy'], str(s['id'])[-3:], fontsize=7)
    x, y = zip(*(s['xy'] for s in ss))
    ax.set_xlim(min(x)-700, max(x)+700);ax.set_ylim(min(y)-700,max(y)+700)
    ax.set_aspect('equal');ax.set_title(city+' retained cycling network / UTM metres')
    fig.savefig(ROOT/'output'/f'{city}_map_inspection.png', dpi=130)
    print(city, [(b['id'], b['name'], b['xy']) for b in net['crossings']])
