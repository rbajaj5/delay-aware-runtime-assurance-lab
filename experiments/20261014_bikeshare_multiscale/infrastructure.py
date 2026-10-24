"""Bounded hypothetical connections on the retained geography and demand.

Candidate anchors are frozen before service outcomes are calculated. These are
abstract cycling links, not engineered bridge designs or construction proposals.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import networkx as nx
from pyproj import Transformer

from prepare import ROOT, DATA, C, dump, csvout, sha
from routing import network, stations, construct
from corrected_model import CorrectedRouter
from simulate import simulate, fingerprint
from replay_export import gzwrite, gzread, compact_replay, export_network, identity

OUT = ROOT / 'output' / 'infrastructure'
# UTM metre coordinates read from the retained river/cycling-network map.
ANCHORS = {
    'washington': [
        ('I1', 'North riverbank bicycle link', 'new_crossing', [[320650, 4307750], [321600, 4307900]]),
        ('I2', 'Central riverbank bicycle link', 'new_crossing', [[321100, 4306700], [322000, 4306800]]),
        ('I3', 'West-bank approach connection', 'approach_connection', [[320500, 4307350], [320950, 4306820]])],
    'kaliningrad': [
        ('I1', 'West Pregolya bicycle link', 'new_crossing', [[467550, 6062220], [467550, 6062000]]),
        ('I2', 'East Pregolya bicycle link', 'new_crossing', [[469300, 6062380], [469300, 6062140]]),
        ('I3', 'South-bank approach connection', 'approach_connection', [[468850, 6061870], [469100, 6061420]])]
}


def specs(city, net):
    # Endpoints must already have usable two-way reachability. Avoid via nodes
    # so new connections cannot implicitly bypass a modeled turn restriction.
    graph = nx.DiGraph()
    graph.add_edges_from((e['u'], e['v']) for e in net['edges'])
    component = max(nx.strongly_connected_components(graph), key=len)
    forbidden = {r['via'] for r in net.get('turn_rules', [])}
    candidates = sorted(component - forbidden)
    result = []
    for ident, name, kind, anchors in ANCHORS[city]:
        endpoints = [min(candidates, key=lambda n: (math.dist(net['nodes'][str(n)]['xy'], a), n)) for a in anchors]
        points = [net['nodes'][str(n)] for n in endpoints]
        assert endpoints[0] != endpoints[1]
        snap = [math.dist(a, p['xy']) for a, p in zip(anchors, points)]
        assert max(snap) < 260, (city, ident, snap)
        length = math.dist(points[0]['xy'], points[1]['xy']) * 1.10
        result.append({'id': ident, 'name': name, 'kind': kind, 'city': city,
                       'anchor_utm_m': anchors, 'nodes': endpoints, 'snap_m': snap,
                       'endpoints': [{'node': n, 'xy': p['xy'], 'latitude': p['lat'], 'longitude': p['lon']} for n, p in zip(endpoints, points)],
                       'length_m': length, 'length_rule': 'Straight endpoint distance times 1.10, allowing a hypothetical approach allowance.',
                       'direction': 'bidirectional', 'access': 'Hypothetical unrestricted bicycle and cargo-cycle access, no toll, no capacity constraint.',
                       'rider_speed_m_s': C['rider_speed_m_s'], 'crew_speed_m_s': C['crew_speed_m_s'],
                       'assumption': 'Map-anchored abstract connection only. No terrain, grade, land ownership, navigation clearance, structural feasibility, environmental or regulatory validation.',
                       'baseline': 'corrected existing network, all crossings open; each link installed separately'})
    return result


def modified(net, spec):
    d = copy.deepcopy(net)
    u, v = spec['nodes']
    for i, (a, b) in enumerate(((u, v), (v, u))):
        d['edges'].append({'id': f"hyp:{spec['id']}:{i}", 'u': a, 'v': b,
                           'way': -9000-int(spec['id'][1:]), 'm': spec['length_m'],
                           'bridge': spec['kind'] == 'new_crossing',
                           'river_bridge': spec['kind'] == 'new_crossing',
                           'crossing': 'H'+spec['id'][1:], 'name': spec['name'],
                           'tags': {'highway': 'cycleway', 'bicycle': 'yes', 'hypothetical': 'yes'}})
    d['crossings'].append({'id': 'H'+spec['id'][1:], 'name': spec['name'],
                           'xy': [(a+b)/2 for a, b in zip(spec['endpoints'][0]['xy'], spec['endpoints'][1]['xy'])],
                           'hypothetical': True, 'evidence': 'Synthetic intervention, not OSM'})
    return d


def get_model(city, spec, net, ss, scale, stamp):
    path = OUT / 'models' / f"{city}_{spec['id']}_{scale}.json.gz"
    if path.exists():
        data = gzread(path)
        assert data['stamp'] == stamp
        graph = nx.MultiDiGraph()
        for e in data['arcs']:
            graph.add_edge(e['u'], e['v'], key=e['key'], **{k: v for k, v in e.items() if k not in ['u', 'v', 'key']})
        return graph, data['labels'], data['info']
    graph, labels, info = construct(net, ss, scale)
    gzwrite(path, {'stamp': stamp, 'labels': labels, 'info': info,
                   'arcs': [{'u': u, 'v': v, 'key': k, **d} for u, v, k, d in graph.edges(keys=True, data=True)]})
    return graph, {str(k): v for k, v in labels.items()}, info


def main(smoke=False):
    OUT.mkdir(exist_ok=True)
    rows, catalog, costs, checks = [], [], [], []
    frozen = {city: specs(city, network(city)) for city in C['cities']}
    specpath = OUT / 'interventions.json'
    if specpath.exists():
        assert json.loads(specpath.read_text()) == frozen, 'Frozen candidate definitions changed.'
    else:
        dump(specpath, frozen)
    source = {f: sha(ROOT/f) for f in ['config.json', 'infrastructure.py', 'simulate.py', 'routing.py', 'corrected_model.py']}
    source['specs'] = fingerprint(frozen)
    for city in C['cities']:
        for suffix in ['network', 'stations']:
            source[f'{city}_{suffix}'] = sha(DATA/f'{city}_{suffix}.json')
    stamp = fingerprint(source)
    start = time.perf_counter()
    for city in C['cities']:
        original, ss = network(city), stations(city)
        tapes = [json.loads(p.read_text()) for p in sorted((ROOT/'output'/'tapes').glob(f'{city}_*.json'))]
        if smoke:
            tapes = [next(t for t in tapes if t['workload'] == 'controlled')]
        for spec in frozen[city]:
            net = modified(original, spec)
            ei = export_network(city, net, f"{city}_{spec['id']}")
            for scale in C['scales_m']:
                graph, labels, info = get_model(city, spec, net, ss, scale, stamp)
                router = CorrectedRouter(net, graph)
                router.station_groups = {s['id']: labels.get(str(s['node']), labels.get(s['node'], s['id'])) for s in ss}
                # All selected-station OD routes and all hypothetical arcs are checked
                # before the workload; Router verifies continuity and allowed turns.
                cold = time.perf_counter()
                reachable = 0
                new_arc_uses = 0
                for a in ss:
                    for b in ss:
                        if a['id'] == b['id']:continue
                        p = router.route(a['node'], b['node'])
                        if p:
                            reachable += 1
                            new_arc_uses += any(k.startswith('hyp:') for k in p['edges'])
                assert reachable == 552
                represented = {k for _, _, d in graph.edges(data=True) for k in d['path'] if k.startswith('hyp:')}
                assert represented == {f"hyp:{spec['id']}:0", f"hyp:{spec['id']}:1"}
                costs.append({**info, 'city': city, 'intervention': spec['id'], 'cold_routing_s': time.perf_counter()-cold,
                              'reachable_station_od': reachable, 'od_routes_using_intervention': new_arc_uses})
                for tape in tapes:
                    assert fingerprint({k:v for k,v in tape.items() if k != 'sha256'}) == tape['sha256']
                    for policy in C['policies']:
                        ident = f"{city}_{tape['workload']}_{tape['seed']}_{spec['id']}_{policy}_{scale}"
                        path = OUT / 'replays' / (ident+'.json.gz')
                        if path.exists():
                            replay = gzread(path)
                            assert replay['source_fingerprint'] == stamp
                            result = replay['result']
                        else:
                            result, replay = simulate(net, ss, router, tape, None, policy, True)
                            result.update(scenario=spec['id'], scale_m=scale)
                            replay = compact_replay(replay, ei, router.edges, 'investment', ident,
                                                     'Newly simulated hypothetical topology; exact saved input tape, corrected no-closure timing')
                            replay['source_fingerprint'] = stamp
                            gzwrite(path, replay)
                            gzwrite(OUT/'logs'/(ident+'.json.gz'), {'events': replay['events'], 'result': result, 'initial': tape['initial']})
                        assert result['closure_wait_rider_hours'] == 0
                        rows.append(result)
                        catalog.append({'version': 'investment', **result, 'file': str(path.relative_to(ROOT)).replace('\\','/'), 'network': f"{city}_{spec['id']}"})
                        checks.append({'episode': ident, 'matched_tape': True, 'all_od_reachable': True, 'hypothetical_arcs_preserved': True})
                print(f"Infrastructure {'smoke' if smoke else 'full'} {city} {spec['id']} {scale}m: {len(rows)} episodes, {new_arc_uses}/552 OD use link; elapsed {time.perf_counter()-start:.0f}s", flush=True)
    suffix = '_smoke' if smoke else ''
    csvout(OUT/f'episode_results{suffix}.csv', rows)
    csvout(OUT/f'model_costs{suffix}.csv', costs)
    dump(OUT/f'catalog{suffix}.json', catalog)
    dump(OUT/f'validation{suffix}.json', {'passed': True, 'episodes': len(rows), 'source': source, 'checks': checks,
                                         'all_reduced_models_rebuilt': True, 'old_route_caches_reused': False})
    print('DONE', len(rows), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke', action='store_true')
    main(p.parse_args().smoke)
