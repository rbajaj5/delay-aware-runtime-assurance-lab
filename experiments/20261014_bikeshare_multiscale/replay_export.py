"""Reproduce retained episodes, then export a separately corrected baseline.

Run with Python 3.11. No demand, topology, or retained output is replaced.
"""
import csv
import gzip
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

from prepare import ROOT, DATA, C, dump, csvout, sha
from routing import Router, network, stations, load_model
from corrected_model import CorrectedRouter
from simulate import simulate, fingerprint

OUT = ROOT / 'output'
REPLAY = OUT / 'journeys'
APP = ROOT / 'app' / 'data'


def gzread(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def gzwrite(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 makes the compressed export itself reproducible.
    with path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, compresslevel=3) as f:
            f.write(json.dumps(value, separators=(',', ':'), allow_nan=False).encode())


def identity(r):
    return f"{r['city']}_{r['workload']}_{int(r['seed'])}_{r['scenario']}_{r['policy']}_{int(r['scale_m'])}"


def compare(a, b):
    for key, expected in b.items():
        if key == 'wall_s':
            continue
        actual = a[key]
        if isinstance(expected, (int, float)):
            assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), (key, actual, expected)
        else:
            assert actual == expected, (key, actual, expected)


def export_network(city, net=None, name=None):
    net = net or network(city)
    ss = stations(city)
    ids = list(net['nodes'])
    ni = {n: i for i, n in enumerate(ids)}
    center = net['nodes'][ids[0]]['xy']
    def xy(p):
        return [round(p[0] - center[0], 4), round(p[1] - center[1], 4)]
    edges = [[ni[str(e['u'])], ni[str(e['v'])], e['m'], e['crossing']] for e in net['edges']]
    compact = {'city': city, 'crs': C['cities'][city]['epsg'], 'origin': center,
               'nodes': [xy(net['nodes'][n]['xy']) for n in ids], 'edges': edges,
               'stations': [{**s, 'xy': xy(s['xy'])} for s in ss],
               'water': [{**w, 'xy': [xy(p) for p in w['xy']]} for w in net['water']],
               'crossings': [{**b, 'xy': xy(b['xy'])} for b in net['crossings']],
               'source': 'OpenStreetMap October 2026; finite buffered extract',
               'groups': {}}
    for scale in C['scales_m']:
        d = json.loads((DATA / f'{city}_scale_{scale}.json').read_text())
        labels = d['node_to_group']
        compact['groups'][str(scale)] = [labels.get(str(s['node']), s['id']) for s in ss]
    gzwrite(APP / f'{name or city}.json.gz', compact)
    return {e['id']: i for i, e in enumerate(net['edges'])}


def compact_replay(replay, edge_index, edges, version, ident, verification):
    paths, lookup, moves = [], {}, []
    for m in replay['movements']:
        path = tuple(edge_index[leg[0]] for leg in m['legs'])
        if path not in lookup:
            lookup[path] = len(paths)
            paths.append(path)
        cursor = m['start']
        waits = []
        speed = None
        for j, (key, depart, finish) in enumerate(m['legs']):
            if depart - cursor > 1e-7:
                waits.append([j, depart - cursor])
            measured = edges[key]['m'] / (finish - depart)
            if speed is None:
                speed = measured
            assert math.isclose(measured, speed, rel_tol=1e-7)
            cursor = finish
        assert abs(cursor - m['arrive']) < 1e-6
        total_m = sum(edges[k]['m'] for k, _, _ in m['legs'])
        if speed is not None:
            assert abs(m['start'] + total_m / speed + sum(w[1] for w in waits) - m['arrive']) < 1e-5
        moves.append({k: v for k, v in m.items() if k != 'legs'} |
                     {'path': lookup[path], 'speed': speed or C['crew_speed_m_s'], 'waits': waits})
    return {**replay, 'movements': moves, 'paths': paths, 'episode': ident,
            'version': version, 'verification': verification,
            'coverage': 'Complete episode: every request, movement, event; snapshots every 60 s. No new randomness.'}


def source_fingerprint():
    files = [ROOT / n for n in ('config.json', 'simulate.py', 'routing.py', 'corrected_model.py', 'replay_export.py')]
    files += sorted((OUT / 'tapes').glob('*.json'))
    files += [OUT / 'episode_results.csv']
    files += [DATA / f'{city}_{suffix}.json' for city in C['cities']
              for suffix in ['network', 'stations', 'scenarios', 'scale_0', 'scale_200', 'scale_600', 'scale_1400']]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in files}


def main():
    sources = source_fingerprint()
    stamp = fingerprint(sources)
    manifest_path = REPLAY / 'manifest.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if old:
        assert old['source_fingerprint'] == stamp, 'Source changed: retain old exports and use a new version directory.'
    REPLAY.mkdir(exist_ok=True)
    corrected = OUT / 'corrected'
    (corrected / 'logs').mkdir(parents=True, exist_ok=True)
    catalog, retained_rows, corrected_rows = [], [], []
    checked = events = requests = corrected_count = 0
    started = time.perf_counter()
    for city in C['cities']:
        net, ss = network(city), stations(city)
        ei = export_network(city, net)
        tapes = {f.stem: json.loads(f.read_text()) for f in sorted((OUT / 'tapes').glob(f'{city}_*.json'))}
        for tape in tapes.values():
            assert fingerprint({k: v for k, v in tape.items() if k != 'sha256'}) == tape['sha256']
        for scale in C['scales_m']:
            graph, info = load_model(city, scale)
            labels = json.loads((DATA / f'{city}_scale_{scale}.json').read_text())['node_to_group']
            routers = {v: cls(net, graph) for v, cls in [('retained', Router), ('corrected', CorrectedRouter)]}
            for r in routers.values():
                r.station_groups = {s['id']: labels.get(str(s['node']), s['id']) for s in ss}
            files = sorted((OUT / 'logs').glob(f'{city}_*_{scale}.json.gz'))
            for file in files:
                saved = gzread(file)
                result = saved['result']
                ident = identity(result)
                tape = tapes[f"{city}_{result['workload']}_{result['seed']}"]
                assert tape['sha256'] == result['input_sha256']
                crossing = None if result['scenario'] == 'all_open' else result['scenario']
                path = REPLAY / 'retained' / (ident + '.json.gz')
                if path.exists():
                    replay = gzread(path)
                    assert replay['source_fingerprint'] == stamp
                    compare(replay['result'], result)
                    assert replay['events'] == saved['events']
                else:
                    rerun, replay = simulate(net, ss, routers['retained'], tape, crossing, result['policy'], True)
                    rerun['scale_m'] = scale
                    compare(rerun, result)
                    assert replay['events'] == saved['events'], ('event mismatch', ident)
                    replay['result'] = result
                    replay = compact_replay(replay, ei, routers['retained'].edges, 'retained', ident, 'Every non-timing summary field and every event matches retained output')
                    replay['source_fingerprint'] = stamp
                    gzwrite(path, replay)
                checked += 1
                events += len(replay['events'])
                requests += len(replay['requests'])
                retained_rows.append(result)
                catalog.append({'version': 'retained', **result, 'file': str(path.relative_to(ROOT)).replace('\\', '/')})
                cpath = REPLAY / 'corrected' / (ident + '.json.gz')
                if crossing is None:
                    if cpath.exists():
                        creplay = gzread(cpath)
                        assert creplay['source_fingerprint'] == stamp
                    else:
                        cr, creplay = simulate(net, ss, routers['corrected'], tape, None, result['policy'], True)
                        cr['scale_m'] = scale
                        creplay = compact_replay(creplay, ei, routers['corrected'].edges, 'corrected', ident,
                                                  'Versioned all-open correction; same saved input tape')
                        creplay['source_fingerprint'] = stamp
                        gzwrite(cpath, creplay)
                    assert creplay['result']['closure_wait_rider_hours'] == 0
                    gzwrite(corrected / 'logs' / file.name, {'events': creplay['events'], 'result': creplay['result'], 'initial': tape['initial']})
                    cr = creplay['result']
                    corrected_count += 1
                else:
                    cpath = path
                    cr = result
                    target = corrected / 'logs' / file.name
                    if not target.exists():
                        shutil.copy2(file, target)
                corrected_rows.append(cr)
                catalog.append({'version': 'corrected', **cr, 'file': str(cpath.relative_to(ROOT)).replace('\\', '/')})
                if checked % 32 == 0:
                    print(f'Reconciled {checked}/992; corrected open baselines {corrected_count}; elapsed {time.perf_counter()-started:.0f}s', flush=True)
            dump(manifest_path, {'status': 'in_progress', 'source_fingerprint': stamp, 'sources': sources, 'retained_verified': checked})
    assert checked == 992 and corrected_count == 248
    csvout(corrected / 'episode_results.csv', corrected_rows)
    dump(APP / 'catalog.json', catalog)
    # Request-level BI export covers every retained and corrected view, including reused closures.
    fields = ['version', 'episode', 'city', 'workload', 'day', 'seed', 'scenario', 'policy', 'scale_m',
              'input_sha256', 'id', 'time', 'origin', 'destination', 'speed_factor', 'status', 'picked_up_s',
              'completed_s', 'distance_m', 'closure_wait_s', 'full_events', 'alternative_returns', 'current_destination']
    with gzip.open(REPLAY / 'request_outcomes.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for item in catalog:
            d = gzread(ROOT / item['file'])
            for r in d['requests']:
                writer.writerow({**item, 'episode': d['episode'], **r})
    manifest = {'status': 'complete', 'source_fingerprint': stamp, 'sources': sources,
                'retained_verified': checked, 'corrected_open_episodes': corrected_count,
                'corrected_closure_episodes_reused': checked-corrected_count, 'paired_input_tapes': 31,
                'retained_events_verified_exactly': events, 'retained_request_outcomes': requests,
                'summary_mismatches': 0, 'event_mismatches': 0, 'excluded_comparison_field': 'wall_s (runtime varies)',
                'scope': 'All 992 retained episodes plus a corrected 992-episode view. Full journey replay coverage; no smoke/pilot substitution.',
                'correction': 'No closure is now distinct from an edge with no crossing ID. Retained source/results are untouched.',
                'elapsed_s': time.perf_counter()-started}
    dump(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
