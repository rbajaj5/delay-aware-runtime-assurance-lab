"""Resume the interrupted multiscale run without replacing valid episode logs."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from prepare import C, DATA, ROOT, dump, sha
from routing import Router, load_model, network, stations
from simulate import controlled_tape, historical_tapes, simulate

OUT = ROOT / "output"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_name(city, tape, crossing, policy, scale):
    ident = f"{city}_{tape['workload']}_{tape['seed']}_{crossing or 'all_open'}_{policy}_{scale}"
    return ident, ident + ".json.gz"


def save_log(path, result, replay, tape):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"events": replay["events"], "result": result, "initial": tape["initial"]}, stream)


def load_tapes(city, ss, base_router):
    expected = [controlled_tape(city, C["seed_start"] + i, base_router, ss) for i in range(C["seeds"])]
    if city == "washington":
        expected.extend(historical_tapes(ss)[:7])
    for tape in expected:
        path = OUT / "tapes" / f"{city}_{tape['workload']}_{tape['seed']}.json"
        if not path.exists():
            raise FileNotFoundError(f"Expected retained demand tape is missing: {path}")
        saved = read_json(path)
        if saved.get("sha256") != tape.get("sha256"):
            raise ValueError(f"Retained tape disagrees with current code/config: {path.name}")
    return expected


def selected_crossings(city, ss, base_router, first_tape):
    usage = Counter()
    byid = {s["id"]: s for s in ss}
    for request in first_tape["requests"]:
        route = base_router.route(byid[request["origin"]]["node"], byid[request["destination"]]["node"])
        if route:
            usage.update(set(base_router.edges[key]["crossing"] for key in route["edges"]) - {None})
    stored = read_json(DATA / f"{city}_scenarios.json")
    selected = stored["selected_crossings"]
    if dict(usage) != stored["baseline_usage"]:
        raise ValueError(f"Current route/scenario ranking differs from retained {city} scenario file")
    cutoff = sorted(usage.values(), reverse=True)[2]
    if len(selected) != 3 or any(usage[name] < cutoff for name in selected):
        raise ValueError(f"Retained {city} closures are not among the three highest-use crossings")
    return selected


def check_result(result, city, tape, crossing, policy, scale):
    expected = (city, tape["workload"], tape["seed"], crossing or "all_open", policy, scale, tape["sha256"])
    actual = tuple(result[k] for k in ["city", "workload", "seed", "scenario", "policy", "scale_m", "input_sha256"])
    if actual != expected:
        raise ValueError(f"Existing episode does not match current design: expected {expected}, got {actual}")


def write_csv_if_absent(path: Path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    if path.exists():
        if path.read_bytes() != content.encode("utf-8"):
            raise FileExistsError(f"Refusing to replace existing nonmatching summary: {path}")
        return
    path.write_text(content, encoding="utf-8", newline="")


def main():
    expected_count = 0
    initial_logs = len(list((OUT / "logs").glob("*.json.gz")))
    completed_rows = []
    validations = []

    for city in C["cities"]:
        net = network(city)
        ss = stations(city)
        base_graph, _ = load_model(city, 0)
        base_router = Router(net, base_graph)
        tapes = load_tapes(city, ss, base_router)
        crossings = selected_crossings(city, ss, base_router, tapes[0])

        for scale in C["scales_m"]:
            graph, _ = load_model(city, scale)
            router = Router(net, graph)
            model = read_json(DATA / f"{city}_scale_{scale}.json")
            groups = model["node_to_group"]
            router.station_groups = {s["id"]: groups.get(str(s["node"]), s["id"]) for s in ss}

            for tape in tapes:
                for crossing in [None] + crossings:
                    for policy in C["policies"]:
                        expected_count += 1
                        ident, name = file_name(city, tape, crossing, policy, scale)
                        log_path = OUT / "logs" / name
                        replay_path = OUT / "replays" / f"{ident}.json"
                        capture = tape["workload"] == "controlled" and tape["seed"] < C["seed_start"] + 2

                        if log_path.exists():
                            with gzip.open(log_path, "rt", encoding="utf-8") as stream:
                                saved = json.load(stream)
                            check_result(saved["result"], city, tape, crossing, policy, scale)
                            if saved["initial"] != tape["initial"]:
                                raise ValueError(f"Existing initial inventory differs: {name}")
                            result = saved["result"]
                            if capture and not replay_path.exists():
                                result, replay = simulate(net, ss, router, tape, crossing, policy, capture=True)
                                result["scale_m"] = scale
                                check_result(result, city, tape, crossing, policy, scale)
                                dump(replay_path, replay)
                                save_log(log_path, result, replay, tape)
                        else:
                            result, replay = simulate(net, ss, router, tape, crossing, policy, capture=capture)
                            result["scale_m"] = scale
                            check_result(result, city, tape, crossing, policy, scale)
                            save_log(log_path, result, replay, tape)
                            if capture:
                                dump(replay_path, replay)
                            print("completed", name, flush=True)

                        completed_rows.append(result)
                        validations.append({"episode": ident, "conservation": True, "capacities": True,
                                            "closure_arc_checks": result["route_arcs_validated"],
                                            "request_accounting": result["requests"] == result["completed"] + result["failed_pickups"] + result["unreachable"] + result["unfinished"]})

    if len(completed_rows) != expected_count:
        raise AssertionError(f"Expected {expected_count} episodes; reconciled {len(completed_rows)}")
    actual_names = {p.name for p in (OUT / "logs").glob("*.json.gz")}
    expected_names = {file_name(row["city"], next(t for t in load_tape_cache[row["city"]] if t["seed"] == row["seed"] and t["workload"] == row["workload"]), None if row["scenario"] == "all_open" else row["scenario"], row["policy"], row["scale_m"])[1]
                      for row in completed_rows}
    if actual_names != expected_names:
        raise AssertionError(f"Unexpected or stale episode logs: {len(actual_names - expected_names)} extra, {len(expected_names - actual_names)} missing")

    write_csv_if_absent(OUT / "episode_results.csv", completed_rows)
    validation_path = OUT / "simulation_validation.json"
    if validation_path.exists():
        existing = read_json(validation_path)
        if len(existing) != expected_count:
            raise FileExistsError(f"Existing validation has unexpected row count: {validation_path}")
    else:
        dump(validation_path, validations)

    manifest = {
        "status": "complete",
        "demand_source": C["trip_url"],
        "week_start": C["week_start"],
        "week_end_exclusive": C["week_end_exclusive"],
        "config_sha256": sha(ROOT / "config.json"),
        "simulate_sha256": sha(ROOT / "simulate.py"),
        "routing_sha256": sha(ROOT / "routing.py"),
        "resume_runner_sha256": sha(ROOT / "resume_run.py"),
        "expected_episodes": expected_count,
        "logs_before_resume": initial_logs,
        "logs_after_resume": len(actual_names),
        "newly_simulated": len(actual_names) - initial_logs,
        "matched_input_tapes": len(list((OUT / "tapes").glob("*.json"))),
        "network_validation_sha256": sha(OUT / "network_validation.json"),
        "boundary_and_restriction_caveats": [
            "Historical April 2025 demand is routed on the October 2026 OSM snapshot; this is not historical routing reconstruction.",
            "Study boundaries are finite map extracts; the Kaliningrad half-width sensitivity lost 147 station OD paths.",
            "Via-way restriction relaxation is only an optimistic sensitivity bound, not legally validated routing.",
            "Station connectors are map-screened, not field verified; capacity and initial inventory are hypothetical.",
        ],
        "request_level_export": "Not regenerated from event logs; retained detailed replay files cover the first two controlled seeds only.",
    }
    manifest_path = OUT / "resume_manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        for key in ["config_sha256", "simulate_sha256", "routing_sha256", "expected_episodes", "network_validation_sha256"]:
            if old.get(key) != manifest[key]:
                raise FileExistsError(f"Existing resume manifest does not match current inputs at {key}")
    else:
        dump(manifest_path, manifest)
    print(json.dumps({"expected_episodes": expected_count, "logs_before": initial_logs,
                      "logs_after": len(actual_names), "newly_simulated": len(actual_names)-initial_logs,
                      "results_csv": str(OUT / "episode_results.csv")}, indent=2), flush=True)


if __name__ == "__main__":
    load_tape_cache = defaultdict(list)
    for tape_path in (OUT / "tapes").glob("*.json"):
        tape = read_json(tape_path)
        load_tape_cache[tape["city"]].append(tape)
    main()
