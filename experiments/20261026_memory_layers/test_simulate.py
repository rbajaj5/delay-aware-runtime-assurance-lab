import json
import unittest
from simulate import CONFIG, OUT, digest, inputs, run


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {(p,a):run(a,p,inputs(CONFIG["seed_start"]))
                    for p in CONFIG["profiles"] for a in CONFIG["policies"]}

    def test_exact_saved_reproduction(self):
        for (p,a), result in self.runs.items():
            self.assertEqual(result,json.loads((OUT/f"{p}_{a}.json").read_text()))

    def test_matched_inputs_and_knowledge(self):
        for p in CONFIG["profiles"]:
            left,right = (self.runs[p,a] for a in CONFIG["policies"])
            self.assertEqual(left["inputs"],right["inputs"])
            self.assertEqual(left["summary"]["knowledge_sha256"],right["summary"]["knowledge_sha256"])

    def test_timely_control_identical(self):
        left,right = (self.runs["timely",a] for a in CONFIG["policies"])
        self.assertEqual(left["frames"],right["frames"])
        self.assertEqual(left["events"],right["events"])
        self.assertEqual(left["summary"]["completed"],4)
        self.assertEqual(left["summary"]["conflict_steps"],0)

    def test_queue_aware_one_unfinished_command_per_pad(self):
        for p in CONFIG["profiles"]:
            for frame in self.runs[p,"queue_aware"]["frames"]:
                self.assertTrue(all(len(ids)<=1 for ids in frame["unresolved_commands"].values()))

    def test_release_requires_evidence(self):
        for p in CONFIG["profiles"]:
            result=self.runs[p,"queue_aware"]
            for e in result["events"]:
                if e["event"]=="RELEASE":
                    self.assertEqual(e["reason"],"VERIFIED_COMPLETION")
                    self.assertLessEqual(e["t"]-e["observed_s"],CONFIG["observation_freshness_s"]+1e-8)
                    observed=next(f for f in result["frames"] if f["t"]==e["observed_s"])
                    d=e["command_id"].split("-")[1]
                    self.assertEqual(observed["drones"][d]["mode"],"FINISHED")
                    self.assertFalse(observed["occupancy"][e["pad"]])

    def test_metrics_from_geometry(self):
        for result in self.runs.values():
            steps=0
            for f in result["frames"]:
                counts={p:0 for p in CONFIG["knowledge"]["pads"]}
                for v in f["drones"].values():
                    if v["mode"]=="ON PAD":
                        self.assertEqual(v["xy_m"],CONFIG["knowledge"]["pads"][v["pad"]])
                        self.assertEqual(v["altitude_m"],0)
                        counts[v["pad"]]+=1
                expected=[p for p,count in counts.items() if count>1]
                self.assertEqual(f["conflict_pads"],expected)
                steps+=bool(expected)
            self.assertEqual(steps,result["summary"]["conflict_steps"])

    def test_all_frames_preserve_identity(self):
        for result in self.runs.values():
            for f in result["frames"]:
                self.assertEqual(set(f["drones"]),set("ABCD"))
                self.assertEqual(len({v["command_id"] for v in f["drones"].values()}),4)

    def test_old_observation_never_rolls_memory_back(self):
        for result in self.runs.values():
            stamps=[f["last_observed_s"] for f in result["frames"] if f["last_observed_s"] is not None]
            self.assertEqual(stamps,sorted(stamps))
            self.assertGreaterEqual(result["summary"]["rejected_old"],1)

    def test_outage_stalls_queue_aware(self):
        r=self.runs["telemetry_outage","queue_aware"]
        self.assertEqual(r["summary"]["completed"],2)
        self.assertEqual(r["summary"]["conflict_steps"],0)
        self.assertTrue(r["frames"][-1]["state"].startswith("HOLD"))
        self.assertEqual(r["frames"][-1]["waiting"],["C","D"])

    def test_no_reissue_or_duplicate_execution(self):
        for r in self.runs.values():
            starts=[e["command_id"] for e in r["events"] if e["event"]=="START"]
            self.assertEqual(len(starts),len(set(starts)))

    def test_full_sweep_reproduces_and_aggregates(self):
        rows=json.loads((OUT/"episode_summaries.json").read_text())
        self.assertEqual(len(rows),192)
        for row in rows:
            self.assertEqual(row,run(row["policy"],row["profile"],inputs(row["seed"]))["summary"])
        for agg in json.loads((OUT/"aggregates.json").read_text()):
            subset=[r for r in rows if all(r[k]==agg[k] for k in ("profile","policy"))]
            self.assertEqual(agg["n"],len(subset))
            self.assertEqual(agg["conflict_episodes"],sum(r["conflict_episode"] for r in subset))
            self.assertAlmostEqual(agg["mean_completed"],sum(r["completed"] for r in subset)/len(subset))


if __name__ == "__main__":
    unittest.main(verbosity=2)
