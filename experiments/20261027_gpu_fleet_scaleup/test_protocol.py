"""Small task-protocol tests independent of GPU trajectories."""
import unittest

import numpy as np

from model import START, N, P
from simulate import CONFIG, Episode, tape


class ProtocolTests(unittest.TestCase):
    def episode(self, policy="queue_aware", profile="lost_ack"):
        episode = Episode(policy, profile, tape(45300, CONFIG), CONFIG)
        qpos = np.zeros((N, 7))
        qpos[:, :3] = START
        qpos[:, 3] = 1
        return episode, qpos.ravel(), np.zeros(N*6)

    def test_unresolved_commitments_distinguish_policies(self):
        for policy, expected_grants in (("timeout_only", P*2), ("queue_aware", P)):
            with self.subTest(policy=policy):
                e, q, v = self.episode(policy)
                for step in range(101):
                    e.tick(step, q, v)
                grants = [r for r in e.events if r["event"] == "GRANT"]
                self.assertEqual(len(grants), expected_grants)
                self.assertEqual(len({r["drone"] for r in grants}), expected_grants)

    def test_timely_ack_preserves_owner_in_both_arms(self):
        for policy in ("timeout_only", "queue_aware"):
            e, q, v = self.episode(policy, "timely")
            for step in range(101):
                e.tick(step, q, v)
            self.assertEqual([o["drone"] for o in e.owners], list(range(P)))
            self.assertTrue(all(o["ack"] for o in e.owners))

    def test_stale_evidence_blocks_new_clearances(self):
        e, q, v = self.episode(profile="telemetry_outage")
        e.latest = dict(observed_s=1., finished=[], clear=[True]*P)
        e.tick(100, q, v)
        self.assertEqual(e.waiting, list(range(N)))
        self.assertEqual(e.frames[-1]["decision"], "HOLD: STALE EVIDENCE")

    def test_out_of_order_observation_cannot_replace_newer_evidence(self):
        e, q, v = self.episode()
        e.latest = dict(observed_s=.6, finished=[], clear=[False]*P)
        e.send("OBSERVATION", .7, observed_s=.1, finished=[0, 1], clear=[True]*P)
        e.tick(40, q, v)
        self.assertEqual(e.latest["observed_s"], .6)
        self.assertEqual(e.waiting, list(range(N)))
        self.assertEqual(e.old_rejected, 1)

    def test_late_ack_does_not_acknowledge_a_different_owner(self):
        e, q, v = self.episode("timeout_only")
        e.owners[0] = dict(drone=P, issued_s=0., ack=False, command_id=f"land-{P}-001")
        e.send("ACK", 0., drone=0)
        e.tick(0, q, v)
        self.assertEqual(e.owners[0]["drone"], P)
        self.assertFalse(e.owners[0]["ack"])

    def test_release_requires_owner_completion_and_clear_pad(self):
        for finished, clear, released in (([], True, False), ([2], True, False),
                                         ([0], False, False), ([0], True, True)):
            with self.subTest(finished=finished, clear=clear):
                e, q, v = self.episode()
                e.waiting = []
                e.owners[0] = dict(drone=0, issued_s=0., ack=False, command_id="land-0-001")
                e.latest = dict(observed_s=0., finished=finished, clear=[clear]+[True]*(P-1))
                e.tick(0, q, v)
                self.assertEqual(e.owners[0] is None, released)

    def test_excessive_tilt_latches_disarm_in_both_arms(self):
        for policy in ("timeout_only", "queue_aware"):
            e, q, v = self.episode(policy)
            q[3:7] = [0., 1., 0., 0.]
            _, off = e.tick(0, q, v)
            self.assertTrue(off[0])
            self.assertEqual(e.mode[0], 5)
            q[3:7] = [1., 0., 0., 0.]
            _, off = e.tick(1, q, v)
            self.assertTrue(off[0])
            self.assertEqual(sum(a["event"] == "DISARM" for a in e.events), 1)


if __name__ == "__main__":
    unittest.main()
