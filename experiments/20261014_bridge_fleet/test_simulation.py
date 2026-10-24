"""Small regression tests supplement the complete event-log reconstruction."""
import copy
import json
import unittest
from pathlib import Path

from simulate import make_tape,select_ready,simulate,verify_network,route


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.config=json.loads((Path(__file__).parent/'config.json').read_text())
    def test_historical_network_and_detour(self):verify_network(self.config)
    def test_arrival_order_fifo(self):
        packets=[{'seq':1,'created':0,'due':8,'kind':'road','key':'a','payload':False},
                 {'seq':2,'created':1,'due':2,'kind':'background','key':2,'payload':None}]
        chosen,_,_=select_ready(packets,'latest_received',1)
        self.assertEqual(chosen[0]['seq'],2)
    def test_priority_coalesces_only_arrived_snapshots(self):
        packets=[{'seq':1,'created':0,'due':1,'kind':'road','key':'a','payload':False},
                 {'seq':2,'created':2,'due':3,'kind':'road','key':'a','payload':True},
                 {'seq':3,'created':2,'due':3,'kind':'background','key':3,'payload':None}]
        chosen,left,discarded=select_ready(packets,'priority_updates',1)
        self.assertEqual(chosen[0]['seq'],2);self.assertEqual(discarded,1);self.assertEqual(left[0]['seq'],3)
    def test_freshness_has_a_cost_even_without_a_closure(self):
        known={e['id']:(False,0) for e in self.config['edges']}
        self.assertIsNotNone(route(self.config,'A','B',known,46))
        self.assertIsNone(route(self.config,'A','B',known,46,45))
    def test_reproducible_inputs_and_no_mutation(self):
        first=make_tape(self.config,self.config['seed_start']);second=make_tape(self.config,self.config['seed_start'])
        self.assertEqual(first,second);before=copy.deepcopy(first)
        for policy in self.config['policies']:
            row,_=simulate(self.config,self.config['scenarios'][1],policy,first)
            self.assertEqual(row['input_sha256'],first['sha256'])
        self.assertEqual(first,before)
    def test_no_closure_attempts_in_all_open_case(self):
        tape=make_tape(self.config,self.config['seed_start'])
        for policy in self.config['policies']:
            row,_=simulate(self.config,self.config['scenarios'][0],policy,tape)
            self.assertEqual(row['closed_entry_attempts'],0)
            self.assertEqual(row['forced_stop_vehicle_min'],0)


if __name__=='__main__':unittest.main()
