import unittest
from scenario_runner import validate, saved_match

class RunnerTests(unittest.TestCase):
    def args(self):return dict(city='washington',workload='controlled',seed=2026101400,policy='no_redistribution',scale_m=200,intervention='I1',closure_start_s=3600,closure_duration_s=5400,initial_bikes=8)
    def test_exact_saved_match(self):
        config,tape=validate(self.args());result=saved_match(config,tape)
        self.assertEqual(result['origin'],'exact_saved_match')
        self.assertEqual(result['catalog'][0]['input_sha256'],result['catalog'][1]['input_sha256'])
    def test_changed_timing_requires_new_run(self):
        a=self.args();a.update(intervention='B04',closure_duration_s=2000)
        c,t=validate(a);self.assertIsNone(saved_match(c,t))
    def test_reject_combination(self):
        a=self.args();a['intervention']='I1+I2'
        with self.assertRaises(ValueError):validate(a)
    def test_inventory_bounds(self):
        a=self.args();a['initial_bikes']=17
        with self.assertRaises(ValueError):validate(a)
    def test_unknown_parameters(self):
        a=self.args();a['demand_multiplier']=3
        with self.assertRaises(ValueError):validate(a)
    def test_closure_outside_horizon(self):
        a=self.args();a.update(closure_start_s=14000,closure_duration_s=1000)
        with self.assertRaises(ValueError):validate(a)

if __name__=='__main__':unittest.main()
