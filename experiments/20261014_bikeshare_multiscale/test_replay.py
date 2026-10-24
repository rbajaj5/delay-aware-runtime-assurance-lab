import unittest
from corrected_model import CorrectedRouter
from routing import Router


class BaselineRegression(unittest.TestCase):
    def router(self, cls):
        r = object.__new__(cls)
        r.edges = {'street': {'m': 40, 'crossing': None},
                   'bridge': {'m': 80, 'crossing': 'B01'}}
        return r

    def test_open_street_has_no_closure_wait(self):
        r = self.router(CorrectedRouter)
        arrival, legs, wait = r.itinerary({'edges': ['street']}, 4000, 4, None, 3600, 9000)
        self.assertEqual((arrival, wait), (4010, 0))

    def test_retained_bug_is_reproducible(self):
        r = self.router(Router)
        self.assertEqual(r.itinerary({'edges': ['street']}, 4000, 4, None, 3600, 9000)[2], 5000)

    def test_real_closure_wait_is_preserved(self):
        for cls in (Router, CorrectedRouter):
            r = self.router(cls)
            arrival, legs, wait = r.itinerary({'edges': ['street', 'bridge']}, 4000, 4, 'B01', 3600, 9000)
            self.assertEqual((arrival, wait), (9020, 4990))


if __name__ == '__main__':
    unittest.main()
