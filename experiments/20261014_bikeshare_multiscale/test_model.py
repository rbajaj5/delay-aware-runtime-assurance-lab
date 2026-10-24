import unittest
from unittest.mock import patch
import networkx as nx
from routing import detailed_graph,turn_allowed,construct,Router
from prepare import access,directions

class Tests(unittest.TestCase):
    def test_bicycle_access_precedence(self):
        self.assertIsNotNone(access({'highway':'footway'}))
        self.assertIsNone(access({'highway':'footway','bicycle':'yes'}))
        self.assertIsNotNone(access({'highway':'cycleway','bicycle':'no'}))
        self.assertIsNotNone(access({'highway':'residential','access:conditional':'no @ night'}))
    def test_directions(self):
        self.assertEqual(directions({'oneway':'yes','oneway:bicycle':'no'}),[1,-1])
        self.assertEqual(directions({'oneway':'-1'}),[-1])
    def test_turns_not_junction_deletion(self):
        es=[{'id':str(k),'u':u,'v':v,'way':w,'m':1,'crossing':None,'bridge':False} for k,(u,v,w) in enumerate([(1,2,10),(2,3,20),(2,4,30),(4,3,40),(2,1,10)])]
        net={'city':'fixture','nodes':{str(i):{'xy':[i,0]} for i in range(1,5)},'edges':es,'turn_rules':[{'via':2,'from':10,'to':[20],'value':'no_right_turn'}]}
        g=detailed_graph(net)
        with patch('routing.stations',return_value=[{'node':1},{'node':3},{'node':4}]):
            r=Router(net,g);self.assertEqual(r.route(1,3)['m'],3);self.assertEqual(r.route(1,4)['m'],2)
        self.assertFalse(turn_allowed({'way':10,'u':1},{'way':10,'v':1},[{'from':10,'to':[10],'value':'no_u_turn'}]))
        self.assertTrue(turn_allowed({'way':10,'u':1},{'way':10,'v':5},[{'from':10,'to':[10],'value':'no_u_turn'}]))
    def test_closure_checks_entry_not_departure_only(self):
        r=object.__new__(Router);r.edges={'a':{'m':10,'crossing':None},'b':{'m':10,'crossing':'B'}}
        end,legs,wait=r.itinerary({'edges':['a','b']},0,1,'B',5,20)
        self.assertEqual(end,30);self.assertEqual(wait,10);self.assertEqual(legs[1][1],20)
    def test_parallel_crossings_preserved(self):
        net={'nodes':{'1':{},'2':{}},'edges':[{'id':x,'u':1,'v':2,'way':i,'m':v,'crossing':x} for i,(x,v) in enumerate([('A',2),('B',3)])]}
        g=detailed_graph(net);self.assertEqual(g.number_of_edges(1,2),2)

if __name__=='__main__':unittest.main()
