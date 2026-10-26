"""Three executable, centralized queue guards. Discrete examples, not physics.

The resource authority serializes checks and effects; this is not a distributed
consensus or crash-recovery protocol. Memory on each agent can lag the authority.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass(frozen=True)
class Intent:
    agent: str
    identity: str
    expected_version: int
    target: str
    observed_at: float

class ResourceGuard:
    def __init__(self, state, allow, apply):
        self.state=dict(state)
        self.version=0
        self.seen={}
        self.allow,self.apply=allow,apply
        self.effects=0
    def serve(self, command, now):
        key=(command.agent,command.identity)
        if key in self.seen:
            return 'DUPLICATE_NO_EXECUTION' if self.seen[key]==command else 'REJECT_ID_COLLISION'
        if not 0 <= now-command.observed_at <= .75:
            return 'REJECT_STALE_OBSERVATION'
        if command.expected_version!=self.version:
            return 'REJECT_CHANGED_VERSION'
        if not self.allow(self.state,command):
            return 'REJECT_CURRENT_PRECONDITION'
        self.seen[key]=command
        self.apply(self.state,command)
        self.version+=1; self.effects+=1
        return 'EXECUTE'

def run():
    results=[]
    specs=[
        ('Basketball','Pass only to a currently open teammate',
         {'holder':'Guard','wing_open':True,'center_open':True},
         lambda s,c:s['holder']=='Guard' and s[c.target+'_open'],
         lambda s,c:s.update(holder=c.target.title()),
         Intent('Guard','pass-wing',0,'wing',0.),
         Intent('Guard','pass-center',0,'center',.6),
         'Wing was open when planned; it is covered at service time.'),
        ('Multi-agent gardening','One measured dose for one plant',
         {'plant_ml':0},lambda s,c:s['plant_ml']<60,
         lambda s,c:s.update(plant_ml=s['plant_ml']+60),
         Intent('Gardener A','water-plant',0,'plant',0.),
         Intent('Gardener B','water-plant',0,'plant',0.),
         'Both gardeners remember a dry plant; only one dose is authorized.'),
        ('Multi-agent laundry','One loader for one shared washer',
         {'loaded_by':None},lambda s,c:s['loaded_by'] is None,
         lambda s,c:s.update(loaded_by=c.agent),
         Intent('Loader A','load-washer',0,'washer',0.),
         Intent('Loader B','load-washer',0,'washer',0.),
         'Both loaders remember an empty washer; the second plan is obsolete.')]
    for title,rule,state,allow,apply,first,second,explanation in specs:
        guard=ResourceGuard(state,allow,apply)
        events=[]
        def take(command,time):
            status=guard.serve(command,time)
            events.append(dict(time_s=time,command=asdict(command),status=status,
                               state=dict(guard.state),version=guard.version,effects=guard.effects))
            return status
        if title=='Basketball':
            # The plan's remembered target status is no longer current when served.
            # Version tracks ball ownership; target coverage is rechecked separately.
            guard.state['wing_open']=False
            assert take(first,.4)=='REJECT_CURRENT_PRECONDITION'
            assert take(second,.7)=='EXECUTE'
            assert take(second,.8)=='DUPLICATE_NO_EXECUTION'
        else:
            assert take(first,.4)=='EXECUTE'
            assert take(second,.5)=='REJECT_CHANGED_VERSION'
            assert take(first,.6)=='DUPLICATE_NO_EXECUTION'
        assert guard.effects==1
        stale=Intent('Other','new-intent',guard.version,first.target,0.)
        assert take(stale,2.)=='REJECT_STALE_OBSERVATION'
        results.append(dict(title=title,rule=rule,explanation=explanation,
            initial=state,final=guard.state,events=events,effects=guard.effects,
            scope='Centralized discrete state machine; illustrative domain mapping; no physical simulation.'))
    # Identity collisions are distinct from harmless retries.
    g=ResourceGuard({'x':0},lambda s,c:True,lambda s,c:s.update(x=s['x']+1))
    a=Intent('A','1',0,'x',0.)
    assert g.serve(a,.1)=='EXECUTE'
    assert g.serve(Intent('A','1',0,'y',0.),.2)=='REJECT_ID_COLLISION'
    return results

if __name__=='__main__':
    rows=run()
    out=Path(__file__).parent/'evidence';out.mkdir(exist_ok=True)
    (out/'domain_examples.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('Passed: three domain examples; one effect each; duplicate, stale, conflict and precondition guards.')
