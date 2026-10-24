import cProfile,pstats
from simulate import *
city='washington';n=network(city);ss=stations(city);g,i=load_model(city,0);router=Router(n,g)
t=controlled_tape(city,C['seed_start'],router,ss)
p=cProfile.Profile();p.enable();r,d=simulate(n,ss,router,t,'B03','cargo_crew');p.disable()
print(r);pstats.Stats(p).sort_stats('cumtime').print_stats(20)
