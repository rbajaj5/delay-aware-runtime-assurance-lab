// Shared saved-event semantics used by the dashboard and its browser tests.
export function atTime(replay, net, t) {
  const inventory = [...replay.tape.initial];
  const ids = new Map(net.stations.map((s,i)=>[s.id,i]));
  let completed=0, failed=0, unreachable=0, onboard=0, cargo=0, full=0, redistributed=0;
  for (const e of replay.events) {
    if (e.time_s > t + 1e-6) break;
    if (['pickup','return','crew_load','crew_unload'].includes(e.event)) inventory[ids.get(e.station)] += e.quantity;
    if(e.event==='pickup')onboard++;
    if(e.event==='return'){onboard--;completed++;}
    if(e.event==='failed_pickup')failed++;
    if(e.event==='unreachable')unreachable++;
    if(e.event==='full_return')full++;
    if(e.event==='crew_load'||e.event==='crew_unload')cargo-=e.quantity;
    if(e.event==='crew_unload')redistributed+=e.quantity;
  }
  const attempted=replay.requests.filter(r=>r.time<=t).length;
  return {inventory,completed,failed,unreachable,onboard,cargo,full,redistributed,attempted,
    closed:replay.result.scenario.startsWith('B')&&t>=replay.tape.closure_start&&t<replay.tape.closure_end,
    fleet:inventory.reduce((a,b)=>a+b,0)+onboard+cargo};
}

export function requestState(r,t){
  if(r.time>t)return 'pending';
  if(r.picked_up_s===null)return r.status;
  if(r.completed_s!==null&&r.completed_s<=t)return 'completed';
  return 'active';
}

export function position(m,replay,net,t){
  if(t<m.start||t>m.arrive)return null;
  const path=replay.paths[m.path];
  let cursor=m.start, wi=0;
  for(let j=0;j<path.length;j++){
    const e=net.edges[path[j]], a=net.nodes[e[0]], b=net.nodes[e[1]];
    if(wi<m.waits.length&&m.waits[wi][0]===j){
      cursor+=m.waits[wi++][1];if(t<cursor)return a;
    }
    const finish=cursor+e[2]/m.speed;
    if(t<=finish){const f=Math.max(0,Math.min(1,(t-cursor)/(finish-cursor)));return [a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f];}
    cursor=finish;
  }
  return path.length?net.nodes[net.edges[path.at(-1)][1]]:null;
}
