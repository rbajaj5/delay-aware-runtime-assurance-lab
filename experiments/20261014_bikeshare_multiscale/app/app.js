import {atTime, requestState, position} from './domain.js';
const $=id=>document.getElementById(id), val=id=>$(id).value;
const colors={left:'#3376bd',right:'#0b7a70',red:'#c44b3c',purple:'#9a47a1',amber:'#bd8116'};
const cache=new Map(), backgrounds=new Map();
let report,catalog, pair=[],nets=[],t=0,playing=false,station=0,route='',zoom=1,pan=[0,0],loadId=0,last=0,tableTime=-1,rankHits=[];
let bounds,ready=false;
let labMode='saved',customRun=null,affected=new Set(),jobToken=0;
const num=(v,d=1)=>v===null||v===undefined?'n/a':Number(v).toFixed(d);
const sign=(v,d=1)=>(v>0?'+':'')+num(v,d);
const scaleName=s=>Number(s)?`${s} m`:'Detailed';
const clock=s=>[Math.floor(s/3600),Math.floor(s%3600/60),Math.floor(s%60)].map(v=>String(v).padStart(2,'0')).join(':');
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
async function json(url){if(!cache.has(url))cache.set(url,fetch(url).then(r=>{if(!r.ok)throw new Error(`${url}: ${r.status}`);return r.json();}));return cache.get(url);}
function options(id,items,keep=true){const before=val(id);$(id).innerHTML=items.map(([v,label])=>`<option value="${escape(v)}">${escape(label)}</option>`).join('');if(keep&&items.some(([v])=>String(v)===before))$(id).value=before;}
function refreshOptions(){
  if(val('city')==='kaliningrad')$('workload').value='controlled';
  $('workload').options[1].disabled=val('city')==='kaliningrad';
  const candidates=catalog.filter(r=>r.city===val('city')&&r.workload===val('workload'));
  const seeds=[...new Map(candidates.map(r=>[r.seed,r.day||`Seed ${r.seed}`])).entries()].sort((a,b)=>a[0]-b[0]);options('seed',seeds);
  if(labMode==='new'){
    if(![...$('comparison').options].some(o=>o.value==='custom'))$('comparison').add(new Option('Baseline vs configured scenario','custom'));
    $('comparison').value='custom';
    const bridges=report.network_audit?catalog.filter(r=>r.city===val('city')&&r.scenario.startsWith('B')).map(r=>r.scenario):[];
    const extra=(window.allCrossings?.[val('city')]||[]).map(b=>b.id);
    options('scenario',[['all_open','Existing network / all open'],...report.interventions[val('city')].map(s=>[s.id,`${s.id} · ${s.name}`]),...[...new Set([...bridges,...extra])].sort().map(n=>[n,`${n} · existing crossing closed`])]);
    return;
  }
  const retained=val('evidence')==='retained';
  if(retained)$('comparison').value='recovery';
  $('comparison').disabled=retained;
  if(val('comparison')==='recovery'){
    const names=[...new Set(candidates.filter(r=>r.scenario.startsWith('B')).map(r=>r.scenario))].sort();
    options('scenario',names.map(n=>[n,`${n} closure / reopening`]));
  }else options('scenario',[['all_open','Existing network / all open'],...report.interventions[val('city')].map(s=>[s.id,`${s.id} · ${s.name}`])]);
}
function entry(version,scenario,scale){const r=catalog.find(r=>r.version===version&&r.city===val('city')&&r.workload===val('workload')&&r.seed===+val('seed')&&r.policy===val('policy')&&r.scenario===scenario&&r.scale_m===+scale);if(!r)throw new Error('No saved episode for selected controls');return r;}
async function load(){
  if(labMode==='new'){refreshOptions();invalidate();return;}
  const serial=++loadId;ready=false;playing=false;document.body.classList.add('loading');$('notice').textContent='Loading complete saved-run journeys...';
  try{
    refreshOptions();const scale=+val('scale'),sc=val('scenario'),mode=val('comparison'),old=val('evidence')==='retained';
    let entries,titles;
    customRun=null;
    if(!$('enabled').checked||sc==='all_open'){entries=[entry('corrected','all_open',scale),entry('corrected','all_open',scale)];titles=['Existing network','Existing network / no change'];}
    else if(old){entries=[entry('retained','all_open',scale),entry('retained',sc,scale)];titles=['Retained all-open (known bug)',`Retained closure ${sc}`];}
    else if(mode==='investment'){entries=[entry('corrected','all_open',scale),entry('investment',sc,scale)];titles=['Existing network',`Hypothetical ${sc}`];}
    else if(mode==='scale'){entries=[entry('investment',sc,0),entry('investment',sc,scale)];titles=[`Hypothetical ${sc} · detailed`,`Hypothetical ${sc} · ${scaleName(scale)}`];}
    else{entries=[entry('corrected',sc,scale),entry('corrected','all_open',scale)];titles=[`${sc} disrupted network`,'Existing crossing reopened'];}
    const data=await Promise.all(entries.map(e=>json('/'+e.file)));
    const maps=await Promise.all(entries.map(e=>json(`/app/data/${e.network||e.city}.json.gz`)));
    if(serial!==loadId)return;
    pair=data;nets=maps;pair.forEach((r,i)=>r.displayResult=entries[i]);
    $('closureStart').value=data[0].tape.closure_start/60;$('closureDuration').value=(data[0].tape.closure_end-data[0].tape.closure_start)/60;$('initialBikes').value=8;
    if(data[0].tape.sha256!==data[1].tape.sha256)throw new Error('Unmatched input tapes');
    t=Math.min(t,data[0].tape.horizon);$('time').max=data[0].tape.horizon;
    for(let i=0;i<2;i++){const side=i?'right':'left';$(side+'Title').textContent=titles[i];$(side+'Source').textContent=scaleName(entries[i].scale_m)+' · '+(entries[i].day||entries[i].seed);}
    station=Math.min(station,nets[0].stations.length-1);route='';options('route',[['','No selected request'],...data[1].requests.map(r=>[r.id,`Request ${r.id} · ${clock(r.time)}`])],false);
    fit();tableTime=-1;ready=true;document.body.classList.remove('loading','uncomputed');findAffected();
    $('notice').className=old?'warning':'';
    $('notice').textContent=old?'AUDIT ONLY: these original all-open episodes contain the documented no-closure timing bug. Not used to rank infrastructure.':'Saved-run replay · matched request tape '+data[0].tape.sha256.slice(0,12)+' · all requests and journeys retained · no new scenario calculation';
    description();renderAnalysis();render();
  }catch(error){document.body.classList.remove('loading');$('notice').className='warning';$('notice').textContent='Unable to load verified data: '+error.message;console.error(error);}
}
function fit(){if(!nets.length)return;const ss=nets[0].stations;const x=ss.map(s=>s.xy[0]),y=ss.map(s=>s.xy[1]);bounds=[Math.min(...x)-350,Math.min(...y)-350,Math.max(...x)+350,Math.max(...y)+350];zoom=1;pan=[0,0];backgrounds.clear();}
function ctx(id){const c=$(id),r=c.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);const w=Math.max(1,r.width),h=Math.max(1,r.height);if(c.width!==Math.round(w*dpr)||c.height!==Math.round(h*dpr)){c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);}const g=c.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);g.font='11px Segoe UI';return {c,g,w,h,dpr};}
function projector(w,h){const [x0,y0,x1,y1]=bounds,s=Math.min((w-25)/(x1-x0),(h-25)/(y1-y0))*zoom,cx=(x0+x1)/2,cy=(y0+y1)/2;return p=>[w/2+(p[0]-cx)*s+pan[0],h/2-(p[1]-cy)*s+pan[1]];}
function stroke(g,points,color,width){if(!points.length)return;g.strokeStyle=color;g.lineWidth=width;g.beginPath();points.forEach((p,i)=>i?g.lineTo(...p):g.moveTo(...p));g.stroke();}
function baseMap(net,w,h){const key=[net.city,net.edges.length,w,h,zoom,...pan].join('|');if(backgrounds.has(key))return backgrounds.get(key);const c=document.createElement('canvas');c.width=Math.ceil(w);c.height=Math.ceil(h);const g=c.getContext('2d'),p=projector(w,h);g.fillStyle='#f1f5f4';g.fillRect(0,0,w,h);
  for(const water of net.water){const points=water.xy.map(p);g.beginPath();points.forEach((v,i)=>i?g.lineTo(...v):g.moveTo(...v));if(water.kind==='polygon'){g.fillStyle='#cbe2e8';g.fill();}else{g.strokeStyle='#9bbfce';g.lineWidth=2;g.stroke();}}
  g.strokeStyle='#c5d0d0';g.lineWidth=.55;g.beginPath();for(const e of net.edges){const a=p(net.nodes[e[0]]),b=p(net.nodes[e[1]]);g.moveTo(...a);g.lineTo(...b);}g.stroke();
  g.font='10px Segoe UI';for(const b of net.crossings){const at=p(b.xy);g.fillStyle=b.hypothetical?colors.purple:'#62797e';g.fillText(b.id,...at);}
  g.font='10px Segoe UI';g.fillStyle='#61777d';g.fillText('N ↑',w-35,20);g.fillText('OSM · buffered study area',10,h-9);
  if(backgrounds.size>15)backgrounds.clear();backgrounds.set(key,c);return c;
}
function drawMap(index,state){const id=index?'rightMap':'leftMap',{g,w,h}=ctx(id),net=nets[index],r=pair[index],p=projector(w,h);g.drawImage(baseMap(net,w,h),0,0,w,h);
  for(const e of net.edges){if((state.closed&&e[3]===r.result.scenario)||(e[3]&&e[3].startsWith('H'))){g.setLineDash(e[3].startsWith('H')?[6,3]:[]);stroke(g,[p(net.nodes[e[0]]),p(net.nodes[e[1]])],e[3].startsWith('H')?colors.purple:colors.red,3);}}g.setLineDash([]);
  if($('groups').checked){const group=net.groups[String(r.result.scale_m)]||[];const bins=new Map();net.stations.forEach((s,i)=>{const k=group[i];if(!bins.has(k))bins.set(k,[]);bins.get(k).push(s.xy);});for(const points of bins.values()){if(points.length<2)continue;const mid=[points.reduce((a,b)=>a+b[0],0)/points.length,points.reduce((a,b)=>a+b[1],0)/points.length];for(const a of points)stroke(g,[p(a),p(mid)],'#c49d1a',2);}}
  if(route!==''){g.globalAlpha=.75;for(const m of r.movements.filter(m=>String(m.id)===route)){const path=r.paths[m.path];for(const k of path){const e=net.edges[k];stroke(g,[p(net.nodes[e[0]]),p(net.nodes[e[1]])],index?colors.purple:colors.left,3);}}g.globalAlpha=1;}
  for(const m of r.movements){const at=position(m,r,net,t);if(!at)continue;const [x,y]=p(at);g.fillStyle=m.kind==='crew'?'#a45b11':colors.left;g.beginPath();if(m.kind==='crew')g.rect(x-3,y-3,6,6);else g.arc(x,y,String(m.id)===route?4.5:2.4,0,Math.PI*2);g.fill();}
  net.stations.forEach((s,i)=>{const [x,y]=p(s.xy),v=state.inventory[i];g.fillStyle=v===0?colors.red:v===s.capacity?colors.amber:colors.right;g.strokeStyle='#fff';g.lineWidth=1.3;g.beginPath();g.arc(x,y,i===station?8:5.5,0,Math.PI*2);g.fill();g.stroke();if(i===station){g.strokeStyle='#243237';g.beginPath();g.arc(x,y,10,0,Math.PI*2);g.stroke();}g.font='10px Segoe UI';g.fillStyle='#30464d';g.fillText(String(i+1).padStart(2,'0'),x+9,y+3);});
}
function metrics(side,s,r){$(side+'Metrics').innerHTML=[['Completed',s.completed,`final ${r.result.completed}`],['On bikes',s.onboard,`cargo ${s.cargo}`],['Failed pickup',s.failed,`final ${r.result.failed_pickups}`],['Full returns',s.full,`final ${r.result.full_station_events}`],['Bikes conserved',s.fleet,`of ${r.result.fleet}`]].map(([a,b,c])=>`<div class="metric"><span>${a}</span><b>${b}</b><small>${c}</small></div>`).join('');}
function chart(id,series,maxY,unit,labels){const {g,w,h}=ctx(id),pad={l:42,r:15,t:17,b:26},ww=w-pad.l-pad.r,hh=h-pad.t-pad.b;maxY=Math.max(maxY,1);const x=v=>pad.l+ww*v/pair[0].tape.horizon,y=v=>pad.t+hh*(1-v/maxY);g.fillStyle='#60787e';g.font='10px Segoe UI';for(let i=0;i<=4;i++){let yy=y(maxY*i/4);stroke(g,[[pad.l,yy],[w-pad.r,yy]],'#e0e8e9',1);g.fillText(num(maxY*i/4,0),3,yy+3);}for(let j=0;j<series.length;j++)stroke(g,series[j].map(v=>[x(v[0]),y(v[1])]),j?colors.right:colors.left,2);stroke(g,[[x(t),pad.t],[x(t),h-pad.b]],'#b4593e',1.5);g.fillText('0 h',pad.l,h-7);g.fillText(`${pair[0].tape.horizon/3600} h`,w-35,h-7);g.fillText(unit,3,10);if(labels)labels.forEach((label,i)=>{g.fillStyle=i?colors.right:colors.left;g.fillText(label,pad.l+10+i*130,13);});}
function drawTimelines(){chart('timeline',pair.map(r=>r.frames.map(f=>[f.t,f.completed])),Math.max(...pair.map(r=>r.result.completed)),'trips',['Left','Right']);const s=nets[0].stations[station];$('inventoryTitle').textContent=`Station ${String(station+1).padStart(2,'0')} · ${s.name}`;chart('inventoryPlot',pair.map(r=>r.frames.map(f=>[f.t,f.inventory[station]])),s.capacity,'bikes',['Left','Right']);}
function tables(states){$('stations').innerHTML='<thead><tr><th>Station</th><th>Left</th><th>Right</th><th>Capacity</th></tr></thead><tbody>'+nets[0].stations.map((s,i)=>`<tr data-station="${i}" class="${i===station?'selected':''}"><td>${String(i+1).padStart(2,'0')} · ${escape(s.name)}</td><td>${states[0].inventory[i]}</td><td>${states[1].inventory[i]}</td><td>${s.capacity}</td></tr>`).join('')+'</tbody>';
  const filter=val('journeyFilter'),rs=pair[1].requests.filter(r=>{const status=requestState(r,t);return status!=='pending'&&(filter==='all'||status===filter||(filter==='affected'&&affected.has(r.id)));}).slice(-60).reverse();
  $('journeys').innerHTML='<thead><tr><th>ID</th><th>Origin → destination</th><th>Status now</th><th>Request time</th></tr></thead><tbody>'+rs.map(r=>`<tr data-route="${r.id}" class="${String(r.id)===route?'selected':''}"><td>${r.id}</td><td>${escape(r.origin)} → ${escape(r.destination)}</td><td>${requestState(r,t).replaceAll('_',' ')}</td><td>${clock(r.time)}</td></tr>`).join('')+'</tbody>';
  const r=pair[1].requests.find(r=>String(r.id)===route);$('journeyDetail').textContent=r?`Request ${r.id}: ${requestState(r,t).replaceAll('_',' ')}. Final logged outcome: ${r.status.replaceAll('_',' ')}; routed ${num(r.distance_m,0)} m; closure wait ${num(r.closure_wait_s,0)} s. Blue path shows all planned legs for this request.`:`${rs.length} most recent matching requests shown for the right-hand episode. Entire episode is available in the request export.`;
}
function description(){const s=report.interventions[val('city')].find(s=>s.id===val('scenario'));if(!$('enabled').checked||val('scenario')==='all_open'){$('description').textContent='Existing network on both maps. No added or closed connection. The ranking below describes the completed candidate study, not an effect in these identical replays.';return;}if(val('comparison')==='recovery'){$('description').textContent=`Recovery comparison: reopening ${val('scenario')} against its disrupted baseline. Closure applies ${clock(pair[0].tape.closure_start)}–${clock(pair[0].tape.closure_end)}. Recovery gains are not investment gains. ${val('evidence')==='retained'?'Retained audit view only; original all-open bug is present.':''}`;return;}
  $('description').textContent=`${s.name}: hypothetical ${num(s.length_m,0)} m bidirectional cycling link. Endpoints ${s.endpoints.map(p=>`${num(p.latitude,5)}, ${num(p.longitude,5)}`).join(' ↔ ')}. Cycling 4 m/s × saved rider factor; cargo 3 m/s. No engineering feasibility claim. ${val('comparison')==='scale'?'Both maps contain this same investment; only routing scale changes.':'Same requests, bikes, capacities and policy in both arms.'}`;
}
function filteredBenefits(){const kind=val('comparison')==='recovery'?'recovery':'investment';return report.benefits.filter(r=>r.city===val('city')&&r.workload===val('workload')&&r.policy===val('policy')&&r.comparison===kind);}
function renderAnalysis(){if(customRun){renderCustomAnalysis();return;}const all=filteredBenefits(),scale=+val('scale'),data=all.filter(r=>r.scale_m===scale).sort((a,b)=>b.additional_completed-a.additional_completed||a.intervention.localeCompare(b.intervention));const kind=val('comparison')==='recovery'?'recovery':'investment';
  $('rankingTitle').textContent=kind==='recovery'?'Crossing reopenings ranked by additional completed bike trips':report.title;
  $('rankN').textContent=`${data[0]?.n||0} matched ${val('workload')==='controlled'?'seeds':'days'} · ${scaleName(scale)}`;
  const {g,w,h}=ctx('ranking'),left=Math.min(w*.43,220),right=65,ww=w-left-right;const lo=Math.min(0,...data.map(d=>d.additional_completed_p10)),hi=Math.max(1,...data.map(d=>d.additional_completed_p90)),span=hi-lo||1,x=v=>left+(v-lo)/span*ww;rankHits=[];
  stroke(g,[[x(0),10],[x(0),h-25]],'#aebfc4',1);
  const rowHeight=(h-30)/3;
  data.forEach((d,i)=>{const yy=8+(i+.5)*rowHeight;rankHits.push({top:yy-rowHeight/2,bottom:yy+rowHeight/2,id:d.intervention});g.fillStyle=d.intervention===val('scenario')?'#eaf3ef':'#fafbfb';g.fillRect(0,yy-rowHeight/2+2,w,rowHeight-4);g.font='12px Segoe UI';g.fillStyle='#2f474f';const name=report.interventions[val('city')].find(s=>s.id===d.intervention)?.name||`Reopen ${d.intervention}`;g.fillText(`${d.intervention} · ${name.length>27?name.slice(0,25)+'…':name}`,4,yy+4);g.fillStyle=d.additional_completed>=0?colors.right:colors.red;g.fillRect(Math.min(x(0),x(d.additional_completed)),yy-8,Math.max(2,Math.abs(x(d.additional_completed)-x(0))),16);stroke(g,[[x(d.additional_completed_p10),yy],[x(d.additional_completed_p90),yy]],'#243f47',1.5);for(const v of [d.additional_completed_p10,d.additional_completed_p90])stroke(g,[[x(v),yy-5],[x(v),yy+5]],'#243f47',1.5);g.fillStyle='#243f47';g.fillText(sign(d.additional_completed),w-54,yy+4);});g.fillStyle='#627b80';g.font='10px Segoe UI';g.fillText(`${num(lo)} trips`,left,h-8);g.fillText(`${num(hi)} trips`,w-right-40,h-8);
  $('rankingNote').textContent=`Mean additional completed trips; whiskers are p10–p90 across ${data[0]?.n||0} matched runs, not a confidence interval. ${val('workload')==='observed_week'?'Only seven observed days; many tied effects. ':''}${val('evidence')==='retained'?'Chart still uses the corrected study, not the buggy audit baseline.':''}`;
  $('tradeoffs').innerHTML='<thead><tr><th>Change</th><th>Trips</th><th>Served pp</th><th>Time s*</th><th>Empty h</th><th>Full events</th></tr></thead><tbody>'+data.map(d=>`<tr data-scenario="${d.intervention}"><td>${d.intervention}</td><td>${sign(d.additional_completed)}</td><td>${sign(d.served_change_pp,2)}</td><td>${sign(d.common_completed_journey_change_s)}</td><td>${sign(d.empty_station_hours_change,2)}</td><td>${sign(d.full_events_change)}</td></tr>`).join('')+'</tbody>';
  const selected=all.filter(r=>r.intervention===(val('scenario')==='all_open'?'I1':val('scenario'))).sort((a,b)=>a.scale_m-b.scale_m);const sc=ctx('scalePlot'),q=sc.g,xx=i=>45+i*(sc.w-80)/3;const ymin=Math.min(0,...selected.map(d=>d.additional_completed)),ymax=Math.max(1,...selected.map(d=>d.additional_completed)),yy=v=>25+(sc.h-65)*(1-(v-ymin)/(ymax-ymin));
  for(let i=0;i<=4;i++){let v=ymin+(ymax-ymin)*i/4;stroke(q,[[42,yy(v)],[sc.w-20,yy(v)]],'#e0e7e8',1);q.fillStyle='#5e777d';q.fillText(num(v),2,yy(v)+3);}stroke(q,selected.map((d,i)=>[xx(i),yy(d.additional_completed)]),colors.right,2);selected.forEach((d,i)=>{q.fillStyle=d.scale_m===scale?colors.red:colors.right;q.beginPath();q.arc(xx(i),yy(d.additional_completed),5,0,Math.PI*2);q.fill();q.fillStyle='#60787e';q.fillText(scaleName(d.scale_m),xx(i)-20,sc.h-12);});
  const ranks=report.rankings.filter(r=>r.city===val('city')&&r.workload===val('workload')&&r.policy===val('policy')&&r.comparison===kind).sort((a,b)=>a.scale_m-b.scale_m);
  $('scaleTable').innerHTML='<thead><tr><th>Scale</th><th>Preferred set</th><th>Same as detailed?</th><th>Largest mean benefit error</th></tr></thead><tbody>'+ranks.map(r=>`<tr><td>${scaleName(r.scale_m)}</td><td>${r.reduced_top}</td><td>${r.same_preferred_set?'Yes':'No'}</td><td>${num(r.max_mean_benefit_error_trips)} trips</td></tr>`).join('')+'</tbody>';
  const cost=report.costs.find(r=>r.city===val('city')&&r.intervention===val('scenario')&&r.scale_m===scale),base=report.costs.find(r=>r.city===val('city')&&r.intervention===val('scenario')&&r.scale_m===0);
  const d=data.find(r=>r.intervention===val('scenario'));
  $('costNote').textContent=`*Travel time: same request IDs completed in both arms; mean cohort ${num(d?.common_completed_n,0)}. Excludes failed/unfinished requests. ${cost?`Graph ${num(cost.nodes,0)} nodes vs ${num(base.nodes,0)} detailed; cold routing ${num(cost.cold_routing_s,2)} s vs ${num(base.cold_routing_s,2)} s; rebuild ${num(cost.build_s,2)} s. Single wall-clock measurements, not hardware-normalized benchmarks.`:''} ${d?`Unserved ${num(d.baseline_unserved)} → ${num(d.intervention_unserved)}; unfinished ${num(d.baseline_unfinished)} → ${num(d.intervention_unfinished)}. Cargo distance change ${sign(d.crew_distance_km_change,2)} km; redistributed bikes ${sign(d.redistributed_bikes_change)}.`:''}`;
}
function render(){if(!ready)return;const states=pair.map((r,i)=>atTime(r,nets[i],t));for(let i=0;i<2;i++){drawMap(i,states[i]);metrics(i?'right':'left',states[i],pair[i]);}drawTimelines();$('clock').textContent=clock(t);$('time').value=t;$('clockNote').textContent=states.some(s=>s.closed)?'Crossing closure active':'Crossings available';if(Math.floor(t)!==tableTime){tables(states);tableTime=Math.floor(t);}window.lastRendered={t,states,episodes:pair.map(r=>r.episode)};}
function setTime(v){t=Math.min(Math.max(0,v),pair[0]?.tape.horizon||14400);render();}
function findAffected(){
  affected=new Set();const maps=pair.map(r=>new Map(r.requests.map(q=>[q.id,q])));
  const signatures=pair.map(r=>{const result=new Map();for(const m of r.movements){if(m.kind!=='rider')continue;const str=JSON.stringify(r.paths[m.path]);result.set(m.id,(result.get(m.id)||'')+str);}return result;});
  for(const [id,b]of maps[1]){const a=maps[0].get(id);if(!a)continue;if(signatures[0].get(id)!==signatures[1].get(id)||a.status!==b.status||Math.abs((a.completed_s||0)-(b.completed_s||0))>.01)affected.add(id);}
  $('affectedNote').textContent=`${affected.size} request IDs have different routed paths, completion times, or outcomes between these two episodes. Selected paths: left blue, right purple. These are consequences in the model, not additional real-world demand.`;
}
function invalidate(message='Configuration changed. Run the matched pair to calculate these settings.'){
  ready=false;playing=false;customRun=null;jobToken++;document.body.classList.add('uncomputed');document.body.classList.remove('loading');
  $('notice').className='warning';$('notice').textContent=message;$('leftTitle').textContent='Baseline not yet calculated';$('rightTitle').textContent='Intervention not yet calculated';$('leftSource').textContent='Pending';$('rightSource').textContent='Pending';
  $('runScenario').disabled=false;$('configDownload').hidden=true;$('affectedNote').textContent='No previous trajectories or results are displayed under these new settings.';
  $('description').textContent='One new matched pair. The source requests and speed factors are fixed; closure timing and starting inventory apply to both arms. Cargo dispatch thresholds remain 3 / 11 bikes with target inventory 8.';
}
async function setMode(mode){
  labMode=mode;jobToken++;$('savedMode').classList.toggle('active',mode==='saved');$('newMode').classList.toggle('active',mode==='new');$('runScenario').hidden=mode!=='new';
  for(const id of ['closureStart','closureDuration','initialBikes'])$(id).readOnly=mode==='saved';
  $('evidence').disabled=mode==='new';$('comparison').disabled=mode==='new';$('modeNote').textContent=mode==='saved'?'Complete saved journeys, not a new calculation':'Bounded matched-pair runner / one topology change / one saved demand tape';
  if(mode==='new'){
    invalidate('Configure a matched scenario. No previous trajectories are displayed.');
    window.allCrossings={};for(const city of ['washington','kaliningrad'])window.allCrossings[city]=(await json(`/app/data/${city}.json.gz`)).crossings;
    if(labMode!=='new')return;
    $('evidence').value='corrected';refreshOptions();invalidate();
  }else{customRun=null;if(val('comparison')==='custom')$('comparison').value='investment';const customOption=[...$('comparison').options].find(o=>o.value==='custom');if(customOption)customOption.remove();document.body.classList.remove('uncomputed');$('configDownload').hidden=true;await load();}
}
function renderCustomAnalysis(){
  const m=customRun.metrics,a=pair[0].result,b=pair[1].result;
  $('rankingTitle').textContent='New scenario: one matched comparison';$('rankN').textContent='n = 1 seed pair';
  const {g,w,h}=ctx('ranking'),max=Math.max(1,a.completed,b.completed),x=110,ww=w-165;
  [['Baseline',a.completed,colors.left],['Intervention',b.completed,colors.right]].forEach(([label,value,color],i)=>{g.fillStyle='#334a51';g.font='13px Segoe UI';g.fillText(label,3,65+i*65);g.fillStyle=color;g.fillRect(x,45+i*65,ww*value/max,28);g.fillStyle='#243237';g.fillText(value,w-45,65+i*65);});
  $('rankingNote').textContent=`${sign(m.additional_completed,0)} completed trips; ${sign(m.served_change_pp,2)} percentage points. This is one seed, not a mean effect. No uncertainty interval or multi-candidate ranking is inferred.`;
  $('tradeoffs').innerHTML=`<thead><tr><th>Measure</th><th>Baseline</th><th>Intervention</th></tr></thead><tbody><tr><td>Unserved requests</td><td>${m.baseline_unserved}</td><td>${m.intervention_unserved}</td></tr><tr><td>Unfinished requests</td><td>${m.baseline_unfinished}</td><td>${m.intervention_unfinished}</td></tr><tr><td>Empty-station hours</td><td>${num(a.empty_station_hours,2)}</td><td>${num(b.empty_station_hours,2)}</td></tr></tbody>`;
  const sc=ctx('scalePlot');sc.g.fillStyle='#5a7279';sc.g.font='13px Segoe UI';sc.g.fillText('Only the selected routing resolution has been calculated.',14,70);sc.g.fillText('Other resolutions and seed variability are not inferred.',14,100);$('scaleTable').innerHTML='';
  $('costNote').textContent=`Travel-time change: ${sign(m.common_completed_time_change_s)} s among ${m.common_completed_n} request IDs completed in both arms. Full-return events: ${sign(m.full_events_change,0)}. Cargo distance: ${sign(m.crew_distance_km_change,2)} km. Complete settings and provenance are in Configuration JSON.`;
}
async function runScenario(){
  const horizon=val('workload')==='observed_week'?86400:14400;
  const config={city:val('city'),workload:val('workload'),seed:+val('seed'),policy:val('policy'),scale_m:+val('scale'),intervention:$('enabled').checked?val('scenario'):'all_open',closure_start_s:+val('closureStart')*60,closure_duration_s:+val('closureDuration')*60,initial_bikes:+val('initialBikes')};
  invalidate('Pending: calculating baseline and intervention under the same configuration.');const token=jobToken;$('runScenario').disabled=true;
  try{
    let response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)});let result=await response.json();if(!response.ok||result.error)throw new Error(result.error||'Scenario request failed');
    while(result.status==='queued'||result.status==='running'){
      if(token!==jobToken)return;$('notice').textContent=result.message+' · '+result.id;await new Promise(r=>setTimeout(r,600));result=await (await fetch('/api/jobs/'+result.id)).json();
    }
    if(token!==jobToken)return;if(result.status!=='complete')throw new Error(result.message||result.error||'Calculation failed');
    const data=await Promise.all(result.catalog.map(e=>json('/'+e.file))),maps=await Promise.all(result.catalog.map(e=>json(`/app/data/${e.network||e.city}.json.gz`)));
    if(token!==jobToken)return;pair=data;nets=maps;customRun=result;t=Math.min(t,horizon);$('time').max=horizon;
    $('leftTitle').textContent='Matched existing-network baseline';$('rightTitle').textContent=result.config.intervention==='all_open'?'Existing network / no change':result.config.intervention+' / newly configured';
    $('leftSource').textContent=scaleName(config.scale_m);$('rightSource').textContent=result.origin==='exact_saved_match'?'Exact saved match':'New calculation, n = 1';
    options('route',[['','No selected request'],...data[1].requests.map(r=>[r.id,`Request ${r.id} · ${clock(r.time)}`])],false);route='';station=0;fit();findAffected();tableTime=-1;ready=true;document.body.classList.remove('uncomputed');
    $('notice').className='';$('notice').textContent=(result.origin==='exact_saved_match'?'Exactly matching saved experiments loaded. ':result.origin==='cached_custom_calculation'?'Previously calculated custom configuration loaded. ':'New matched pair calculated and event-audited. ')+result.coverage;
    $('description').textContent=`Configuration: ${config.city}; ${config.intervention}; ${config.initial_bikes} starting bikes per station; ${scaleName(config.scale_m)}; closure ${clock(config.closure_start_s)}–${clock(config.closure_start_s+config.closure_duration_s)}${config.intervention.startsWith('B')?'':' (inactive: no closure selected)'}. Same requests, random speed factors, station capacities and operating policy in both arms. Cargo target remains 8.`;
    const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});$('configDownload').href=URL.createObjectURL(blob);$('configDownload').hidden=false;renderAnalysis();render();
  }catch(e){if(token===jobToken){$('notice').textContent='Scenario not calculated: '+e.message;$('notice').className='warning';}}
  finally{if(token===jobToken)$('runScenario').disabled=false;}
}
$('savedMode').onclick=()=>setMode('saved');$('newMode').onclick=()=>setMode('new');$('runScenario').onclick=runScenario;
$('enabled').onchange=()=>labMode==='new'?invalidate():load();
for(const id of ['closureStart','closureDuration','initialBikes'])$(id).oninput=()=>{if(labMode==='new')invalidate();};
$('journeyFilter').add(new Option('Changed route / outcome','affected'));
for(const id of ['city','workload','seed','policy','evidence','comparison','scenario','scale'])$(id).addEventListener('change',()=>load());
$('time').oninput=()=>{playing=false;setTime(+val('time'));};$('play').onclick=()=>{playing=!playing;$('play').innerHTML=`<i data-lucide="${playing?'pause':'play'}"></i>`;window.lucide?.createIcons();};$('restart').onclick=()=>setTime(0);$('fit').onclick=()=>{fit();render();};$('groups').onchange=render;
$('journeyFilter').onchange=()=>{tableTime=-1;render();};$('route').onchange=()=>{route=val('route');tableTime=-1;render();};
$('stations').onclick=e=>{const row=e.target.closest('[data-station]');if(row){station=+row.dataset.station;tableTime=-1;render();}};
$('journeys').onclick=e=>{const row=e.target.closest('[data-route]');if(row){route=row.dataset.route;$('route').value=route;tableTime=-1;render();}};
$('tradeoffs').onclick=e=>{const row=e.target.closest('[data-scenario]');if(row){$('scenario').value=row.dataset.scenario;load();}};
$('ranking').onclick=e=>{const y=e.clientY-e.target.getBoundingClientRect().top,hit=rankHits.find(h=>y>=h.top&&y<=h.bottom);if(hit){$('scenario').value=hit.id;load();}};
for(const id of ['leftMap','rightMap']){const c=$(id);let drag=null;c.onwheel=e=>{e.preventDefault();zoom=Math.max(.5,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));backgrounds.clear();render();};c.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,px:pan[0],py:pan[1]};c.setPointerCapture(e.pointerId);};c.onpointermove=e=>{if(drag){pan=[drag.px+e.clientX-drag.x,drag.py+e.clientY-drag.y];backgrounds.clear();render();}};c.onpointerup=e=>{if(drag&&Math.hypot(e.clientX-drag.x,e.clientY-drag.y)<5){const rect=c.getBoundingClientRect(),p=projector(rect.width,rect.height);let best=-1,dist=20;nets[0].stations.forEach((s,i)=>{const at=p(s.xy),d=Math.hypot(at[0]-e.clientX+rect.left,at[1]-e.clientY+rect.top);if(d<dist){dist=d;best=i;}});if(best>=0){station=best;tableTime=-1;render();}}drag=null;};}
window.addEventListener('resize',()=>{backgrounds.clear();if(ready){render();renderAnalysis();}});
function tick(now){if(playing&&ready){setTime(t+Math.min((now-last)/1000,.1)*+val('speed'));if(t>=pair[0].tape.horizon)playing=false;}last=now;requestAnimationFrame(tick);}requestAnimationFrame(tick);
window.demo={async configure(settings){for(const [k,v]of Object.entries(settings))if($(k))$(k).value=v;await load();},setTime,setMode,runScenario,getState:()=>({ready,t,labMode,customRun,affected:[...affected],settings:Object.fromEntries(['city','workload','seed','policy','evidence','comparison','scenario','scale'].map(k=>[k,val(k)])),results:pair.map(r=>r.result),rendered:window.lastRendered}), getDomain:()=>({pair,nets,atTime,position,requestState}),selectRoute:id=>{route=String(id);$('route').value=route;tableTime=-1;render();},selectStation:i=>{station=i;tableTime=-1;render();}};
(async()=>{try{[report,catalog]=await Promise.all([json('/app/data/report.json'),json('/app/data/catalog.json')]);$('coverage').textContent='992 retained episodes verified · 744 new investments';$('coverageDetail').textContent=`Full replay: 992 retained episodes / 445,824 request outcomes; every non-timing summary field and ${report.coverage.retained_events_verified_exactly.toLocaleString()} events reconciled. Corrected view: 248 recomputed open baselines plus 744 unchanged closure episodes. Investments: 744 new episodes, all four scales and all 31 saved input tapes. No smoke-run substitution. Snapshots every 60 seconds; live inventories use event timestamps.`;$('caveats').innerHTML=report.caveats.map(s=>`<li>${escape(s)}</li>`).join('');refreshOptions();t=4500;await load();window.lucide?.createIcons();}catch(e){$('notice').textContent=e.message;console.error(e);}})();
