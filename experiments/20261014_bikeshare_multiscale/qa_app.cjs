const {chromium}=require('C:/Users/anaxe/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');
const path=require('path');
const assert=require('assert');
(async()=>{
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 const page=await browser.newPage({viewport:{width:1600,height:1050},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('http://127.0.0.1:8766/');await page.waitForFunction(()=>window.demo?.getState().ready,{timeout:120000});
 const checks=[];
 async function verify(name){const state=await page.evaluate(()=>{const d=window.demo.getDomain();const result=[];for(let i=0;i<2;i++){const r=d.pair[i],s=d.atTime(r,d.nets[i],r.tape.horizon);result.push({completed:s.completed,expected:r.result.completed,failed:s.failed,expectedFailed:r.result.failed_pickups,fleet:s.fleet,expectedFleet:r.result.fleet});}return {state:window.demo.getState(),result};});for(const r of state.result){assert.equal(r.completed,r.expected);assert.equal(r.failed,r.expectedFailed);assert.equal(r.fleet,r.expectedFleet);}checks.push({name,settings:state.state.settings,results:state.result});}
 await verify('Initial investment');
 const combinations=[
  {city:'washington',workload:'controlled',policy:'cargo_crew',comparison:'scale',scenario:'I1',scale:'1400',seed:'2026101405'},
  {city:'washington',workload:'observed_week',policy:'no_redistribution',comparison:'investment',scenario:'I2',scale:'600',seed:'2026101403'},
  {city:'kaliningrad',workload:'controlled',policy:'cargo_crew',comparison:'investment',scenario:'I3',scale:'200',seed:'2026101411'},
  {city:'kaliningrad',comparison:'recovery',scenario:'B03',scale:'0',seed:'2026101402'},
  {city:'washington',workload:'controlled',evidence:'retained',comparison:'recovery',scenario:'B04',scale:'600',seed:'2026101400'},
  {city:'washington',evidence:'corrected',comparison:'investment',scenario:'I1',scale:'200'}];
 for(const settings of combinations){await page.evaluate(async s=>{await window.demo.configure(s);window.demo.setTime(9500);},settings);await verify(JSON.stringify(settings));}
 // Exercise every option of each selector, using real change events.
 for(const id of ['city','workload','policy','evidence','comparison','scale','seed','scenario']){
   const opts=await page.locator('#'+id+' option:not([disabled])').evaluateAll(a=>a.map(x=>x.value));
   if(await page.locator('#'+id).isDisabled())continue;
   for(const value of opts){await page.selectOption('#'+id,value);await page.waitForFunction(()=>window.demo.getState().ready);await verify(id+'='+value);}
 }
 await page.evaluate(async()=>{await window.demo.configure({city:'washington',workload:'controlled',policy:'no_redistribution',evidence:'corrected',comparison:'scale',scenario:'I1',scale:'1400',seed:'2026101400'});window.demo.setTime(6500);});
 await page.locator('#groups').check();await page.locator('[data-station="3"]').click();
 await page.selectOption('#route','20');await page.selectOption('#journeyFilter','active');
 await page.locator('#leftMap').hover();await page.mouse.wheel(0,-250);await page.locator('#fit').click();
 const before=await page.evaluate(()=>window.demo.getState().t);await page.locator('#play').click();await page.waitForTimeout(500);await page.locator('#play').click();assert((await page.evaluate(()=>window.demo.getState().t))>before);
 await page.evaluate(()=>window.demo.setTime(6500));
 await page.screenshot({path:'output/planning/dashboard_desktop.png',fullPage:true});
 for(const size of [{width:1920,height:1080},{width:390,height:844}]){
  await page.setViewportSize(size);await page.waitForTimeout(200);
  const fit=await page.evaluate(()=>({body:document.body.scrollWidth,viewport:innerWidth,canvases:[...document.querySelectorAll('canvas')].map(c=>{let d=c.getContext('2d').getImageData(0,0,c.width,c.height).data,count=0;for(let i=0;i<d.length;i+=4)if(d[i+3]&&d[i]<240)count++;return {id:c.id,nonblank:count};})}));
  assert(fit.body<=fit.viewport+2,JSON.stringify(fit));assert(fit.canvases.every(c=>c.nonblank>100));checks.push({name:'viewport',size,fit});
  await page.screenshot({path:`output/planning/dashboard_${size.width}.png`,fullPage:true});
 }
 assert.deepEqual(errors,[]);fs.writeFileSync('output/planning/browser_validation.json',JSON.stringify({passed:true,checks,errors},null,2));await browser.close();console.log('Browser QA passed',checks.length,'checks');
})().catch(e=>{console.error(e);process.exit(1);});
