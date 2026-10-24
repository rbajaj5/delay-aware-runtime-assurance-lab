const {chromium}=require('C:/Users/anaxe/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),{spawn}=require('child_process'),{once}=require('events'),assert=require('assert');
(async()=>{
 const plan=JSON.parse(fs.readFileSync('output/movie/walkthrough.json','utf8')),caps=JSON.parse(fs.readFileSync('output/movie/captions.json','utf8'));const preview=process.argv.includes('--preview');
 const browser=await chromium.launch({headless:true,channel:'msedge'}),page=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('http://127.0.0.1:8766/');await page.waitForFunction(()=>window.demo?.getState().ready);await page.addStyleTag({url:'http://127.0.0.1:8766/app/demo.css'});await page.evaluate(()=>document.body.classList.add('movie-mode'));await page.evaluate(()=>document.fonts.ready);
 let encoder,log,frameLog;
 if(!preview){log=fs.openSync('output/movie/dashboard_encoder.log','w');encoder=spawn('ffmpeg',['-y','-v','warning','-f','image2pipe','-vcodec','mjpeg','-framerate','12','-i','pipe:0','-vf','fps=24','-an','-c:v','h264_nvenc','-preset','p5','-cq','20','-pix_fmt','yuv420p','-movflags','+faststart','output/movie/dashboard_silent.mp4'],{stdio:['pipe','ignore',log]});frameLog=fs.createWriteStream('output/movie/dashboard_frame_index.jsonl');}
 const actual=[];let count=0;const start=Date.now();
 for(let si=0;si<plan.scenes.length;si++){
  const scene=plan.scenes[si];
  await page.evaluate(async s=>{await window.demo.setMode(s.lab_mode||'saved');await window.demo.configure(s.settings);document.getElementById('enabled').checked=true;if(s.lab_mode==='new'){document.getElementById('closureStart').value=s.closure_start_min;document.getElementById('closureDuration').value=s.closure_duration_min;document.getElementById('initialBikes').value=s.initial_bikes;await window.demo.runScenario();}if(s.route!==undefined)window.demo.selectRoute(s.route);if(s.station!==undefined)window.demo.selectStation(s.station);},scene);
  const state=await page.evaluate(()=>window.demo.getState());assert(state.ready);actual.push({scene:si,config:scene,episodes:state.rendered.episodes,results:state.results,custom:state.customRun?.config||null});
  const times=preview?[(scene.start+scene.end)/2]:Array.from({length:(scene.end-scene.start)*12},(_,i)=>scene.start+i/12);
  for(const movieTime of times){
   const frac=(movieTime-scene.start)/(scene.end-scene.start),simTime=scene.time_start+(scene.time_end-scene.time_start)*frac,caption=caps.find(c=>movieTime>=c.start_s&&movieTime<c.end_s)?.text||'';
   const rendered=await page.evaluate(({t,caption})=>{window.demo.setTime(t);document.getElementById('movieCaption').textContent=caption;return window.demo.getState().rendered;},{t:simTime,caption});
   const jpeg=await page.screenshot({type:'jpeg',quality:88});
   if(preview)fs.writeFileSync(`output/movie/dashboard_preview_${si}.jpg`,jpeg);
   else {if(!encoder.stdin.write(jpeg))await once(encoder.stdin,'drain');frameLog.write(JSON.stringify({capture:count,movie_time_s:movieTime,simulation_time_s:simTime,scene:si,episodes:rendered.episodes,states:rendered.states.map(s=>({completed:s.completed,onboard:s.onboard,fleet:s.fleet}))})+'\n');if(count%180===0){console.log(`Dashboard capture ${count}/2160; elapsed ${Math.round((Date.now()-start)/1000)}s`);fs.writeFileSync(`output/movie/dashboard_sample_${String(count).padStart(4,'0')}.jpg`,jpeg);}count++;}
  }
 }
 if(!preview){encoder.stdin.end();const [code]=await once(encoder,'exit');assert.equal(code,0);frameLog.end();fs.closeSync(log);assert.equal(count,2160);}
 assert.deepEqual(errors,[]);fs.writeFileSync('output/movie/walkthrough_resolved.json',JSON.stringify({scenes:actual,capture_frames:count,errors,preview},null,2));await browser.close();console.log(preview?'Dashboard previews ready':'Dashboard recording complete');
})().catch(e=>{console.error(e);process.exit(1);});
