const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('playwright');

async function main() {
  const folder = path.resolve(__dirname, '../artifacts/20261013/portfolio_rta/tableau');
  const source = JSON.parse(fs.readFileSync(path.join(folder, '../synchronized_frames.json'), 'utf8')).frames;
  const queries = JSON.parse(fs.readFileSync(path.join(folder, 'dashboard_validation.json'), 'utf8'));
  const browser = await chromium.launch({headless:true,channel:'msedge'});
  const page = await browser.newPage({viewport:{width:1440,height:1050},deviceScaleFactor:1});
  const errors = [];
  page.on('pageerror', e=>errors.push(String(e)));
  await page.goto(pathToFileURL(path.join(folder,'dashboard_review.html')).href);
  await page.waitForFunction(()=>window.dashboardState);
  const results = [];
  for (const test of queries.filter_combinations) {
    await page.selectOption('#phase', {label:test.phase});
    await page.selectOption('#controller', {label:test.controller});
    const state = await page.evaluate(()=>window.dashboardState);
    const chosen = source.filter(r=>(test.phase==='All'||r.phase===test.phase)&&(test.controller==='All'||r.controller_decision===test.controller));
    let passed = state.frames === test.frames && state.hidden_frames === test.hidden_frames && state.small_group === test.small_group;
    passed &&= JSON.stringify(state.frameIds)===JSON.stringify(chosen.map(r=>r.frame));
    passed &&= state.mean_leading_share===null ? test.mean_leading_share===null : Math.abs(state.mean_leading_share-test.mean_leading_share)<1e-10;
    for(const [name, key] of [['proposed','weights_proposed'],['applied','weights_applied']]){
      for(let j=0;j<6;j++){
        const expected=chosen.length?chosen.reduce((s,r)=>s+r[key][j],0)/chosen.length:null;
        passed &&= expected===null?state[name][j]===null:Math.abs(state[name][j]-expected)<1e-10;
      }
    }
    results.push({...test, passed});
  }
  await page.selectOption('#phase','All');
  await page.selectOption('#controller','All');
  await page.screenshot({path:path.join(folder,'dashboard_desktop.png'),fullPage:true});
  await page.selectOption('#phase','Geometry under stress');
  await page.selectOption('#controller','RTA HOLD');
  await page.screenshot({path:path.join(folder,'dashboard_hold.png'),fullPage:true});
  await page.selectOption('#controller','ACCEPT');
  const smallWarning=await page.locator('#warning').innerText();
  await page.screenshot({path:path.join(folder,'dashboard_small_group.png'),fullPage:true});
  await page.selectOption('#phase','Stable five-out');
  await page.selectOption('#controller','RTA HOLD');
  const emptyWarning=await page.locator('#warning').innerText();
  await page.screenshot({path:path.join(folder,'dashboard_empty.png'),fullPage:true});
  await page.selectOption('#phase','All');
  await page.selectOption('#controller','All');
  await page.locator('#cursor').fill('1800');
  const cursorText=await page.locator('#readout').innerText();
  const desktopOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(folder,'dashboard_mobile.png'),fullPage:true});
  const mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  const result={tool:'Playwright Chromium; local HTML review only',native_tableau_ui_tested:false,
    filter_tests:results,console_errors:errors,small_group_warning:smallWarning,empty_warning:emptyWarning,cursor_readout:cursorText,
    desktop_horizontal_overflow:desktopOverflow,mobile_horizontal_overflow:mobileOverflow,
    all_passed:results.every(t=>t.passed)&&errors.length===0&&!desktopOverflow&&!mobileOverflow&&smallWarning.includes('14')&&emptyWarning.includes('No observations')&&cursorText.includes('75.00')};
  fs.writeFileSync(path.join(folder,'preview_validation.json'),JSON.stringify(result,null,2));
  await browser.close();
  console.log(JSON.stringify(result,null,2));
  if(!result.all_passed)process.exitCode=1;
}
main().catch(e=>{console.error(e);process.exitCode=1;});
