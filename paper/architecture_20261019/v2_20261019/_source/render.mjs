import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {spawn} from 'node:child_process';
import {fileURLToPath,pathToFileURL} from 'node:url';

const ROOT=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const input=path.resolve(process.argv[2]||path.join(ROOT,'_source/alfnet_architecture.svg'));
const output=path.resolve(process.argv[3]||path.join(ROOT,'alfnet_architecture.png'));
const scale=Number(process.argv[4]||1);
const profile=fs.mkdtempSync(path.join(os.tmpdir(),'alfnet-figure-'));
const browser=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',[
 '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
 '--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'
],{windowsHide:true,stdio:['ignore','ignore','pipe']});
let browserError='';browser.stderr.on('data',d=>{browserError+=d.toString();});
let ws;
try{
 let port;
 for(let i=0;i<100;i++){try{port=Number(fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0]);break;}catch{await new Promise(r=>setTimeout(r,100));}}
 if(!port)throw new Error('Chrome DevTools endpoint unavailable: '+browserError.slice(-2000));
 const tabs=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
 ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
 let id=0;const pending=new Map();
 ws.onmessage=event=>{const m=JSON.parse(event.data);if(m.id&&pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}};
 function call(method,params={}){return new Promise((resolve,reject)=>{const i=++id;pending.set(i,{resolve,reject});ws.send(JSON.stringify({id:i,method,params}));});}
 await call('Page.enable');
 await call('Page.navigate',{url:pathToFileURL(input).href});
 let loaded=false;
 for(let i=0;i<60;i++){const a=await call('Runtime.evaluate',{expression:'document.documentElement.tagName',returnByValue:true});if(a.result.value==='svg'){loaded=true;break;}await new Promise(r=>setTimeout(r,100));}
 if(!loaded)throw new Error('SVG document did not load');
 const dims=await call('Runtime.evaluate',{expression:'({w:document.documentElement.viewBox.baseVal.width,h:document.documentElement.viewBox.baseVal.height})',returnByValue:true});
 await call('Emulation.setDeviceMetricsOverride',{width:dims.result.value.w,height:dims.result.value.h,deviceScaleFactor:scale,mobile:false});
 await call('Runtime.evaluate',{expression:'document.fonts.ready',awaitPromise:true});
 const auditExpression=`(()=>{
 const svg=document.documentElement,v=svg.viewBox.baseVal;
 const texts=[...document.querySelectorAll('text')].map(e=>{const b=e.getBoundingClientRect();return {text:e.textContent,x:b.x,y:b.y,w:b.width,h:b.height,max:+e.getAttribute('data-max-width'),fontSize:parseFloat(getComputedStyle(e).fontSize)};});
 const segments=[...document.querySelectorAll('polyline')].flatMap(e=>{const m=e.getScreenCTM();const p=[...e.points].map(p=>new DOMPoint(p.x,p.y).matrixTransform(m));return p.slice(1).map((b,i)=>({a:p[i],b}));});
 const hits=(s,t)=>{let lo=0,hi=1;for(const [a,d,min,max] of [[s.a.x,s.b.x-s.a.x,t.x+4,t.x+t.w-4],[s.a.y,s.b.y-s.a.y,t.y+4,t.y+t.h-4]]){if(Math.abs(d)<1e-7){if(a<min||a>max)return false;}else{const q=(min-a)/d,r=(max-a)/d;lo=Math.max(lo,Math.min(q,r));hi=Math.min(hi,Math.max(q,r));if(lo>hi)return false;}}return lo<=hi;};
 return {width:v.width,height:v.height,textCount:texts.length,minFontSize:Math.min(...texts.map(t=>t.fontSize)),rasterCount:document.querySelectorAll('image').length,
 overflow:texts.filter(t=>t.x<0||t.y<0||t.x+t.w>v.width+1||t.y+t.h>v.height+1),overwide:texts.filter(t=>t.max&&t.w>t.max),
 textOverlaps:texts.flatMap((a,i)=>texts.slice(i+1).filter(b=>Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x)>2&&Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y)>2).map(b=>[a.text,b.text])),
 lineTextIntersections:texts.flatMap(t=>segments.filter(s=>hits(s,t)).map(s=>({text:t.text,line:[[s.a.x,s.a.y],[s.b.x,s.b.y]]})))
 };})()`;
 const audit=await call('Runtime.evaluate',{expression:auditExpression,returnByValue:true});
 const result=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
 fs.writeFileSync(output,Buffer.from(result.data,'base64'));
 const auditPath=output.replace(/\.png$/,'_audit.json');
 fs.writeFileSync(auditPath,JSON.stringify(audit.result.value,null,2));
 console.log(JSON.stringify({output,dimensions:[dims.result.value.w*scale,dims.result.value.h*scale],audit:audit.result.value}));
 if(process.argv.includes('--audit-modules')){
  const index=JSON.parse(fs.readFileSync(path.join(ROOT,'qa/module_index.json'),'utf8'));
  const reports=[];
  for(const m of index){
   const target=pathToFileURL(path.join(ROOT,m.path)).href;
   await call('Page.navigate',{url:target});
   let ready=false;
   for(let i=0;i<60;i++){
    const check=await call('Runtime.evaluate',{expression:`document.location.href===${JSON.stringify(target)}&&document.readyState==='complete'&&document.documentElement.tagName==='svg'`,returnByValue:true});
    if(check.result.value){ready=true;break;}await new Promise(r=>setTimeout(r,50));
   }
   if(!ready)throw new Error('Module did not load: '+m.path);
   await call('Emulation.setDeviceMetricsOverride',{width:m.width,height:m.height,deviceScaleFactor:1,mobile:false});
   await call('Runtime.evaluate',{expression:'document.fonts.ready',awaitPromise:true});
   const a=await call('Runtime.evaluate',{expression:auditExpression,returnByValue:true});
   reports.push({id:m.id,...a.result.value});
  }
  fs.writeFileSync(path.join(ROOT,'qa/module_audits.json'),JSON.stringify(reports,null,2));
  console.log(JSON.stringify({modules:reports.length,failures:reports.filter(a=>a.overflow.length||a.overwide.length||a.textOverlaps.length||a.lineTextIntersections.length)}));
 }
 await call('Browser.close').catch(()=>{});
}finally{ws?.close();browser.kill();}
