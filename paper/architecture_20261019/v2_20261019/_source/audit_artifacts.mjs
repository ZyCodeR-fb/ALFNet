import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const repo=path.resolve(root,'../../..');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const manifest=JSON.parse(fs.readFileSync(path.join(repo,'paper/pems08_wd_test_20261012/selected_configuration_manifest.json'),'utf8'));
const configs=Object.entries(manifest.entries).map(([dataset,e])=>{
 const raw=fs.readFileSync(path.join(repo,e.config),'utf8');
 const actual=sha(raw.replaceAll('\r\n','\n'));
 const experts=[...raw.matchAll(/config_path:\s*(\S+)/g)].map(m=>m[1]);
 return {dataset,path:e.config,sha256_lf:actual,manifest_match:actual===e.config_sha256,experts:experts.map(p=>({path:p,sha256_lf:sha(fs.readFileSync(path.join(repo,p),'utf8').replaceAll('\r\n','\n'))})),interpolation:e.interpolation,decoder_scale:e.decoder_scale,center_alpha:e.center_alpha,capacity:e.hyperparameters.capacity};
});
const index=JSON.parse(fs.readFileSync(path.join(root,'qa/module_index.json'),'utf8'));
const files=['alfnet_architecture.png','module_library.svg',...index.map(m=>m.path),'_source/alfnet_architecture.svg'];
const inventory=files.map(p=>{const b=fs.readFileSync(path.join(root,p));return {path:p,bytes:b.length,sha256:sha(b)};});
const svgs=files.filter(p=>p.endsWith('.svg')).map(p=>{
 const s=fs.readFileSync(path.join(root,p),'utf8');
 return {path:p,text:(s.match(/<text\b/g)||[]).length,raster:(s.match(/<image\b/g)||[]).length,foreignObject:s.includes('<foreignObject'),externalReference:/\b(?:href|src)\s*=\s*["'](?:https?:|file:|data:)/.test(s)};
});
const png=fs.readFileSync(path.join(root,'alfnet_architecture.png'));
const dimensions={width:png.readUInt32BE(16),height:png.readUInt32BE(20)};
const main=JSON.parse(fs.readFileSync(path.join(root,'alfnet_architecture_audit.json'),'utf8'));
const mods=JSON.parse(fs.readFileSync(path.join(root,'qa/module_audits.json'),'utf8'));
const noGeometryErrors=a=>['overflow','overwide','textOverlaps','lineTextIntersections'].every(k=>a[k]?.length===0);
const checks={selectedConfigHashes:configs.every(c=>c.manifest_match),expertConfigCount:configs.reduce((n,c)=>n+c.experts.length,0)===8,pngDimensions:dimensions.width===6000&&dimensions.height===3960,editableVectorOnly:svgs.every(s=>s.text>0&&s.raster===0&&!s.foreignObject&&!s.externalReference),mainGeometry:noGeometryErrors(main),moduleGeometry:mods.length===19&&mods.every(noGeometryErrors),minimumTypeface:main.minFontSize>=28};
const report={scope:'Static source/configuration and generated-figure checks only; no model execution or dataset/checkpoint access.',dimensions,minimumFontPtAt180mm:main.minFontSize/2000*180/25.4*72,checks,configs,inventory,svgs};
fs.writeFileSync(path.join(root,'qa/final_artifact_checks.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify({checks,dimensions,modules:mods.length,minFontPtAt180mm:report.minimumFontPtAt180mm}));
if(Object.values(checks).some(x=>!x))process.exitCode=1;
