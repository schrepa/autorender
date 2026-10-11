import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { sceneUrl, openPaused, routedScene } from "./harness.mjs";

export const fixtureCamera={position:[128,140,300],target:[128,0,128],fov_degrees:55,near:0.1,far:1000};
export function heightBytes(ridge=false){
  const out=Buffer.alloc(257*257*2);
  for(let z=0;z<257;z++)for(let x=0;x<257;x++)out.writeUInt16LE(x===0||z===0||x===256||z===256?65535:ridge&&x===128?21845:0,2*(z*257+x));
  return out;
}
export function document({ridge=false,budget=0.25,origin=[0,0,0],hole=false,sea=null,copies=1,hydrology=false}={}){
  const terrain={origin,tiles:[1,1],spacing:1,layers:[{heightmap:{uri:"basin-fog-height.r16",width:257,height:257,min:[0,0],size:[256,256],low:0,high:ridge?6:2,blend:"replace"}}],
    surface:{materials:[{id:"ground",color:[0.4,0.4,0.4]}],biomes:[{id:"ground",layers:[{material:"ground"}]}]}};
  if(hole)terrain.layers.push({stamp:{center:[64.5,64.5],radius:0.5,op:"hole"}});
  if(hydrology)terrain.layers.push({hydrology:{level:0,channel_area:1e10,depth:0,max_depth:0}});
  return {version:9,id:"basin-fog-fixture",revision:1,camera:fixtureCamera,materials:[],meshes:[],
    environment:{basin_fog:{enabled:true,budget_m:budget,extinction_per_m:0.025,top_fade_m:3,level:0},
      terrain:{samples_per_frame:50000000,loading_work_ms:50,surroundings:false},water:{enabled:hydrology},ocean:{enabled:sea!==null,level:sea??0},
      ground_cover:{enabled:false},wind:{velocity:[0,0],gusts:0},clouds:{enabled:false},shadows:{enabled:false},ambient_occlusion:{intensity:0},bloom:{intensity:0},reflections:{intensity:0},sun_shafts:{intensity:0},contact_shadows:{enabled:false},temporal_antialiasing:{enabled:false}},
    entities:Array.from({length:copies},(_,i)=>({id:i?`ground${i}`:"ground",terrain:structuredClone(terrain)}))};
}
export async function session(page,command,options={}){
  let ridge=options.ridge??false;
  await page.route("**/basin-fog-height.r16*",route=>route.fulfill({body:heightBytes(ridge),contentType:"application/octet-stream"}));
  const scene=routedScene(page,document(options));await scene.route();await openPaused(page,sceneUrl(),{timeout:120000});
  return {async reload(next){ridge=next.ridge??false;await scene.reload(command,document(next));},scene};
}
export async function inspect(command){return (await command("inspect",{component:"basin_fog"})).result.value;}
export async function settle(command,predicate=v=>v.terrains.length>0&&v.terrains.every(t=>t.state==="ready")){
  let state,rendering;
  for(let frame=0;frame<1200;frame++){
    rendering=(await command("render")).result.rendering;state=await inspect(command);
    if(predicate(state)&&!rendering.loading&&!rendering.pipelines_compiling){return {state,rendering};}
  }
  assert.fail(`fog did not settle: ${JSON.stringify(state)}`);
}
export function near(value,expected,tolerance=0.05){assert.ok(Number.isFinite(value)&&Math.abs(value-expected)<=tolerance,`${value} differs from ${expected}`);}
export async function capture(page,command,output,name){
  await command("render");await command("capture");return (await page.locator("canvas").screenshot({path:`${output}/${name}.png`})).toString("base64");
}
export async function record(output,name,value){await writeFile(`${output}/${name}.json`,JSON.stringify(value,null,2));}
