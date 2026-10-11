/** Authored prism optics and the fixed Highlands demonstration.
 * Every image error and monotonicity result is descriptive, without a pass threshold. */
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { artifacts,commander,launch,sceneUrl,openPaused,routedScene,devUrl,differentPixels } from "./harness.mjs";
import { document,heightBytes,inspect,settle,capture,record,near } from "./basin-fog-fixture.mjs";
const output=await artifacts("basin-fog-render-smoke");
const {browser,page,errors}=await launch({width:1400,height:900,timeout:120000});
const command=commander(page,{deadline:120000});const results=[];
const H=[0.08,0.18,0.32],L=[0.6,0.12,0.03];
const camera={position:[128,1,200],target:[128,1,64],fov_degrees:35,near:0.1,far:1000};

// Independent binary64 reference: an authored rectangular medium, geometric
// ray/slab clipping, and the antiderivative of its vertical ramp. No production
// maps, pool IDs, tree nodes or shader text are read by this model.
async function reference(name,png,{sigma,fade,copies=1,eye=camera.position,target=camera.target,fov=35}){
  const answer=await page.evaluate(async({png,H,L,sigma,fade,copies,eye,target,fov})=>{
    const image=new Image();image.src=`data:image/png;base64,${png}`;await image.decode();
    const canvas=document.createElement("canvas");canvas.width=image.width;canvas.height=image.height;
    const ctx=canvas.getContext("2d");ctx.drawImage(image,0,0);const actual=ctx.getImageData(0,0,canvas.width,canvas.height);
    const expected=ctx.createImageData(canvas.width,canvas.height),error=ctx.createImageData(canvas.width,canvas.height);
    const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
    const unit=a=>{const n=Math.hypot(...a);return a.map(v=>v/n);},forward=unit(target.map((v,i)=>v-eye[i])),right=unit(cross(forward,[0,1,0])),up=cross(right,forward);
    const interval=(d,lo,hi,end=Infinity)=>{let a=0,b=end;for(let i=0;i<3;i++){if(d[i]===0){if(eye[i]<lo[i]||eye[i]>hi[i])return null;}else{const x=(lo[i]-eye[i])/d[i],y=(hi[i]-eye[i])/d[i];a=Math.max(a,Math.min(x,y));b=Math.min(b,Math.max(x,y));}}return b>a?[a,b]:null;};
    // A primitive whose derivative is clamp((2-y)/fade,0,1).
    const primitive=y=>{if(fade===0)return y;const start=2-fade;if(y<=start)return y;const t=Math.min(y,2)-start;return start+t-t*t/(2*fade);};
    const grade=v=>{const curve=x=>Math.max(0,Math.min(1,x*(2.51*x+0.03)/(x*(2.43*x+0.59)+0.14)));const peak=Math.max(...v),keep=curve(peak)/Math.max(peak,1e-6);return v.map(x=>{const linear=0.5*(curve(x)+x*keep);return Math.round(255*(linear<=0.0031308?12.92*linear:1.055*linear**(1/2.4)-0.055));});};
    const histogram=Array(256).fill(0),rays=[];let count=0,totalError=0,changed=0,maxError=0;
    const tangent=Math.tan(fov*Math.PI/360),aspect=canvas.width/canvas.height;
    for(let y=0;y<canvas.height;y++)for(let x=0;x<canvas.width;x++){
      const u=(2*(x+0.5)/canvas.width-1)*tangent*aspect,v=(1-2*(y+0.5)/canvas.height)*tangent;
      const ray=unit(forward.map((w,i)=>w+u*right[i]+v*up[i]));
      const wall=interval(ray,[64,0,63.5],[192,20,64.5]);let distance=wall?.[0]??Infinity,light=wall?L:H,valid=true;
      if(ray[1]<0){const ground=-eye[1]/ray[1],p=eye.map((w,i)=>w+ray[i]*ground);if(ground<distance){distance=ground;light=[0,0,0];valid=p[0]>=1&&p[0]<=255&&p[2]>=1&&p[2]<=255;}}
      // Sky rays exiting through the high border are outside this analytic
      // interior/box reference; their rendered captures remain preserved.
      if(!wall&&light===H)valid=false;
      const medium=interval(ray,[1,0,1],[255,2,255],distance);let optical=0;
      if(medium){const [a,b]=medium,ya=eye[1]+ray[1]*a,yb=eye[1]+ray[1]*b;const integral=fade===0?b-a:Math.abs(ray[1])<1e-12?(b-a)*Math.max(0,Math.min(1,(2-ya)/fade)):(primitive(yb)-primitive(ya))/ray[1];optical=Math.min(80,copies*sigma*integral);}
      const passing=Math.exp(-optical),rgb=grade(light.map((w,i)=>passing*w+(1-passing)*H[i])),at=4*(y*canvas.width+x);
      if(valid){let largest=0;for(let c=0;c<3;c++){expected.data[at+c]=rgb[c];const e=Math.abs(rgb[c]-actual.data[at+c]);error.data[at+c]=e;largest=Math.max(largest,e);totalError+=e;}expected.data[at+3]=error.data[at+3]=255;histogram[largest]++;count++;maxError=Math.max(maxError,largest);if(largest)changed++;}
      if(x===Math.floor(canvas.width/2)&&[Math.floor(canvas.height*.4),Math.floor(canvas.height*.5),Math.floor(canvas.height*.6)].includes(y))rays.push({x,y,ray,distance,optical,passing,expected:rgb,actual:Array.from(actual.data.slice(at,at+3)),valid});
    }
    ctx.putImageData(expected,0,0);const reference=canvas.toDataURL();ctx.putImageData(error,0,0);const difference=canvas.toDataURL();
    return {reference,difference,count,changed,maxError,meanAbsoluteError:totalError/(3*count),histogram,rays};
  },{png,H,L,sigma,fade,copies,eye,target,fov});
  for(const field of ["reference","difference"]){await writeFile(`${output}/${name}-${field}.png`,Buffer.from(answer[field].split(",")[1],"base64"));delete answer[field];}
  results.push({case:name,opticalReference:answer,parameters:{sigma,fade,copies,eye,target,fov}});
}

try{
  const fixture=document({budget:2});fixture.camera=camera;fixture.entities[0].terrain.layers[0].heightmap.high=20;
  fixture.entities[0].terrain.surface.materials[0].color=[0,0,0];
  fixture.materials=[{id:"lamp",color:[0,0,0,1],emissive:L}];fixture.meshes=[{id:"box",source:{kind:"cube"}}];
  fixture.entities.push({id:"wall",mesh:"box",material:"lamp",transform:{translation:[128,10,64],scale:[128,20,1]}});
  fixture.environment.atmosphere={preset:"day",sun_color:[0,0,0],zenith:H,horizon:H,ground:H,ambient:0,rim:0,specular:0,emission:1,exposure:1,saturation:1,contrast:1,vibrance:0,shadow_tint:[1,1,1],highlight_tint:[1,1,1],fog_density:0,aerial_density:0};
  await page.route("**/basin-fog-height.r16*",route=>route.fulfill({body:heightBytes(),contentType:"application/octet-stream"}));
  const scene=routedScene(page,fixture);await scene.route();await openPaused(page,sceneUrl(),{timeout:120000});
  let {state}=await settle(command);near(state.terrains[0].lid_range[0],2,0.0001);
  const captures={};
  for(const [name,sigma,fade]of[["off-a",0,3],["off-b",0,3],["ramp",0.025,3],["denser",0.05,3],["hard-top",0.025,0]]){
    await command("environment",{basin_fog:{extinction_per_m:sigma,top_fade_m:fade}});await settle(command);
    captures[name]=await capture(page,command,output,name);await reference(name,captures[name],{sigma,fade});
  }
  results.push({case:"descriptive capture changes",offRepeat:await differentPixels(page,[captures["off-a"],captures["off-b"]]),on:await differentPixels(page,[captures["off-a"],captures.ramp]),density:await differentPixels(page,[captures.ramp,captures.denser])});
  for(const [name,position,target]of[["above-lid",[128,4,200],[128,4,64]],["inside",[128,1,170],[128,1,64]],["below-floor",[128,-1,200],[128,-1,64]],["vertical",[128,3,120],[128,0,120]]]){
    await command("camera",{...camera,position,target});await settle(command);await capture(page,command,output,name);results.push({case:name,position,target});
  }
  const overlap=structuredClone(fixture);overlap.entities.push({...structuredClone(overlap.entities[0]),id:"ground1"});
  await scene.reload(command,overlap);await command("environment",{basin_fog:{top_fade_m:3}});({state}=await settle(command));assert.equal(state.terrains.length,2);
  const doubled=await capture(page,command,output,"overlap");await reference("overlap",doubled,{sigma:0.025,fade:3,copies:2});
  // Distinct integration paths retain raw matched captures. These deliberately
  // have no appearance assertions; the optical reference above covers opaque rays.
  const paths=structuredClone(fixture);paths.materials.push({id:"glass",color:[0.8,0.9,1,0.5],alpha_mode:"blend",transmission:1,thickness:1,ior:1.5});
  paths.entities.push({id:"glass",mesh:"box",material:"glass",transform:{translation:[128,2,120],scale:[30,4,1]}});
  paths.environment.refraction={enabled:true};await scene.reload(command,paths);await settle(command);
  for(const sigma of [0,0.025]){await command("environment",{basin_fog:{extinction_per_m:sigma}});await settle(command);await capture(page,command,output,`glass-${sigma}`);}
  for(const view of [false,true]){
    await command("environment",{clouds:{enabled:true,coverage:0.5,base:80,thickness:100,in_view:view}});await settle(command);await capture(page,command,output,view?"view-clouds":"dome-clouds");
  }
  // The unchanged Highlands overview, at the catalog viewport/seed/settings.
  await openPaused(page,devUrl("stress.html",{preset:"highlands",view:"overview",look:"scene",effects:"all",visibility:"off",lod:"on",impostors:"on","basin-fog":"on"}),{timeout:180000});
  ({state}=await settle(command));const demo=(await command("inspect",{component:"BasinDemo"})).result.value;
  assert.equal(demo.ready,true);assert.equal(demo.final_edits-demo.original_edits,4);assert.equal(state.dropped_terrains,0);
  assert.ok(state.terrains.some(t=>t.wet_cells>0));const seen=(await command("inspect",{component:"Camera"})).result.value;
  results.push({case:"Highlands authored content",demo,camera:seen,state});
  await capture(page,command,output,"highlands-on");await command("environment",{basin_fog:{extinction_per_m:0}});await settle(command);
  const zero=await capture(page,command,output,"highlands-same-content-zero");await command("environment",{basin_fog:{extinction_per_m:0.025}});await settle(command);
  const fog=await capture(page,command,output,"highlands-same-content-fog");results.push({case:"Highlands same content",changed:await differentPixels(page,[zero,fog]),state:await inspect(command)});
  for(const t of state.terrains.filter(t=>t.largest)){
    const p=t.largest.deepest,position=[p[0]+t.origin[0],p[1]+t.origin[1]+t.largest.depth_m*.5,p[2]+t.origin[2]];
    await command("camera",{position,target:[position[0]+20,position[1],position[2]],fov_degrees:55,near:0.1,far:20000});await settle(command);await capture(page,command,output,"highlands-inside-largest");
    results.push({case:"Highlands inside pool",position,pool:t.largest});break;
  }
  await command("camera",{position:demo.eye,target:demo.target,fov_degrees:55,near:0.5,far:20000});await command("resume");await page.waitForTimeout(1500);
  await page.locator("canvas").screenshot({path:`${output}/highlands-ordinary-a.png`});await page.waitForTimeout(417);await page.locator("canvas").screenshot({path:`${output}/highlands-ordinary-b.png`});await command("pause");
  await openPaused(page,devUrl("stress.html",{preset:"highlands",view:"overview",look:"scene",effects:"all",visibility:"off",lod:"on",impostors:"on","basin-fog":"off"}),{timeout:180000});
  let rendering;for(let i=0;i<1200;i++){rendering=(await command("render")).result.rendering;if(!rendering.loading&&!rendering.pipelines_compiling)break;}
  assert.equal(rendering.basin_fog.gpu_bytes,0);await capture(page,command,output,"highlands-original-off");
  assert.equal(errors.length,0,errors.join("\n"));await record(output,"results",results);console.log(JSON.stringify({ok:true,results:results.length,output}));
}catch(error){await record(output,"failure",{error:String(error),errors,results});console.error(error);process.exitCode=3;}
finally{await browser.close();}
