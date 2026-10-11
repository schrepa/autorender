/** Authored prism optics and the fixed Highlands demonstration.
 * Every image error and monotonicity result is descriptive, without a pass threshold. */
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { artifacts,commander,launch,sceneUrl,openPaused,routedScene,devUrl,differentPixels } from "./harness.mjs";
import { document,heightBytes,inspect,settle,capture,record,near } from "./basin-fog-fixture.mjs";
const output=await artifacts("basin-fog-render-smoke");
const {browser,page,errors}=await launch({width:1400,height:900,timeout:120000});
const command=commander(page,{deadline:120000});const results=[];
const H=[0.08,0.18,0.32],L=[0.6,0.12,0.03];
const camera={position:[128,0.125,200],target:[128,0.125,64],fov_degrees:35,near:0.1,far:1000};
const highlandsBytes=await readFile(new URL("../../bench/scenes/highlands.json",import.meta.url));
const highlands=JSON.parse(highlandsBytes);
const highlandsUrl=mode=>devUrl("stress.html",{...highlands.options,look:"scene",effects:"all",visibility:"off",lod:"on",impostors:"on","basin-fog":mode});
const resetCamera=()=>command("camera",{position:camera.position,target:camera.target,vertical_fov_radians:camera.fov_degrees*Math.PI/180,near:camera.near,far:camera.far});

async function highlandsRays(demo,state,seen){
  const terrain=state.terrains.find(t=>String(t.terrain)===String(demo.terrain));
  assert.ok(terrain&&terrain.state==="ready");
  const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0),unit=v=>{const length=Math.hypot(...v);return v.map(x=>x/length);};
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const eye=seen.position,forward=unit(seen.target.map((v,i)=>v-eye[i])),right=unit(cross(forward,[0,1,0])),up=cross(right,forward);
  const tan=Math.tan(seen.vertical_fov_radians/2),aspect=1400/900,answer=[];
  for(let basin=0;basin<demo.centers.length;basin++){
    const center=demo.centers[basin],points=[];
    for(let z=-6;z<=6;z++)for(let x=-6;x<=6;x++)points.push([0,1].map((axis)=>{
      const origin=terrain.origin[axis*2],offset=axis===0?x:z;
      return origin+(Math.floor((center[axis]-origin)/terrain.spacing)+offset+0.5)*terrain.spacing;
    }));
    const samples=(await command("basin_fog_sample",{terrain:String(demo.terrain),points})).result.samples;
    const intersections=[];
    for(let at=0;at<points.length;at++){
      const sample=samples[at];if(!sample?.valid||sample.depth<=0)continue;
      const p=[points[at][0],sample.floor+sample.depth*.5,points[at][1]],delta=p.map((v,i)=>v-eye[i]),forwardDistance=dot(delta,forward);
      if(forwardDistance<=seen.near)continue;
      const u=dot(delta,right)/(forwardDistance*tan*aspect),v=dot(delta,up)/(forwardDistance*tan);
      if(Math.abs(u)>=1||Math.abs(v)>=1)continue;
      const pixel=[Math.floor((u+1)*700),Math.floor((1-v)*450)];
      const pu=((pixel[0]+.5)/700-1)*tan*aspect,pv=(1-(pixel[1]+.5)/450)*tan;
      const ray=unit(forward.map((w,i)=>w+pu*right[i]+pv*up[i]));
      const lo=[points[at][0]-terrain.spacing/2,sample.floor,points[at][1]-terrain.spacing/2],hi=[lo[0]+terrain.spacing,sample.top,lo[2]+terrain.spacing];
      let enter=0,leave=seen.far;
      for(let axis=0;axis<3;axis++){
        if(ray[axis]===0){if(eye[axis]<lo[axis]||eye[axis]>=hi[axis])leave=0;}
        else{const a=(lo[axis]-eye[axis])/ray[axis],b=(hi[axis]-eye[axis])/ray[axis];enter=Math.max(enter,Math.min(a,b));leave=Math.min(leave,Math.max(a,b));}
      }
      if(leave>enter)intersections.push({pixel,ray,interval:[enter,leave],cell:points[at],sample});
    }
    assert.ok(intersections.length>0,`fixed Highlands basin ${basin} has no catalog camera pixel ray crossing a wet cell`);
    // Collision visibility is reported separately: it samples resident terrain
    // and object colliders, not the renderer's exact depth buffer.
    for(const ray of intersections.slice(0,4))ray.collision=(await command("raycast",{origin:eye,direction:ray.ray,max_distance:ray.interval[1]})).result;
    answer.push({basin,geometricIntersections:intersections.length,examples:intersections.slice(0,4)});
  }
  return answer;
}

// Independent binary64 reference: an authored rectangular medium, geometric
// ray/slab clipping, and the antiderivative of its vertical ramp. No production
// maps, pool IDs, tree nodes or shader text are read by this model.
async function reference(name,png,{sigma,fade,copies=1,eye=camera.position,target=camera.target,fov=35}){
  const actualCamera=(await command("inspect",{component:"Camera"})).result.value;
  for(let i=0;i<3;i++){near(actualCamera.position[i],eye[i],0.0001);near(actualCamera.target[i],target[i],0.0001);}
  near(actualCamera.vertical_fov_radians,fov*Math.PI/180,0.00001);
  const answer=await page.evaluate(async({png,H,L,sigma,fade,copies,eye,target,fov})=>{
    const image=new Image();image.src=`data:image/png;base64,${png}`;await image.decode();
    const canvas=document.createElement("canvas");canvas.width=image.width;canvas.height=image.height;
    const ctx=canvas.getContext("2d");ctx.drawImage(image,0,0);const actual=ctx.getImageData(0,0,canvas.width,canvas.height);
    const expected=ctx.createImageData(canvas.width,canvas.height),error=ctx.createImageData(canvas.width,canvas.height);
    const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
    const unit=a=>{const n=Math.hypot(...a);return a.map(v=>v/n);},forward=unit(target.map((v,i)=>v-eye[i])),right=unit(cross(forward,[0,1,0])),up=cross(right,forward);
    const interval=(d,lo,hi,end=Infinity)=>{let a=0,b=end;for(let i=0;i<3;i++){if(d[i]===0){if(eye[i]<lo[i]||eye[i]>hi[i])return null;}else{const x=(lo[i]-eye[i])/d[i],y=(hi[i]-eye[i])/d[i];a=Math.max(a,Math.min(x,y));b=Math.min(b,Math.max(x,y));}}return b>a?[a,b]:null;};
    // A primitive whose derivative is clamp((0.25-y)/fade,0,1).
    const primitive=y=>{if(fade===0)return y;const start=0.25-fade;if(y<=start)return y;const t=Math.min(y,0.25)-start;return start+t-t*t/(2*fade);};
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
      const medium=interval(ray,[1,0,1],[255,0.25,255],distance);let optical=0;
      if(medium){const [a,b]=medium,ya=eye[1]+ray[1]*a,yb=eye[1]+ray[1]*b;const integral=fade===0?b-a:Math.abs(ray[1])<1e-12?(b-a)*Math.max(0,Math.min(1,(0.25-ya)/fade)):(primitive(yb)-primitive(ya))/ray[1];optical=Math.min(80,copies*sigma*integral);}
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
  const hashes={highlands:createHash("sha256").update(highlandsBytes).digest("hex")};for(const path of ["./basin-fog-render-smoke.mjs","./basin-fog-fixture.mjs"]){hashes[path]=createHash("sha256").update(await readFile(new URL(path,import.meta.url))).digest("hex");}
  results.push({case:"reference provenance",hashes,model:"binary64 authored 254 m square prism, floor 0, lid 0.25 m; closed-form ramp integral; configured grade/sRGB",viewport:[1400,900]});
  results.push({case:"zero-length ray limit",distance:0,opticalDepth:0,passing:1,
    note:"The geometric reference has an empty interval. Validated perspective cameras require positive near distance, so captured surface rays do not exercise the shader's zero-distance return."});
  const fixture=document({budget:0.25});fixture.camera=camera;
  fixture.entities[0].terrain.surface.materials[0].color=[0,0,0];
  fixture.materials=[{id:"lamp",color:[0,0,0,1],emissive:L}];fixture.meshes=[{id:"box",source:{kind:"cube"}}];
  fixture.entities.push({id:"wall",mesh:"box",material:"lamp",transform:{translation:[128,10,64],scale:[128,20,1]}});
  fixture.environment.atmosphere={preset:"day",sun_color:[0,0,0],zenith:H,horizon:H,ground:H,ambient:0,rim:0,specular:0,emission:1,exposure:1,saturation:1,contrast:1,vibrance:0,shadow_tint:[1,1,1],highlight_tint:[1,1,1],fog_density:0,aerial_density:0};
  await page.route("**/basin-fog-height.r16*",route=>route.fulfill({body:heightBytes(),contentType:"application/octet-stream"}));
  const scene=routedScene(page,fixture);await scene.route();await openPaused(page,sceneUrl(),{timeout:120000});
  let {state}=await settle(command);near(state.terrains[0].lid_range[0],0.25,0.0001);
  const captures={};
  for(const [name,sigma,fade]of[["off-a",0,3],["off-b",0,3],["ramp",0.025,3],["denser",0.05,3],["hard-top",0.025,0]]){
    await command("environment",{merge:true,basin_fog:{extinction_per_m:sigma,top_fade_m:fade}});await settle(command);
    captures[name]=await capture(page,command,output,name);await reference(name,captures[name],{sigma,fade});
  }
  results.push({case:"descriptive capture changes",offRepeat:await differentPixels(page,[captures["off-a"],captures["off-b"]]),on:await differentPixels(page,[captures["off-a"],captures.ramp]),density:await differentPixels(page,[captures.ramp,captures.denser])});
  for(const [name,position,target]of[["above-lid",[128,4,200],[128,4,64]],["inside",[128,0.125,170],[128,0.125,64]],["below-floor",[128,-1,200],[128,-1,64]],["vertical",[128,3,120],[128,0,120]]]){
    await command("camera",{position,target,vertical_fov_radians:camera.fov_degrees*Math.PI/180,near:camera.near,far:camera.far});await settle(command);await capture(page,command,output,name);results.push({case:name,position,target});
  }
  const overlap=structuredClone(fixture);overlap.entities.push({...structuredClone(overlap.entities[0]),id:"ground1"});
  await scene.reload(command,overlap);await resetCamera();await command("environment",{merge:true,basin_fog:{top_fade_m:3}});({state}=await settle(command));assert.equal(state.terrains.length,2);
  const doubled=await capture(page,command,output,"overlap");await reference("overlap",doubled,{sigma:0.025,fade:3,copies:2});
  // Distinct integration paths retain raw matched captures. These deliberately
  // have no appearance assertions; the optical reference above covers opaque rays.
  const paths=structuredClone(fixture);paths.materials.push({id:"glass",color:[0.8,0.9,1,1],opacity:0.5,alpha_mode:"blend",transmission:1,thickness:1,ior:1.5});
  paths.entities.push({id:"glass",mesh:"box",material:"glass",transform:{translation:[128,2,120],scale:[30,4,1]}});
  paths.materials.push({id:"alpha",color:[0.7,0.2,0.1,1],opacity:0.4},{id:"mask",color:[0.2,0.7,0.1,1],opacity:0.8,alpha_mode:"mask",alpha_cutoff:0.5});
  paths.entities.push({id:"alpha",mesh:"box",material:"alpha",transform:{translation:[100,2,140],scale:[18,4,1]}},{id:"mask",mesh:"box",material:"mask",transform:{translation:[155,2,140],scale:[18,4,1]}},
    {id:"particles",transform:{translation:[128,1,155]},particles:{rate:20,lifetime:2,speed:0,size:[1,1],radius:8}});
  paths.environment.refraction={strength:1};await scene.reload(command,paths);await resetCamera();await command("step",{ticks:60});const integration=await settle(command);
  assert.ok(integration.rendering.particles.drawn>0);assert.equal(integration.rendering.refraction.active,true);results.push({case:"integration paths",rendering:integration.rendering});
  for(const sigma of [0,0.025]){await command("environment",{merge:true,basin_fog:{extinction_per_m:sigma}});await settle(command);await capture(page,command,output,`glass-${sigma}`);}
  for(const view of [false,true]){
    await command("environment",{merge:true,clouds:{enabled:true,coverage:0.5,base:80,thickness:100,in_view:view}});
    const pair=[];
    for(const sigma of [0,0.025]){await command("environment",{merge:true,basin_fog:{extinction_per_m:sigma}});await settle(command);pair.push(await capture(page,command,output,`${view?"view":"dome"}-clouds-${sigma}`));}
    results.push({case:view?"view clouds":"dome clouds",changed:await differentPixels(page,pair),note:"Representative-front fog and nonlinear cloud grading are approximations."});
  }
  const ocean=structuredClone(fixture);ocean.environment.ocean={enabled:true,level:0.5};await scene.reload(command,ocean);
  await command("camera",{position:[128,0.625,200],target:[128,0.625,64],vertical_fov_radians:35*Math.PI/180,near:0.1,far:1000});await settle(command);
  for(const sigma of [0,0.025]){await command("environment",{merge:true,basin_fog:{extinction_per_m:sigma}});await settle(command);await capture(page,command,output,`water-${sigma}`);}
  await command("camera",{position:[128,0.25,200],target:[128,0.25,64],vertical_fov_radians:35*Math.PI/180,near:0.1,far:1000});
  const underwater=[];
  for(const sigma of [0,0.025]){await command("environment",{merge:true,basin_fog:{extinction_per_m:sigma}});await settle(command);underwater.push(await capture(page,command,output,`underwater-${sigma}`));}
  results.push({case:"underwater early return",changed:await differentPixels(page,underwater)});
  // The unchanged Highlands overview, at the catalog viewport/seed/settings.
  await openPaused(page,highlandsUrl("on"),{timeout:180000});
  ({state}=await settle(command));const demo=(await command("inspect",{component:"BasinDemo"})).result.value;
  assert.equal(demo.ready,true);assert.equal(demo.final_edits-demo.original_edits,4);assert.equal(state.dropped_terrains,0);
  assert.equal(demo.final_layers.length-demo.original_layers.length,4);
  assert.deepEqual(demo.final_layers.slice(0,demo.original_layers.length),demo.original_layers);
  const layerIdentity=layers=>createHash("sha256").update(JSON.stringify(layers)).digest("hex");
  const layerHashes={original:layerIdentity(demo.original_layers),enabled:layerIdentity(demo.final_layers)};
  assert.ok(state.terrains.some(t=>t.wet_cells>0));const seen=(await command("inspect",{component:"Camera"})).result.value;
  for(let i=0;i<3;i++){near(seen.position[i],demo.eye[i],0.001);near(seen.target[i],demo.target[i],0.001);}
  near(seen.vertical_fov_radians,55*Math.PI/180,0.00001);near(seen.near,0.5,0.00001);near(seen.far,20000,0.001);
  const rays=await highlandsRays(demo,state,seen);
  results.push({case:"Highlands authored content",catalog:highlands.options,demo,layerHashes,camera:seen,state,rays});
  await capture(page,command,output,"highlands-on");await command("environment",{merge:true,basin_fog:{extinction_per_m:0}});await settle(command);
  const zero=await capture(page,command,output,"highlands-same-content-zero");await command("environment",{merge:true,basin_fog:{extinction_per_m:0.025}});await settle(command);
  const fog=await capture(page,command,output,"highlands-same-content-fog");results.push({case:"Highlands same content",changed:await differentPixels(page,[zero,fog]),state:await inspect(command)});
  for(const t of state.terrains.filter(t=>t.largest)){
    const p=t.largest.deepest,position=[p[0]+t.origin[0],p[1]+t.origin[1]+t.largest.depth_m*.5,p[2]+t.origin[2]];
    await command("camera",{position,target:[position[0]+20,position[1],position[2]],vertical_fov_radians:55*Math.PI/180,near:0.1,far:20000});await settle(command);await capture(page,command,output,"highlands-inside-largest");
    results.push({case:"Highlands inside pool",position,pool:t.largest});break;
  }
  for(const t of state.terrains.filter(t=>t.largest)){
    const p=t.largest.deepest,position=[p[0]+t.origin[0],t.largest.top+t.origin[1]+20,p[2]+t.origin[2]];
    await command("camera",{position,target:[position[0]+20,position[1]-15,position[2]],vertical_fov_radians:55*Math.PI/180,near:0.1,far:20000});await settle(command);await capture(page,command,output,"highlands-above-largest");
    results.push({case:"Highlands above pool",position,pool:t.largest});break;
  }
  await command("camera",{position:demo.eye,target:demo.target,vertical_fov_radians:55*Math.PI/180,near:0.5,far:20000});await command("resume");await page.waitForTimeout(1500);
  await page.locator("canvas").screenshot({path:`${output}/highlands-ordinary-a.png`});await page.waitForTimeout(417);await page.locator("canvas").screenshot({path:`${output}/highlands-ordinary-b.png`});await command("pause");
  const solved=state.terrains.map(t=>({origin:t.origin,dem_hash:t.dem_hash,wet_cells:t.wet_cells,pools:t.pools}));
  await command("game",{name:"workload",args:{config:highlands.options.config}});
  ({state}=await settle(command));const replacement=(await command("inspect",{component:"BasinDemo"})).result.value;
  assert.equal(replacement.ready,true);assert.equal(state.pending_uploads,0);assert.equal(state.dropped_terrains,0);
  assert.equal(layerIdentity(replacement.original_layers),layerHashes.original);assert.equal(layerIdentity(replacement.final_layers),layerHashes.enabled);
  assert.deepEqual(state.terrains.map(t=>({origin:t.origin,dem_hash:t.dem_hash,wet_cells:t.wet_cells,pools:t.pools})),solved);
  results.push({case:"new world current fog upload",demo:replacement,state});
  await capture(page,command,output,"highlands-replaced-world");
  await openPaused(page,highlandsUrl("off"),{timeout:180000});
  let rendering;for(let i=0;i<1200;i++){rendering=(await command("render")).result.rendering;if(!rendering.loading&&!rendering.pipelines_compiling)break;}
  assert.equal(rendering.basin_fog.gpu_bytes,0);await capture(page,command,output,"highlands-original-off");
  assert.equal(errors.length,0,errors.join("\n"));await record(output,"results",results);
  const measurements=[];
  for(const result of results){
    if(result.opticalReference){
      for(const [key,label] of [["meanAbsoluteError","mean absolute RGB error"],["maxError","maximum RGB error"]])measurements.push({name:`${result.case}: ${label}`,value:result.opticalReference[key],unit:"8-bit levels",note:"Independent authored optical reference; descriptive, without an appearance pass threshold."});
      measurements.push({name:`${result.case}: reference-covered pixels`,value:result.opticalReference.count,unit:"pixels"});
    }
    for(const key of ["changed","offRepeat","on","density"]){
      const changed=result[key];if(changed&&typeof changed.changed==="number")measurements.push({name:`${result.case}: ${key}`,value:100*changed.changed/changed.total,unit:"percent pixels",note:"Descriptive capture difference, never an appearance pass threshold."});
    }
  }
  console.log(JSON.stringify({ok:true,results:results.length,measurements,output}));
}catch(error){await record(output,"failure",{error:String(error),errors,results});console.error(error);process.exitCode=3;}
finally{await browser.close();}
