/** Fixed DEM conservation/topology, revision, admission and lifecycle checks.
 * Appearance is recorded by the separate render check without image limits. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { artifacts,commander,launch } from "./harness.mjs";
import { document,heightBytes,session,settle,near,record } from "./basin-fog-fixture.mjs";
const output=await artifacts("basin-fog-topology-smoke");
const fixtureFolder=new URL(`../public/scenes/basin-fog-topology-${process.pid}/`,import.meta.url);
const {browser,page,errors}=await launch({width:800,height:600,timeout:120000});
const command=commander(page,{deadline:120000});const checks=[];
const inspect=async command=>(await command("inspect",{component:"basin_fog"})).result.value;
try{
  const scene=await session(page,command);
  console.log("Authored scene host ready.");
  let {state,rendering}=await settle(command),t=state.terrains[0];
  console.log("Initial fog map and upload ready.");
  assert.equal(t.size[0],256);assert.equal(t.size[1],256);assert.equal(t.wet_cells,64516);assert.equal(t.pools,1);
  near(t.input_m3,16384);near(t.retained_m3,16129);near(t.drained_m3,255);near(t.lid_range[0],0.25,0.0001);
  assert.ok(rendering.basin_fog.gpu_bytes>0);assert.equal(state.pending_jobs,0);assert.equal(state.pending_uploads,0);
  const sample=async(points,terrain="ground")=>(await command("basin_fog_sample",{terrain,points})).result.samples;
  const map=async()=>{
    const labels=new Map(),hash=createHash("sha256");let wet=0,volume=0;
    for(let z=0;z<256;z++){
      const row=await sample(Array.from({length:256},(_,x)=>[x+.5,z+.5]));
      for(const cell of row){
        assert.ok(cell&&Number.isFinite(cell.floor)&&Number.isFinite(cell.top)&&cell.top>=cell.floor);
        if(cell.pool){wet++;volume+=cell.depth;if(labels.has(cell.pool))assert.equal(cell.top,labels.get(cell.pool));else labels.set(cell.pool,cell.top);}
        else assert.equal(cell.depth,0);
      }
      hash.update(JSON.stringify(row));
    }
    return {sha256:hash.digest("hex"),wet,volume,lids:Array.from(labels)};
  };
  let samples=await sample([[100.5,100.5],[0.5,0.5],[-1,100]]);
  near(samples[0].floor,0,0.0001);near(samples[0].top,0.25,0.0001);assert.equal(samples[1].pool,0);assert.equal(samples[2],null);
  const partial=await map();assert.equal(partial.wet,t.wet_cells);near(partial.volume,t.published_retained_m3);
  checks.push({case:"single partial",state,samples,map:partial});
  console.log("Single-bowl full-map conservation checked.");
  const revision=t.solve_revision,source=t.source_revision,pool=samples[0].pool;
  await command("environment",{merge:true,basin_fog:{extinction_per_m:0.05,top_fade_m:0}});
  ({state}=await settle(command));assert.equal(state.terrains[0].solve_revision,revision);
  assert.equal((await sample([[100.5,100.5]]))[0].pool,pool);
  await command("camera",{position:[140,140,300],target:[128,0,128],vertical_fov_radians:55*Math.PI/180,near:0.1,far:1000});
  ({state}=await settle(command));assert.equal(state.terrains[0].solve_revision,revision);
  await command("environment",{merge:true,basin_fog:{budget_m:2}});
  ({state}=await settle(command));t=state.terrains[0];near(t.retained_m3,64516);near(t.drained_m3,66556);near(t.lid_range[0],1,0.0001);
  assert.equal(t.source_revision,source);assert.notEqual(t.solve_revision,revision);checks.push({case:"refill spill",state});
  for(const [budget,pools,retained,top] of [[0.5,2,32258,null],[1.25,1,80645,1.2578740157480315]]){
    await scene.reload({ridge:true,budget});({state}=await settle(command));t=state.terrains[0];
    assert.equal(t.pools,pools);near(t.retained_m3,retained);near(t.input_m3,t.retained_m3+t.drained_m3);
    if(top!==null)near(t.lid_range[0],top,0.0001);
    samples=await sample([[100.5,100.5],[127.5,100.5],[170.5,100.5]]);
    if(pools===2){assert.notEqual(samples[0].pool,samples[2].pool);assert.equal(samples[1].pool,0);}
    else assert.equal(samples[0].pool,samples[2].pool);
    const all=await map();assert.equal(all.wet,t.wet_cells);assert.equal(all.lids.length,pools);near(all.volume,t.published_retained_m3);
    checks.push({case:`two bowls ${budget}`,state,samples,map:all});
    console.log(`Two-bowl full-map budget ${budget} checked.`);
  }
  await scene.reload({hole:true});({state}=await settle(command));t=state.terrains[0];
  assert.equal(t.wet_cells,0);near(t.retained_m3,0);near(t.input_m3,t.drained_m3);checks.push({case:"hole drains",state});
  samples=await sample([[64.5,64.5]]);assert.equal(samples[0].valid,false);
  await scene.reload({sea:0.1});({state}=await settle(command));samples=await sample([[100.5,100.5]]);
  near(samples[0].floor,0.1,0.0001);near(samples[0].top,0.35,0.0001);checks.push({case:"sea effective floor",state,samples});
  const seaRevision=state.terrains[0].source_revision;
  await command("environment",{merge:true,ocean:{enabled:false}});({state}=await settle(command));samples=await sample([[100.5,100.5]]);
  near(samples[0].floor,0,0.0001);near(samples[0].top,0.25,0.0001);assert.notEqual(state.terrains[0].source_revision,seaRevision);
  await scene.reload({hydrology:true});({state}=await settle(command));
  const wetTerrain=(await command("terrain_height",{points:[[100.5,100.5]]})).result.samples[0];
  assert.ok(wetTerrain?.standing_water?.depth>0,"the hydrology fixture must actually hold water");
  samples=await sample([[100.5,100.5]]);near(samples[0].floor,wetTerrain.standing_water.surface,0.0001);
  const waterRevision=state.terrains[0].source_revision;
  checks.push({case:"rendered hydrology effective floor",state,wetTerrain,samples});
  await command("environment",{merge:true,water:{enabled:false}});({state}=await settle(command));
  samples=await sample([[100.5,100.5]]);near(samples[0].floor,0,0.0001);near(samples[0].top,0.25,0.0001);
  assert.notEqual(state.terrains[0].source_revision,waterRevision);
  await command("environment",{merge:true,water:{enabled:true}});({state}=await settle(command));
  samples=await sample([[100.5,100.5]]);near(samples[0].floor,wetTerrain.standing_water.surface,0.0001);
  checks.push({case:"Water off/on invalidation",state,samples});
  console.log("Hole, Ocean and hydrology floors checked.");
  await scene.reload({});await settle(command);const repeated=await map();assert.deepEqual(repeated,partial);
  // Supersede an ordinary edit before it can finish, then undo both. The
  // published map must follow the newest layers, never a cancelled solution.
  const beforeEdit=(await inspect(command)).terrains[0].source_revision;
  await command("environment",{merge:true,terrain:{samples_per_frame:1,loading_work_ms:0.1}});
  await command("terrain_edit",{terrain:"ground",id:"fog-first",center:[100,100],radius:12,change:{raise:3}});
  const duringEdit=await inspect(command);
  assert.ok(duringEdit.terrains.every(t=>t.state!=="ready"));assert.equal((await sample([[100.5,100.5]]))[0],null);
  await command("terrain_edit",{terrain:"ground",id:"fog-newest",center:[100,100],radius:12,change:{flatten:5}});
  await command("environment",{merge:true,terrain:{samples_per_frame:50000000,loading_work_ms:50}});
  ({state}=await settle(command));samples=await sample([[100.5,100.5]]);
  near(samples[0].floor,5,0.0001);assert.notEqual(state.terrains[0].source_revision,beforeEdit);
  assert.equal(state.terrains[0].state,"ready");checks.push({case:"superseded geometry",duringEdit,state,samples});
  await command("terrain_undo",{terrain:"ground"});await command("terrain_undo",{terrain:"ground"});await settle(command);
  assert.deepEqual(await map(),partial);
  await scene.reload({origin:[20,7,-30],copies:2});({state}=await settle(command));
  assert.equal(state.terrains.length,2);for(t of state.terrains)near(t.retained_m3,16129);
  samples=await sample([[120.5,70.5]]);near(samples[0].floor,7,0.0001);near(samples[0].top,7.25,0.0001);
  checks.push({case:"translated overlapping independent media",state,samples});
  const admittedBytes=state.terrains[0].requested_bytes;
  await command("environment",{merge:true,basin_fog:{max_bytes:admittedBytes}});({state}=await settle(command,v=>v.dropped_terrains===1&&v.terrains.some(t=>t.state==="ready")));
  assert.equal(state.terrains[0].state,"ready");assert.equal(state.terrains[1].state,"dropped");checks.push({case:"equal-distance entity priority",state});
  await command("environment",{merge:true,basin_fog:{max_bytes:0}});
  ({state,rendering}=await settle(command,v=>v.dropped_terrains===2));assert.equal(state.pending_jobs,0);assert.equal(rendering.basin_fog.nodes,0);
  await command("environment",{merge:true,basin_fog:{max_bytes:null}});({state}=await settle(command));assert.equal(state.dropped_terrains,0);
  for(let cycle=0;cycle<3;cycle++){
    await command("environment",{merge:true,basin_fog:{enabled:false}});({state,rendering}=await settle(command,v=>v.terrains.length===0));
    assert.equal(rendering.basin_fog.gpu_bytes,0);assert.equal(rendering.basin_fog.upload_bytes,0);assert.equal(state.requested_bytes,0);
    checks.push({case:`off ${cycle}`,state,rendering:rendering.basin_fog});
    await command("environment",{merge:true,basin_fog:{enabled:true}});const on=await settle(command);checks.push({case:`on ${cycle}`,state:on.state,rendering:on.rendering.basin_fog});
  }
  await command("environment",{merge:true,basin_fog:{budget_m:0}});({state,rendering}=await settle(command,v=>v.terrains.length===0));assert.equal(rendering.basin_fog.gpu_bytes,0);
  await command("environment",{merge:true,basin_fog:{budget_m:-1}},false);
  await command("environment",{merge:true,basin_fog:{level:999}},false);
  const invalid=document();invalid.entities[0].terrain.tiles=[4294967295,4294967295];invalid.revision=scene.scene.revision+1;
  scene.scene.serve(invalid);const refused=await command("reload",{expected_revision:String(scene.scene.revision),revision:String(invalid.revision)},false);
  checks.push({case:"overflowing dimensions refused by scene validation",error:refused.error});
  // Use the actual public file baker, as the terrain regression check does.
  // Its generated artifact goes through the ordinary range loader unchanged.
  await mkdir(fixtureFolder,{recursive:true});
  const fileDocument=document();fileDocument.entities[0].terrain.layers[0].heightmap.uri="height.r16";
  await writeFile(new URL("height.r16",fixtureFolder),heightBytes());await writeFile(new URL("source.scene.json",fixtureFolder),JSON.stringify(fileDocument));
  execFileSync("cargo",["run","-q","-p","engine_assetc","--release","--bin","terrain_bake","--",fileURLToPath(new URL("source.scene.json",fixtureFolder)),"ground",fileURLToPath(new URL("ground.etr",fixtureFolder))],{stdio:"inherit"});
  fileDocument.entities[0].terrain={origin:[0,0,0],file:`basin-fog-topology-${process.pid}/ground.etr`};
  await scene.scene.reload(command,fileDocument);({state,rendering}=await settle(command,v=>v.unsupported_terrains===1));
  assert.equal(rendering.basin_fog.gpu_bytes,0);assert.equal(state.pending_jobs,0);assert.equal((await sample([[100.5,100.5]]))[0],null);
  checks.push({case:"file-backed terrain explicitly unsupported",state});
  assert.equal(errors.length,0,errors.join("\n"));
  await record(output,"checks",checks);console.log(JSON.stringify({ok:true,checks:checks.length,output}));
}catch(error){await record(output,"failure",{error:String(error),errors,checks});console.error(error);process.exitCode=3;}
finally{
  await browser.close();
  for(const name of ["height.r16","source.scene.json","ground.etr"])await rm(new URL(name,fixtureFolder),{force:true});
  // Only these three owned files are removed; no recursive directory deletion.
}
