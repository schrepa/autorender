/** Fixed DEM conservation/topology, revision, admission and lifecycle checks.
 * Appearance is recorded by the separate render check without image limits. */
import assert from "node:assert/strict";
import { artifacts,commander,launch } from "./harness.mjs";
import { session,inspect,settle,near,record } from "./basin-fog-fixture.mjs";
const output=await artifacts("basin-fog-topology-smoke");
const {browser,page,errors}=await launch({width:800,height:600,timeout:120000});
const command=commander(page,{deadline:120000});const checks=[];
try{
  const scene=await session(page,command);
  let {state,rendering}=await settle(command),t=state.terrains[0];
  assert.equal(t.size[0],256);assert.equal(t.size[1],256);assert.equal(t.wet_cells,64516);assert.equal(t.pools,1);
  near(t.input_m3,16384);near(t.retained_m3,16129);near(t.drained_m3,255);near(t.lid_range[0],0.25,0.0001);
  assert.ok(rendering.basin_fog.gpu_bytes>0);assert.equal(state.pending_jobs,0);assert.equal(state.pending_uploads,0);
  const sample=async(points,terrain="ground")=>(await command("basin_fog_sample",{terrain,points})).result.samples;
  let samples=await sample([[100.5,100.5],[0.5,0.5],[-1,100]]);
  near(samples[0].floor,0,0.0001);near(samples[0].top,0.25,0.0001);assert.equal(samples[1].pool,0);assert.equal(samples[2],null);
  checks.push({case:"single partial",state,samples});
  const revision=t.solve_revision,source=t.source_revision,pool=samples[0].pool;
  await command("environment",{basin_fog:{extinction_per_m:0.05,top_fade_m:0}});
  ({state}=await settle(command));assert.equal(state.terrains[0].solve_revision,revision);
  assert.equal((await sample([[100.5,100.5]]))[0].pool,pool);
  await command("camera",{position:[140,140,300],target:[128,0,128],fov_degrees:55,near:0.1,far:1000});
  ({state}=await settle(command));assert.equal(state.terrains[0].solve_revision,revision);
  await command("environment",{basin_fog:{budget_m:2}});
  ({state}=await settle(command));t=state.terrains[0];near(t.retained_m3,64516);near(t.drained_m3,66556);near(t.lid_range[0],1,0.0001);
  assert.equal(t.source_revision,source);assert.notEqual(t.solve_revision,revision);checks.push({case:"refill spill",state});
  for(const [budget,pools,retained,top] of [[0.5,2,32258,null],[1.25,1,80645,1.2578740157480315]]){
    await scene.reload({ridge:true,budget});({state}=await settle(command));t=state.terrains[0];
    assert.equal(t.pools,pools);near(t.retained_m3,retained);near(t.input_m3,t.retained_m3+t.drained_m3);
    if(top!==null)near(t.lid_range[0],top,0.0001);
    samples=await sample([[100.5,100.5],[127.5,100.5],[170.5,100.5]]);
    if(pools===2){assert.notEqual(samples[0].pool,samples[2].pool);assert.equal(samples[1].pool,0);}
    else assert.equal(samples[0].pool,samples[2].pool);
    checks.push({case:`two bowls ${budget}`,state,samples});
  }
  await scene.reload({hole:true});({state}=await settle(command));t=state.terrains[0];
  assert.equal(t.wet_cells,0);near(t.retained_m3,0);near(t.input_m3,t.drained_m3);checks.push({case:"hole drains",state});
  await scene.reload({sea:0.1});({state}=await settle(command));samples=await sample([[100.5,100.5]]);
  near(samples[0].floor,0.1,0.0001);near(samples[0].top,0.35,0.0001);checks.push({case:"sea effective floor",state,samples});
  await scene.reload({origin:[20,7,-30],copies:2});({state}=await settle(command));
  assert.equal(state.terrains.length,2);for(t of state.terrains)near(t.retained_m3,16129);
  samples=await sample([[120.5,70.5]]);near(samples[0].floor,7,0.0001);near(samples[0].top,7.25,0.0001);
  checks.push({case:"translated overlapping independent media",state,samples});
  await command("environment",{basin_fog:{max_bytes:0}});
  ({state,rendering}=await settle(command,v=>v.dropped_terrains===2));assert.equal(state.pending_jobs,0);assert.equal(rendering.basin_fog.nodes,0);
  await command("environment",{basin_fog:{max_bytes:null}});({state}=await settle(command));assert.equal(state.dropped_terrains,0);
  for(let cycle=0;cycle<3;cycle++){
    await command("environment",{basin_fog:{enabled:false}});({state,rendering}=await settle(command,v=>v.terrains.length===0));
    assert.equal(rendering.basin_fog.gpu_bytes,0);assert.equal(rendering.basin_fog.upload_bytes,0);assert.equal(state.requested_bytes,0);
    await command("environment",{basin_fog:{enabled:true}});await settle(command);
  }
  await command("environment",{basin_fog:{budget_m:0}});({state,rendering}=await settle(command,v=>v.terrains.length===0));assert.equal(rendering.basin_fog.gpu_bytes,0);
  await command("environment",{basin_fog:{budget_m:-1}},false);
  await command("environment",{basin_fog:{level:999}},false);
  assert.equal(errors.length,0,errors.join("\n"));
  await record(output,"checks",checks);console.log(JSON.stringify({ok:true,checks:checks.length,output}));
}catch(error){await record(output,"failure",{error:String(error),errors,checks});console.error(error);process.exitCode=3;}
finally{await browser.close();}
