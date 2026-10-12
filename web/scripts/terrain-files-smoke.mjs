/**
 * The terrain tools' files: an ETR1 bake streams, and an import loads.
 *
 * engine_assetc's `terrain_bake` bakes the fixture's layers into an ETR1
 * file, which streams by byte range with the heights generating gives;
 * `terrain_import` brings the dome in as a RAW heightmap of its own, and
 * the scene it writes loads with the dome's summit, its source and its
 * attribution. Both tools are built with the quick profile by the cargo on
 * PATH while the page opens; they write into web/public/scenes/, and what
 * they wrote is removed afterwards. The fixture: terrain-fixture.mjs.
 */
// @ts-check
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile, rm, writeFile } from "node:fs/promises";
import { cargoBinaries } from "./harness.mjs";
import { fast, fixture, folder, openTerrain, overview, scenes, terrain } from "./terrain-fixture.mjs";

const tools = cargoBinaries(["-p", "engine_assetc", "--profile", "quick", "--bin", "terrain_bake", "--bin", "terrain_import"]);
// The fixture as a file, which the bake reads.
const source = `${scenes}terrain-smoke.scene.json`;
const opened = await openTerrain("terrain-files");
const { errors, command, scene, settle, heights } = opened;
// What the tools wrote goes with the fixture's files.
async function close() {
  await opened.close();
  await rm(source, { force: true });
  await rm(`${scenes}terrain-smoke-import`, { recursive: true, force: true });
  await rm(`${scenes}terrain-smoke-import.scene.json`, { force: true });
}
const checks = [];

try {
  // An ETR1 file of the same layers streams by byte range and matches; the
  // bake reads the heightmap beside the scene, as the page does.
  await writeFile(source, JSON.stringify(fixture));
  execFileSync((await tools).terrain_bake, [source, "ground", `${folder}ground.etr`], { stdio: "inherit" });
  const generated = await heights([[10, -7], [-200, 150], [300, -310]]);
  const filed = structuredClone(fixture);
  filed.entities[1].terrain = { origin: terrain.origin, file: "terrain-smoke/ground.etr" };
  await scene.reload(command, filed);
  await command("environment", { terrain: fast });
  await command("camera", overview);
  let streamed;
  for (let attempt = 0; attempt < 60; attempt++) {
    streamed = (await command("render")).result.rendering;
    if (streamed.terrain.tiles > 0 && streamed.terrain.loading_tiles === 0 && streamed.terrain.fallback_patches === 0) break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.equal(streamed.terrain.loading_tiles, 0, "file tiles arrive");
  const read = await heights([[10, -7], [-200, 150], [300, -310]]);
  for (const [index, sample] of read.entries()) {
    assert.ok(sample && generated[index], `point ${index} resident from both`);
    if (sample.level === generated[index].level) {
      assert.ok(Math.abs(sample.height - generated[index].height) < 2e-3, `file height ${sample.height} vs ${generated[index].height}`);
    }
  }
  checks.push({ check: "ETR1 file streaming", tiles: streamed.terrain.tiles });

  // The import tool reads the dome as a RAW heightmap of its
  // own, centered on its middle without detail noise, and the scene it
  // writes loads with the dome's summit, its source and its attribution.
  execFileSync((await tools).terrain_import, [
    "--out", scenes, "--id", "terrain-smoke-import", "--size", "1024", "--extent", "128,128", "--range", "0,200",
    "--detail", "0", "--attribution", "smoke fixture", `${folder}dome.r16`], { stdio: "inherit" });
  const report = JSON.parse(await readFile(`${scenes}terrain-smoke-import/import.json`, "utf8"));
  assert.ok(report.samples === 513 && Math.abs(report.highest - 200) < 1e-6 && report.filled_gaps > 0, `the import resamples and fills: ${JSON.stringify(report)}`);
  // Reloads keep the scene's ID.
  const imported = { ...JSON.parse(await readFile(`${scenes}terrain-smoke-import.scene.json`, "utf8")), id: fixture.id };
  await scene.reload(command, imported);
  await command("environment", { terrain: fast });
  await command("camera", overview);
  await settle();
  const [summit] = await heights([[0, 0]]);
  assert.ok(summit && Math.abs(summit.height - 200) < 0.01, `the imported summit stands 200 m high: ${JSON.stringify(summit)}`);
  const importedView = (await command("inspect", { entity: "ground", component: "Terrain" })).result.entity.components.Terrain;
  const importedMap = importedView.terrain.layers[0].heightmap;
  assert.ok(importedMap.attribution === "smoke fixture" && importedMap.source.includes("dome.r16"), `the layer keeps its source and credit: ${JSON.stringify(importedMap)}`);
  assert.equal(importedView.terrain.layers.length, 2, "without detail, the import and its rivers");
  checks.push({ check: "import", samples: report.samples, summit: summit.height, filled: report.filled_gaps });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, checks }));
} finally {
  await close();
}
