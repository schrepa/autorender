/**
 * The terrain tools' files: an import loads.
 *
 * engine_assetc's `terrain_import` brings the dome in as a RAW heightmap of
 * its own, and the scene it writes loads with the dome's summit, its
 * source and its attribution, its tiles generated from that heightmap as
 * every terrain's are. The tool is built with the quick profile by the
 * cargo on PATH while the page opens; it writes into web/public/scenes/,
 * and what it wrote is removed afterwards. The fixture:
 * terrain-fixture.mjs.
 */
// @ts-check
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile, rm } from "node:fs/promises";
import { cargoBinaries } from "./harness.mjs";
import { fast, fixture, folder, openTerrain, overview, scenes } from "./terrain-fixture.mjs";

const tools = cargoBinaries(["-p", "engine_assetc", "--profile", "quick", "--bin", "terrain_import"]);
const opened = await openTerrain("terrain-files");
const { errors, command, scene, settle, heights } = opened;
// What the tool wrote goes with the fixture's files.
async function close() {
  await opened.close();
  await rm(`${scenes}terrain-smoke-import`, { recursive: true, force: true });
  await rm(`${scenes}terrain-smoke-import.scene.json`, { force: true });
}
const checks = [];

try {
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
