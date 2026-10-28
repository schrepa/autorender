// @ts-check
import assert from "node:assert/strict";
import { cp, mkdir, readFile, rm } from "node:fs/promises";
import { sep } from "node:path";
import { artifacts, commander, launch, openPaused, routedScene, sceneUrl } from "./harness.mjs";
import { workspacePath } from "./workspace.mjs";

export const fixture = JSON.parse(await readFile(new URL("fixtures/terrain.scene.json", import.meta.url), "utf8"));
export const terrain = fixture.entities.find(entity => entity.id === "ground").terrain;
export const overview = { position: [0, 80, 140], target: [0, 30, 0], vertical_fov_radians: Math.PI / 3, near: 0.5, far: 4000 };
export const scenes = `${workspacePath("web/public/scenes")}${sep}`;
export const folder = `${workspacePath("web/public/scenes/terrain-smoke")}${sep}`;
export const fast = { samples_per_frame: 50_000_000 };
export const COMPILE_MS = 180_000;

export async function openTerrain(name) {
  const output = await artifacts(`${name}-smoke`);
  const { browser, page, errors } = await launch();
  const command = commander(page);
  const scene = routedScene(page, fixture);
  const close = async () => {
    await browser.close();
    await rm(folder, { recursive: true, force: true });
  };

  async function settle() {
    let rendering;
    let resident = 0;
    for (let frame = 0; frame < 600; frame++) {
      rendering = (await command("render")).result.rendering ?? (await command("status")).result.rendering;
      const ready = rendering.terrain.tiles > 0 && rendering.terrain.loading_tiles === 0 && rendering.terrain.fallback_tiles === 0;
      resident = ready ? resident + 1 : 0;
      if (resident > 1) {
        return rendering;
      }
      if (!ready) await new Promise(resolve => setTimeout(resolve, 25));
    }
    assert.fail(`terrain did not settle: ${JSON.stringify(rendering)}`);
  }

  async function capture(file) {
    const until = Date.now() + COMPILE_MS;
    while ((await command("render")).result.rendering.pipelines_compiling) {
      if (Date.now() > until) assert.fail(`pipelines still compiling after ${COMPILE_MS / 1000} s`);
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    await command("capture");
    const bytes = await page.locator("canvas").screenshot({ path: `${output}/${file}.png` });
    return bytes.toString("base64");
  }

  async function rendersUntil(done, what) {
    let rendering;
    for (let frame = 0; frame < 600; frame++) {
      rendering = (await command("render")).result.rendering;
      if (done(rendering)) return rendering;
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    assert.fail(`${what}: ${JSON.stringify(rendering)}`);
  }

  async function heights(points) {
    return (await command("terrain_height", { points })).result.samples;
  }

  async function inspectTerrains() {
    return (await command("inspect", { component: "Terrains" })).result.value;
  }

  async function inspectGround() {
    return (await command("inspect", { entity: "ground", component: "Terrain" })).result.entity.components.Terrain;
  }

  try {
    await mkdir(folder, { recursive: true });
    await cp(new URL("fixtures/terrain-smoke/", import.meta.url), folder, { recursive: true });
    await scene.route();
    await openPaused(page, sceneUrl());
    const opened = await command("environment", { terrain: fast });
    assert.deepEqual(opened.result.environment_changed, ["terrain"]);
    await command("camera", overview);
    const settled = await settle();
    return {
      page, errors, command, scene, output, settled, close,
      settle, capture, rendersUntil, heights, inspectTerrains, inspectGround,
    };
  } catch (error) {
    await close();
    throw error;
  }
}

export async function skyColor(page, png) {
  return page.evaluate(async png => {
    const image = new Image();
    image.src = `data:image/png;base64,${png}`;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.width; canvas.height = image.height;
    const context = /** @type {CanvasRenderingContext2D} */ (canvas.getContext("2d"));
    context.drawImage(image, 0, 0);
    const { data } = context.getImageData(0, 0, image.width, Math.floor(image.height / 3));
    const sum = [0, 0, 0];
    for (let offset = 0; offset < data.length; offset += 4) {
      sum[0] += data[offset]; sum[1] += data[offset + 1]; sum[2] += data[offset + 2];
    }
    return sum.map(value => value / (data.length / 4));
  }, png);
}
