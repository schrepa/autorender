/**
 * The terrain fixture the terrain smokes share (terrain, terrain-surface,
 * terrain-water-sky, terrain-files), opened in its own page for each.
 *
 * fixtures/terrain.scene.json, its heightmap and texture in
 * fixtures/terrain-smoke/, which the page and the tools read beside the
 * scene in the served scenes' terrain-smoke/: a generated terrain of 4 by 4
 * one-meter tiles at (-512, 0, -512). Its layers, in order: noise hills;
 * six rounds of erosion and the hydrology channels, both baked over the
 * two-meter tiles; a twelve-meter trench across the last stretch of a
 * gravel road through the south-western hills, which bridges it; a
 * forty-meter mound a straight road tunnels through; a plateau flattened at
 * (512, 512), terrain-local; a hole 48 m east of it; and an imported dome
 * 200 m high centered at (764, 764). Meadow below, rock on steep ground and
 * summit snow above 120 m; crates and balls placed on the meadow, a quarter
 * of them also on the pinned level-2 tiles; an author's big crate on the
 * plateau and a ball on the level-1 tiles; surroundings to the horizon,
 * their bakes eased out over 128 m. Beside it: the player, a reflection
 * probe, a buoy, an interior's probe, a hiker and a roller.
 */
import assert from "node:assert/strict";
import { cp, mkdir, readFile, rm } from "node:fs/promises";
import { sep } from "node:path";
import { artifacts, commander, launch, openPaused, routedScene, sceneUrl } from "./harness.mjs";
import { workspacePath } from "./workspace.mjs";

/** The fixture scene, as authored. */
export const fixture = JSON.parse(await readFile(new URL("fixtures/terrain.scene.json", import.meta.url), "utf8"));
/** The fixture's terrain, as authored. */
export const terrain = fixture.entities.find(entity => entity.id === "ground").terrain;
/** The camera most checks look from, and return to. */
export const overview = { position: [0, 80, 140], target: [0, 30, 0], vertical_fov_radians: Math.PI / 3, near: 0.5, far: 4000 };
/** The served workspace's scenes, which the development server serves. */
export const scenes = `${workspacePath("web/public/scenes")}${sep}`;
/** The fixture's files where the page and the tools read them. */
export const folder = `${workspacePath("web/public/scenes/terrain-smoke")}${sep}`;
/** Terrain settings that do all tile work in one extraction, so renders
 * settle quickly: every terrain section a check sends starts from them. */
export const fast = { samples_per_frame: 50_000_000 };
/** How long a capture waits for pipelines to compile. */
export const COMPILE_MS = 180_000;

/**
 * Opens the fixture in a new page for the smoke `name` (its captures in
 * artifacts/NAME-smoke/), looking from the overview once its tiles settled.
 * The tools for the page: `command`, the fixture's `scene` (reloads),
 * `settled` (the first settled frame's stats), the helpers below, the
 * page's `errors`, and `close()`, which closes the browser and removes the
 * fixture's files; a failure while opening closes it too.
 */
export async function openTerrain(name) {
  const output = await artifacts(`${name}-smoke`);
  const { browser, page, errors } = await launch();
  const command = commander(page);
  const scene = routedScene(page, fixture);
  const close = async () => {
    await browser.close();
    await rm(folder, { recursive: true, force: true });
  };

  /** Renders until every wanted tile has been resident for a frame (bakes
   * load them all at once); returns the frame's stats. Job workers bake and
   * generate tiles beside the page, their outputs arriving between
   * commands, so renders wait a little for them. */
  async function settle() {
    let rendering;
    let resident = 0;
    for (let frame = 0; frame < 600; frame++) {
      rendering = (await command("render")).result.rendering ?? (await command("status")).result.rendering;
      const ready = rendering.terrain.tiles > 0 && rendering.terrain.loading_tiles === 0 && rendering.terrain.fallback_patches === 0;
      resident = ready ? resident + 1 : 0;
      if (resident > 1) {
        assert.equal(rendering.terrain.unplaced_tiles, 0, `every resident tile holds the objects its level places: ${JSON.stringify(rendering.terrain)}`);
        return rendering;
      }
      if (!ready) await new Promise(resolve => setTimeout(resolve, 25));
    }
    assert.fail(`terrain did not settle: ${JSON.stringify(rendering)}`);
  }

  /** Captures the view once its pipelines compiled into NAME.png; the
   * capture as base64 PNG. */
  async function capture(file) {
    // Pipelines go on compiling for seconds after the frames that first
    // need them (a minute and more on a busy machine), Chrome presenting
    // nothing meanwhile, and a capture presents only once they have.
    const until = Date.now() + COMPILE_MS;
    while ((await command("render")).result.rendering.pipelines_compiling) {
      if (Date.now() > until) assert.fail(`pipelines still compiling after ${COMPILE_MS / 1000} s`);
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    await command("capture");
    const bytes = await page.locator("canvas").screenshot({ path: `${output}/${file}.png` });
    return bytes.toString("base64");
  }

  /** Renders until `done` holds for a frame's stats, read-backs arriving a
   * few frames late; returns them. */
  async function rendersUntil(done, what) {
    let rendering;
    for (let frame = 0; frame < 600; frame++) {
      rendering = (await command("render")).result.rendering;
      if (done(rendering)) return rendering;
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    assert.fail(`${what}: ${JSON.stringify(rendering)}`);
  }

  /** Height query samples at `points` (x, z). */
  async function heights(points) {
    return (await command("terrain_height", { points })).result.samples;
  }

  /** The terrains' state. */
  async function inspectTerrains() {
    return (await command("inspect", { component: "Terrains" })).result.value;
  }

  /** The fixture terrain entity's inspector view. */
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

/** The mean color of the top third of `png` (a base64 PNG capture) in
 * `page`: the sky. */
export async function skyColor(page, png) {
  return page.evaluate(async png => {
    const image = new Image();
    image.src = `data:image/png;base64,${png}`;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.width; canvas.height = image.height;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);
    const { data } = context.getImageData(0, 0, image.width, Math.floor(image.height / 3));
    const sum = [0, 0, 0];
    for (let offset = 0; offset < data.length; offset += 4) {
      sum[0] += data[offset]; sum[1] += data[offset + 1]; sum[2] += data[offset + 2];
    }
    return sum.map(value => value / (data.length / 4));
  }, png);
}
