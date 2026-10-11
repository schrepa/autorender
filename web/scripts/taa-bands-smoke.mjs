/** Per-band TAA preserves content through history resets, ordinary motion
 * and renderer reload; its five targets follow toggles and canvas resizing,
 * while fresh exact and CPU frames allocate no temporal targets. */
import assert from "node:assert/strict";
import { checkHistory } from "./taa-history-checks.mjs";

const previous = new WeakMap();
const labels = ["engine/taa-motion", "engine/taa-history", "engine/taa-coarse-history"];
try {
  await checkHistory("taa-bands-smoke", {
    async install(context) {
      await context.addInitScript(labels => {
        const records = new Map();
        const ids = new WeakMap();
        let next = 0;
        const epoch = crypto.randomUUID();
        const create = GPUDevice.prototype.createTexture;
        GPUDevice.prototype.createTexture = function (...args) {
          const texture = Reflect.apply(create, this, args);
          const descriptor = args[0];
          if (labels.includes(descriptor.label)) {
            const id = ++next;
            const extent = descriptor.size;
            const size = Symbol.iterator in Object(extent) ? Array.from(extent)
              : [extent.width, extent.height ?? 1, extent.depthOrArrayLayers ?? 1];
            ids.set(texture, id);
            records.set(id, { id, label: descriptor.label, size: [size[0], size[1] ?? 1, size[2] ?? 1],
              format: descriptor.format, mipLevelCount: descriptor.mipLevelCount ?? 1,
              sampleCount: descriptor.sampleCount ?? 1, destroyed: false });
          }
          return texture;
        };
        const destroy = GPUTexture.prototype.destroy;
        GPUTexture.prototype.destroy = function (...args) {
          const result = Reflect.apply(destroy, this, args);
          const record = records.get(ids.get(this));
          if (record) record.destroyed = true;
          return result;
        };
        // Only plain descriptors and integer IDs escape the weak resource map.
        window.__taaTargets = () => ({ epoch, records: [...records.values()] });
      }, labels);
    },
    async observe(page, label, expected) {
      const snapshot = await page.evaluate(() => ({ ...window.__taaTargets(),
        size: [document.querySelector("canvas").width, document.querySelector("canvas").height] }));
      const live = snapshot.records.filter(record => !record.destroyed);
      const prior = previous.get(page);
      const sameDocument = prior?.epoch === snapshot.epoch;
      if (expected === "none") assert.equal(snapshot.records.length, 0, `${label}: no temporal allocation`);
      if (expected === "inactive") assert.equal(live.length, 0, `${label}: disabling destroys every temporal texture`);
      if (expected === "active") {
        assert.equal(live.length, 5, `${label}: exactly five temporal textures`);
        for (const [index, name] of labels.entries()) {
          const textures = live.filter(record => record.label === name);
          assert.equal(textures.length, index === 0 ? 1 : 2, `${label}: ${name} count`);
          for (const texture of textures) {
            assert.deepEqual(texture.size, [...snapshot.size, 1], `${label}: texture size`);
            assert.equal(texture.mipLevelCount, 1, `${label}: one mip`);
            assert.equal(texture.sampleCount, 1, `${label}: one sample`);
            assert.equal(texture.format, index === 0 ? "rg16float" : "rgba8unorm-srgb", `${label}: format`);
          }
        }
        snapshot.target_bytes = live.reduce((sum, texture) => sum + texture.size[0] * texture.size[1] * 4, 0);
        assert.equal(snapshot.target_bytes, 20 * snapshot.size[0] * snapshot.size[1], `${label}: target storage`);
        if (sameDocument && (prior.live.length === 0 || prior.size.some((value, i) => value !== snapshot.size[i]))) {
          assert.ok(live.every(record => !prior.records.some(old => old.id === record.id)), `${label}: a fresh target set`);
          for (const old of prior.live) assert.equal(snapshot.records.find(record => record.id === old.id)?.destroyed,
            true, `${label}: resize destroys old target ${old.id}`);
        }
      }
      if (expected === "retained") {
        assert.ok(sameDocument, `${label}: transition uses the same renderer`);
        assert.equal(snapshot.records.length, prior.records.length, `${label}: exact/CPU transition allocates no new targets`);
      }
      previous.set(page, { ...snapshot, live });
      return { label, expected, ...snapshot };
    },
  });
} catch (error) {
  console.error(error);
  process.exitCode = error?.code === "ERR_ASSERTION" ? 3 : 1;
}
