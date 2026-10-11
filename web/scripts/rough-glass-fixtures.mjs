/** Authored scenes for rough-glass-smoke. No workload or benchmark inputs. */
export function glassFixture({ constant = false, blur = 12, glass = true } = {}) {
  const surface = { roughness: 1, metalness: 0, emission: 0, ripples: 0 };
  return {
    version: 9, id: "rough-glass-fixture", revision: 1,
    camera: { position: [0, 0, 8], target: [0, 0, 0], fov_degrees: 45, near: 0.1, far: 500 },
    materials: [
      { id: "background", color: [1, 1, 1, 1], surface, emissive: [0.15, 0.25, 0.35] },
      { id: "bright", color: [1, 1, 1, 1], surface, emissive: [0.8, 0.6, 0.2] },
      { id: "dark", color: [1, 1, 1, 1], surface, emissive: [0.02, 0.03, 0.04] },
      { id: "glass", color: [0.9, 0.95, 1, 1], surface, transmission: 1, ior: 1.5, thickness: 0 },
    ],
    meshes: [{ id: "box", source: { kind: "cube" } }, { id: "pane", source: { kind: "inline",
      vertices: [
        { position: [-0.5, -0.5, 0], normal: [0, 0, 1] }, { position: [0.5, -0.5, 0], normal: [0, 0, 1] },
        { position: [0.5, 0.5, 0], normal: [0, 0, 1] }, { position: [-0.5, 0.5, 0], normal: [0, 0, 1] },
      ], indices: [0, 1, 2, 0, 2, 3] } }],
    environment: {
      atmosphere: { preset: "day", sun_color: [0, 0, 0], zenith: [0.1, 0.1, 0.1], horizon: [0.1, 0.1, 0.1],
        ground: [0.1, 0.1, 0.1], ambient: 0, specular: 0, rim: 0, emission: 1,
        exposure: 1, saturation: 1, contrast: 1, vibrance: 0, shadow_tint: [1, 1, 1], highlight_tint: [1, 1, 1],
        fog_density: 0, aerial_density: 0 },
      shadows: { enabled: false }, ambient_occlusion: { intensity: 0 }, contact_shadows: { enabled: false },
      bloom: { intensity: 0 }, reflections: { intensity: 0 }, temporal_antialiasing: { enabled: false },
      sun_shafts: { intensity: 0 }, refraction: { strength: 1, blur },
    },
    entities: [
      { id: "background", mesh: "box", material: "background", transform: { translation: [0, 0, -2], scale: [100, 100, 0.1] } },
      ...(!constant ? Array.from({ length: 32 }, (_, i) => ({ id: `stripe-${i}`, mesh: "box", material: i % 2 ? "bright" : "dark",
        transform: { translation: [(i - 15.5) * 0.25, 0, -1.8], scale: [0.25, 12, 0.1] } })) : []),
      ...(glass ? [{ id: "glass", mesh: "pane", material: "glass", transform: { translation: [0, 0, 0], scale: [6, 6, 1] } }] : []),
    ],
  };
}

export function geometryFixture(kind) {
  const fixture = glassFixture();
  fixture.id = `rough-glass-${kind}`;
  const pane = fixture.entities.find(entity => entity.id === "glass");
  const material = fixture.materials.find(material => material.id === "glass");
  if (kind === "tilted-thick") {
    material.thickness = 1.5;
    pane.transform.rotation_degrees = [17, 42, 0];
  } else if (kind === "screen-edge") {
    material.thickness = 8;
    pane.transform = { translation: [2.5, 0, 0], rotation_degrees: [0, 55, 0], scale: [8, 8, 1] };
  } else if (kind === "foreground-edge") {
    fixture.entities.push({ id: "foreground", mesh: "box", material: "bright",
      transform: { translation: [-1.5, 0, 1], scale: [3, 6, 0.3] } });
  }
  return fixture;
}

export function mixedFixture() {
  const fixture = glassFixture();
  fixture.id = "rough-glass-mixed";
  fixture.camera = { position: [0, 45, 100], target: [0, 10, 0], fov_degrees: 55, near: 0.1, far: 1500 };
  // A closed depression before hydrology guarantees standing water, rather
  // than relying on a particular noise seed's chance of forming a lake.
  fixture.entities = [{ id: "ground", terrain: {
    origin: [-128, 0, -128], tiles: [1, 1], spacing: 1,
    layers: [
      { stamp: { center: [128, 128], radius: 190, op: { flatten: 12 } } },
      { stamp: { center: [128, 128], radius: 58, falloff: 20, op: { lower: 8 } } },
      { hydrology: { level: 0, channel_area: 5000, depth: 0.5, width: 2, max_depth: 2, max_width: 8 } },
    ],
  } },
  { id: "glass", mesh: "pane", material: "glass", transform: { translation: [-8, 25, 50], rotation_degrees: [-15, 0, 0], scale: [65, 45, 1] } },
  { id: "glass-overlap", mesh: "pane", material: "glass", transform: { translation: [8, 22, 40], rotation_degrees: [-15, 15, 0], scale: [65, 45, 1] } },
  { id: "decal", decal: { material: "bright", opacity: 0.8 }, transform: { translation: [-45, 4, 25], scale: [20, 4, 20] } },
  ...["alpha", "additive"].map((blend, i) => ({ id: `particles-${blend}`, particles: {
    blend, rate: 100, lifetime: 3, speed: 2, radius: 4, size: [1, 2], color: [[1, 0.4, 0.1, 0.8], [0.3, 0.2, 1, 0.2]],
  }, transform: { translation: [i ? 15 : -15, 18, 55] } }))];
  Object.assign(fixture.environment, {
    atmosphere: { preset: "day", fog_density: 0, aerial_density: 0, exposure: 1 },
    terrain: { samples_per_frame: 50_000_000 }, water: { enabled: true }, ground_cover: { enabled: false },
    particles: { enabled: true }, decals: { enabled: true }, reflections: { intensity: 1 },
    clouds: { enabled: true, coverage: 0.6, base: 20, thickness: 50, density: 1, wind: [0, 0],
      scale: 200, seed: 3, shadows: 0, resolution: 256, slices: 1, in_view: true, view_block: 4 },
  });
  return fixture;
}

export function movingFixture() {
  const fixture = glassFixture();
  fixture.id = "rough-glass-moving";
  fixture.entities.push({ id: "motion-parent", animation: { clip: "translate" } });
  for (const entity of fixture.entities) if (entity.id.startsWith("stripe-")) entity.parent = "motion-parent";
  fixture.animations = [{ id: "translate", targets: ["motion-parent"], channels: [
    { target: 0, property: "translation", times: [0, 1, 2], values: [-0.25, 0, 0, 0.25, 0, 0, -0.25, 0, 0] },
  ] }];
  fixture.environment.temporal_antialiasing.enabled = true;
  return fixture;
}
