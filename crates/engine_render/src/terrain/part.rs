use engine_core::glam::Vec3;
use engine_core::{RenderStats, TerrainBounceStats};

use super::{FRAME_HORIZONS, TerrainPass, TerrainViews};
use crate::cached::Bound;
use crate::entries::PassKind;
use crate::environment::ENVIRONMENT_FORMAT;
use crate::graph::{
    Access, Clear, FrameGraph, FrameShape, GraphTable, PassId, ResourceId, TRANSPARENT,
};
use crate::parts::{Begin, BeginOut, Create, DrawAt, Encode, GpuPart, OpaqueSlot, Step, View};
use crate::renderer::{Core, DEPTH_FORMAT};
use crate::scene_shading::SceneShading;
use crate::shadows::ShadowPass;
use crate::terrain_bounce::SceneSources;

const FRAME: &[&str] = &["sky_tables", "refraction"];

const SUBDIVISION: usize = 0;
const BOUNCE: usize = 1;
const IDS: usize = 2;
const RESOLVE: usize = 3;

pub(super) fn create(cx: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(TerrainPass::new(&cx.core.device, &cx.core.queue))
}

pub(super) struct TerrainIds {
    view: wgpu::TextureView,
    group: wgpu::BindGroup,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Slots {
    subdivision: Option<PassId>,
    relight: Option<PassId>,
    identify: Option<PassId>,
    resolve: Option<PassId>,
    leaves: Option<ResourceId>,
    bounce: Option<ResourceId>,
    ids: Option<ResourceId>,
    depth: Option<ResourceId>,
    cloud_dome: Option<ResourceId>,
    occupancy: Option<ResourceId>,
    shadow_maps: Option<ResourceId>,
    lights: Option<ResourceId>,
    probes: Option<ResourceId>,
}

impl Slots {
    fn of(table: &GraphTable) -> Self {
        Self {
            subdivision: table.pass_named("subdivision"),
            relight: table.pass_named("bounce"),
            identify: table.pass_named("terrain_ids"),
            resolve: table.pass_named("terrain_resolve"),
            leaves: table.resource("leaves"),
            bounce: table.resource("bounce"),
            ids: table.resource("terrain_ids"),
            depth: table.resource("depth"),
            cloud_dome: table.resource("cloud_dome"),
            occupancy: table.resource("occupancy"),
            shadow_maps: table.resource("shadow_maps"),
            lights: table.resource("lights"),
            probes: table.resource("probes"),
        }
    }
}

impl GpuPart for TerrainPass {
    fn install(&mut self, table: &GraphTable) {
        self.slots = Slots::of(table);
    }

    fn begin(&mut self, cx: &mut Begin<'_>, _out: &mut BeginOut) {
        self.pipelines.specialize(SceneShading::of(cx.snapshot));
    }

    fn prepare_draws(&mut self, cx: &mut Step<'_>, _encoder: &mut wgpu::CommandEncoder) {
        self.patches.forget_ground_changes();
        let Some(passes) = cx.frame.passes else {
            self.drawn = false;
            self.deferred = false;
            return;
        };
        self.drawn = self.prepare_frame(cx, passes.opaque_kind);
        self.deferred = self.defers_frame(passes.opaque_kind);
    }

    fn defers(&self) -> bool {
        self.deferred
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let slots = self.slots;
        if self.drawn && self.subdivides() && let Some(pass) = slots.subdivision {
            let _ = graph.add(cx.table, pass).write(slots.leaves);
        }
        if self.bounce.relights() && let Some(pass) = slots.relight {
            let _ = graph
                .add(cx.table, pass)
                .read(slots.cloud_dome)
                .write(slots.bounce);
        }
        if self.deferred && let Some(shape) = &cx.frame.shape {
            self.declare_deferred(cx.table, shape, graph);
        }
    }

    fn encode(
        &mut self,
        pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        match pass {
            SUBDIVISION => {
                let core = &*cx.core;
                let camera = &core.camera.buffer;
                self.encode_subdivision(&core.device, encoder, camera, &core.profiler);
                0
            }
            BOUNCE => {
                self.relight(cx, encoder);
                0
            }
            IDS => {
                let mut pass = cx.render_pass(encoder, "engine/terrain-ids", None);
                self.draw_ids(&cx.view(), &mut pass)
            }
            RESOLVE => {
                let mut pass = cx.render_pass(encoder, "engine/terrain-resolve", None);
                self.draw_resolve(&cx.view(), &mut pass)
            }
            _ => 0,
        }
    }

    fn draw(&self, at: DrawAt<'_>, cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        match at {
            DrawAt::Opaque { slot, .. } => {
                if slot == OpaqueSlot::Ground {
                    self.draw_ground(cx, pass)
                } else {
                    0
                }
            }
            DrawAt::ShadowCasters { map, .. } => {
                if self.drawn {
                    self.draw_shadow(pass, map)
                } else {
                    0
                }
            }
            DrawAt::ProbeFace { face, center, lit } => {
                self.draw_face(cx, pass, (face, center, lit));
                0
            }
        }
    }

    fn attachment(&self, resource: &str) -> Option<wgpu::TextureView> {
        if resource != "terrain_ids" {
            return None;
        }
        self.ids.get().map(|ids| ids.view.clone())
    }

    fn contribute<'s>(&'s self, group: &str, out: &mut Vec<(&'static str, Bound<'s>)>) {
        if group == FRAME_HORIZONS.group {
            out.extend(self.horizons.entries());
        }
    }

    fn after_submit(&mut self) {
        self.surface.after_submit();
        self.drawn = false;
        self.deferred = false;
    }

    fn capacity_bytes(&self) -> u64 {
        self.atlas.capacity_bytes()
            + self.surface.capacity_bytes()
            + self.horizons.capacity_bytes()
            + self.sky.capacity_bytes()
            + self.bounce.capacity_bytes()
            + self.macro_maps.capacity_bytes()
            + self.maps.capacity_bytes()
            + self.water.capacity_bytes()
    }

    fn stats(&self, cx: &View<'_>, stats: &mut RenderStats) {
        let surface = self.surface.stats();
        let held = engine_terrain::RenderTerrains::of(cx.frame.snapshot).stats;
        let terrain = &mut stats.terrain;
        terrain.leaves = surface.leaves;
        terrain.triangles = surface.triangles;
        terrain.refused_splits = surface.refused;
        terrain.shallow_subdivisions = surface.shallow;
        terrain.shadow_patches = self.patches.frame.shadow_patches;
        terrain.tiles = self.resident();
        terrain.loading_tiles = held.loading_tiles;
        terrain.gpu_controls = self.atlas.controls;
        terrain.dropped_tiles = held.dropped_tiles + self.atlas.dropped;
        terrain.fallback_tiles = held.fallback_tiles;
        terrain.horizons = self.horizons.count();
        terrain.sky_maps = self.sky.count();
        terrain.lighting_uploads = self.horizons.uploaded + self.sky.uploaded;
        terrain.bounce = TerrainBounceStats {
            relit: self.bounce.relit,
            probes: self.bounce.relit_probes,
            levels: self.bounce.levels,
            dropped: self.bounce.dropped,
        };
        terrain.macro_maps = self.macro_maps.count();
        terrain.dropped_materials = self.tables.dropped_materials;
        terrain.white_water = held.white_water;
    }
}

impl TerrainPass {
    fn prepare_frame(&mut self, cx: &Step<'_>, kind: PassKind) -> bool {
        let snapshot = cx.frame.snapshot;
        let terrains = engine_terrain::RenderTerrains::of(snapshot);
        if terrains.terrains.is_empty() && self.resident() == 0 {
            self.rest();
            return false;
        }
        let jitter = cx.frame.temporal.as_ref().map_or([0.0; 2], |frame| {
            [frame.camera.jitter[0], frame.camera.jitter[1]]
        });
        let culling = terrains.relative_view_projection;
        let terrain_environment = engine_terrain::TerrainEnvironment::of(snapshot);
        let cover = terrain_environment.ground_cover.sanitized();
        let core = &*cx.core;
        let height = core.display.size().height as f32;
        let canopy = (cover.enabled && cover.density > 0.0)
            .then(|| (cover, crate::ground_cover::pixels_per_meter(culling, height)));
        let regions = cx
            .parts
            .get::<ShadowPass>()
            .map_or(&[][..], ShadowPass::region_views);
        let drawn = self.prepare_terrains(
            &core.device,
            &core.queue,
            terrains,
            &TerrainViews {
                camera: crate::gpu_data::jittered(culling, jitter),
                culling,
                eye: snapshot.camera_position,
                detail_eye: snapshot.detail_eye,
                regions,
                settings: terrain_environment.terrain.sanitized(),
                shadows: snapshot.environment.shadows.enabled,
                water: terrain_environment.water.sanitized(),
                fades: cx.frame.kind.gpu_mode,
                now: snapshot.elapsed_seconds,
                cover: canopy,
            },
            (core.display.max_layers, core.materials.mips()),
        );
        let probes = cx.frame.kind.gpu_mode
            && cx.table.pass_named("probe_captures").is_some()
            && snapshot.environment.reflection_probes.enabled
            && !snapshot.reflection_probes.list.is_empty();
        self.warm_frame(cx, kind, probes);
        if !drawn {
            return false;
        }
        let casts = snapshot.environment.shadows.enabled;
        let holes = self.has_holes();
        self.ensure_pipelines(core, (kind, holes, probes), casts);
        if kind.targets().is_some() {
            self.prepare_ids(core);
        }
        true
    }

    fn warm_frame(&mut self, cx: &Step<'_>, kind: PassKind, probes: bool) {
        let snapshot = cx.frame.snapshot;
        let terrains = &engine_terrain::RenderTerrains::of(snapshot).terrains;
        if terrains.is_empty() {
            return;
        }
        let core = &*cx.core;
        self.warm(&core.device);
        let holes = terrains.iter().any(|terrain| terrain.may_hole) || self.has_holes();
        let casts = snapshot.environment.shadows.enabled;
        self.ensure_pipelines(core, (kind, holes, probes), casts);
    }

    fn ensure_pipelines(
        &mut self,
        core: &Core,
        (kind, holes, probes): (PassKind, bool, bool),
        casts: bool,
    ) {
        let (device, lit) = (&core.device, kind.lit);
        let variants: &[bool] = if holes { &[false, true] } else { &[false] };
        for &holes in variants {
            if let Some(targets) = kind.targets() {
                self.pipelines.ensure_pipeline(
                    device,
                    (targets, lit, holes),
                    core.canvas_format(),
                    DEPTH_FORMAT,
                );
            }
            if probes {
                self.pipelines.ensure_probe_pipeline(
                    device,
                    (lit, holes),
                    ENVIRONMENT_FORMAT,
                    DEPTH_FORMAT,
                );
            }
        }
        if casts {
            self.pipelines.ensure_shadow_pipeline(device);
        }
    }

    fn prepare_ids(&mut self, core: &Core) {
        let size = core.display.size();
        if self.ids.resizes(size) {
            let texture = crate::targets::texture_2d(
                &core.device,
                "engine/terrain-ids",
                size.pixels(),
                1,
                1,
                wgpu::TextureFormat::Rg32Uint,
                crate::targets::RENDER_TARGET,
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let group = crate::pipelines::group(
                &core.device,
                "engine/terrain-resolve-group",
                &crate::bindings::TERRAIN_RESOLVE,
                &self.pipelines.resolve_layout,
                &[("terrain_ids_texture", crate::pipelines::view(&view))],
            );
            self.ids.set(size, TerrainIds { view, group });
        }
    }

    fn defers_frame(&self, kind: PassKind) -> bool {
        self.drawn
            && self.ids.get().is_some()
            && kind.targets().is_some_and(|targets| {
                self.pipelines.resolves.contains_key(&(targets, kind.lit))
            })
            && !self.patches.frame.terrains.is_empty()
    }

    fn declare_deferred(&self, table: &GraphTable, shape: &FrameShape, graph: &mut FrameGraph) {
        let slots = self.slots;
        let (Some(identify), Some(resolve_pass), Some(ids), Some(depth)) =
            (slots.identify, slots.resolve, slots.ids, slots.depth)
        else {
            return;
        };
        let _ = graph
            .add(table, identify)
            .read(slots.leaves)
            .color(ids, Access::Clear(Clear::Color(TRANSPARENT)))
            .depth(depth, Access::Load);
        let mut resolve = graph
            .add(table, resolve_pass)
            .read(ids)
            .read(slots.occupancy)
            .read_written(slots.shadow_maps)
            .read_written(slots.lights)
            .read(slots.probes)
            .read(slots.bounce)
            .read(slots.cloud_dome);
        for (target, _) in crate::parts::main_targets(table, shape) {
            resolve = resolve.color(target, Access::Load);
        }
        let _ = resolve.depth(depth, Access::Load);
    }

    fn relight(&mut self, cx: &Encode<'_>, encoder: &mut wgpu::CommandEncoder) {
        let mut frame = Vec::new();
        for (_, part) in cx.parts.iter() {
            part.contribute("frame", &mut frame);
        }
        let named = |name: &str| frame.iter().find(|(known, _)| *known == name).cloned();
        let (Some(shadow), Some(cloud_shadows)) = (named("shadow"), named("cloud_shadows")) else {
            return;
        };
        let core = &*cx.core;
        let scene = SceneSources {
            lighting: core.camera.lighting_buffer(),
            shadow,
            cloud_shadows,
        };
        self.encode_bounce(&core.device, encoder, &core.profiler, scene);
    }

    fn draw_ground(&self, cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        let Some(kind) = cx.frame.passes.map(|passes| passes.opaque_kind) else {
            return 0;
        };
        let Some(targets) = kind.targets().filter(|_| self.drawn) else {
            return 0;
        };
        pass.set_bind_group(0, cx.core.shared_group("frame", FRAME), &[]);
        if kind.lit {
            pass.set_bind_group(2, &cx.core.mesh_pipelines.empty_group, &[]);
            pass.set_bind_group(3, cx.core.shared_group("lights", &[]), &[]);
        }
        let mut draws = 0;
        for (holes, pipeline) in [false, true].into_iter().filter_map(|holes| {
            Some((holes, self.pipelines.pipeline(targets, kind.lit, holes)?))
        }) {
            pass.set_pipeline(pipeline);
            for terrain in &self.patches.frame.terrains {
                if terrain.holes == holes {
                    draws += self.draw_surface(pass, terrain, 0);
                }
            }
        }
        draws
    }

    fn draw_ids(&self, cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        pass.set_bind_group(0, cx.core.shared_group("frame", FRAME), &[]);
        let mut draws = 0;
        for holes in [false, true] {
            let Some(pipeline) = self.pipelines.deferred_ids.get(&holes) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            for terrain in &self.patches.frame.terrains {
                if terrain.holes == holes {
                    draws += self.draw_surface(pass, terrain, 0);
                }
            }
        }
        draws
    }

    fn draw_resolve(&self, cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        let Some(kind) = cx.frame.passes.map(|passes| passes.opaque_kind) else {
            return 0;
        };
        let (Some(targets), Some(ids)) = (kind.targets(), self.ids.get()) else {
            return 0;
        };
        let Some(pipeline) = self.pipelines.resolves.get(&(targets, kind.lit)) else {
            return 0;
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, cx.core.shared_group("frame", FRAME), &[]);
        pass.set_bind_group(2, &ids.group, &[]);
        if kind.lit {
            pass.set_bind_group(3, cx.core.shared_group("lights", &[]), &[]);
        }
        let render = cx.core.render_pixels();
        let mut draws = 0;
        for terrain in &self.patches.frame.terrains {
            let Some([x, y, width, height]) = terrain.scissor(render) else {
                continue;
            };
            if !self.bind(pass, 0, terrain.uniform) {
                continue;
            }
            pass.set_scissor_rect(x, y, width, height);
            pass.draw(0..3, 0..1);
            draws += 1;
        }
        draws
    }

    fn draw_face(
        &self,
        cx: &View<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        (face, center, lit): (usize, Vec3, bool),
    ) {
        if self.patches.frame.terrains.is_empty() {
            return;
        }
        let view = self.write_probe_view(
            &cx.core.queue,
            face,
            crate::probes::face_view_projection(Vec3::ZERO, face),
            self.eye(),
            center,
        );
        pass.set_bind_group(0, cx.core.probe_group(face, FRAME), &[]);
        if lit {
            pass.set_bind_group(2, &cx.core.mesh_pipelines.empty_group, &[]);
        }
        for holes in [false, true] {
            let Some(pipeline) = self.pipelines.probe_pipeline(lit, holes) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            for terrain in &self.patches.frame.terrains {
                if terrain.holes == holes {
                    self.draw_surface(pass, terrain, view);
                }
            }
        }
    }
}
