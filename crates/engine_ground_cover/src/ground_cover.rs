use crate::shaders::entry::cover_grow;
use engine_core::collections::HashMap;
use std::cell::Cell;
use std::rc::Rc;

use engine_core::glam::{Mat4, Vec3};
use engine_core::{RenderSnapshot, RenderStats};
use engine_terrain::GroundCover;

use crate::bindings::{COVER_DRAW, COVER_GROW};
use crate::gpu_layouts::{BLADE_BYTES, COVER_LEAVES, COVER_LUSH_RATIO};
use engine_faces::heightfield::{CoverTerms, HeightfieldFace};
use engine_render::buffers::ReadbackRing;
use engine_render::cached::Cached;
use engine_render::gpu_data::{pixels_per_meter, planes};
use engine_render::graph::{
    FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId, Stage, Targets,
};
use engine_render::parts::{
    Create, DrawAt, Encode, GpuPart, Hook, OpaqueSlot, PartDecl, PartEntry, Step, View,
};
use engine_render::renderer::Core;

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "ground_cover",
        brick: "ground_cover",
        passes: &[PassDecl {
            name: "ground_cover",
            stage: Stage::Prepare,
        }],
        resources: &[ResourceDecl {
            name: "cover",
            lifetime: Lifetime::Kept,
        }],
        hooks: &[Hook::Opaque(OpaqueSlot::Cover)],
        contributes: &[],
        effects: &[],
        labels: &[],
        lazy: false,
        faces: &[],
        needs: &[],
        piece: None,
    },
    create,
};

fn create(cx: &Create<'_>) -> Box<dyn GpuPart> {
    let _owner = engine_render::memory::owned(PART.decl.brick);
    Box::new(GroundCoverPass::new(&cx.core.device))
}

const SLOT: u64 = 512;
pub(crate) const GROW_BYTES: u64 = size_of::<GrowUniform>() as u64;

use crate::gpu_layouts::GrowUniform;

use crate::gpu_layouts::FrameUniform;

const DRAW_ARGS: u64 = size_of::<wgpu::util::DrawIndirectArgs>() as u64;

struct Growth {
    offset: u32,
    cells: u32,
}

const MAX_TIERS: usize = 8;

pub(crate) struct GroundCoverPass {
    grow_layout: engine_render::pipelines::GroupLayout,
    draw_layout: engine_render::pipelines::GroupLayout,
    compute: Option<(wgpu::ComputePipeline, wgpu::ComputePipeline)>,
    pipelines: HashMap<(Targets, usize), wgpu::RenderPipeline>,
    shadings: Vec<Vec<(&'static str, f64)>>,
    shading: usize,
    blades: engine_render::buffers::Growable,
    capacity: u32,
    blades_generation: u64,
    counts: wgpu::Buffer,
    args: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    frame: wgpu::Buffer,
    grow_group: Cached<(u64, u64, u64, u64), wgpu::BindGroup>,
    draw_group: Cached<u64, wgpu::BindGroup>,
    growths: Vec<Growth>,
    grow_uniforms: Vec<GrowUniform>,
    grow_bytes: Vec<u8>,
    previous_time: Option<f32>,
    readbacks: ReadbackRing<u32>,
    totals: Rc<Cell<Option<[u32; 2]>>>,
    reach: f32,
    tiers: u32,
    drawn: bool,
    framed: bool,
    pass: Option<PassId>,
    cover: Option<ResourceId>,
}

impl GroundCoverPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            grow_layout: engine_render::pipelines::layout(device, &COVER_GROW),
            draw_layout: engine_render::pipelines::layout(device, &COVER_DRAW),
            compute: None,
            pipelines: HashMap::default(),
            shadings: Vec::new(),
            shading: 0,
            blades: engine_render::buffers::Growable::new(
                device,
                "engine/cover-blades",
                BLADE_BYTES,
                wgpu::BufferUsages::empty(),
            ),
            capacity: 0,
            blades_generation: 0,
            counts: engine_render::buffers::storage(
                device,
                "engine/cover-counts",
                16,
                wgpu::BufferUsages::COPY_SRC,
            ),
            args: engine_render::buffers::storage(
                device,
                "engine/cover-draw",
                2 * DRAW_ARGS,
                wgpu::BufferUsages::INDIRECT,
            ),
            uniforms: engine_render::buffers::uniform(device, "engine/cover-grow", SLOT),
            frame: engine_render::buffers::uniform(
                device,
                "engine/cover-frame",
                size_of::<FrameUniform>() as u64,
            ),
            grow_group: Cached::default(),
            draw_group: Cached::default(),
            growths: Vec::new(),
            grow_uniforms: Vec::new(),
            grow_bytes: Vec::new(),
            previous_time: None,
            readbacks: ReadbackRing::new("engine/cover-readback"),
            totals: Rc::new(Cell::new(None)),
            reach: 0.0,
            tiers: 0,
            drawn: false,
            framed: false,
            pass: None,
            cover: None,
        }
    }

    fn reach(&self) -> (f32, u32) {
        (self.reach, self.tiers)
    }

    fn totals(&self) -> [u32; 2] {
        self.totals.get().unwrap_or([0; 2])
    }

    fn prepare_growth(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        terrain: &dyn HeightfieldFace,
        terrains: &engine_terrain::RenderTerrains,
        (settings, wind, culling, eye, time): (&GroundCover, [f32; 2], Mat4, Vec3, f32),
        (height, grounded): (f32, bool),
    ) -> bool {
        self.growths.clear();
        self.reach = 0.0;
        self.tiers = 0;
        let previous = self.previous_time.replace(time).unwrap_or(time);
        let grows = self.plan(
            device,
            queue,
            terrain,
            terrains,
            (settings, wind, grounded),
            (culling, eye, time, previous, height),
        );
        if !grows {
            self.totals.set(Some([0; 2]));
        }
        grows
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn plan(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        terrain: &dyn HeightfieldFace,
        terrains: &engine_terrain::RenderTerrains,
        (settings, wind, grounded): (&GroundCover, [f32; 2], bool),
        (culling, eye, time, previous, height): (Mat4, Vec3, f32, f32, f32),
    ) -> bool {
        let settings = settings.sanitized();
        let Some(sources) = terrain.cover_sources() else {
            return false;
        };
        if !settings.enabled || settings.density <= 0.0 {
            return false;
        }
        let planes = planes(culling);
        let pixels_per_meter = pixels_per_meter(culling, height);
        let mut uniforms = std::mem::take(&mut self.grow_uniforms);
        uniforms.clear();
        let mut most = 0.0;
        for (index, render) in terrains.terrains.iter().enumerate() {
            let terms = terrain.cover_terms(index);
            let (directory, palette, colors) = terrain.cover_terrain(index);
            let density = terms.densest * settings.density * lushest();
            let law = CoverLaw::of(terms, pixels_per_meter);
            if density <= 0.0 || law.span <= 0.0 || !render.drawn || render.surroundings {
                continue;
            }
            let spacing = render.pyramid.spacing(0);
            let candidates = engine_core::math::max((density * spacing * spacing).ceil(), 1.0);
            let local = eye - render.origin;
            let (tiers, count) = law.tiers();
            self.reach = engine_core::math::max(self.reach, tiers[count - 1].1);
            self.tiers = self.tiers.max(count as u32);
            for &(inner, outer, scale) in &tiers[..count] {
                let tried = if scale > 1.0 {
                    engine_core::math::min((candidates * law.share(inner)).ceil(), candidates)
                } else {
                    candidates
                };
                let cell = spacing * scale;
                let first = [
                    ((local.x - outer) / cell).floor() as i32,
                    ((local.z - outer) / cell).floor() as i32,
                ];
                let cells = (2.0 * outer / cell).ceil() as u32 + 1;
                let uniform = GrowUniform {
                    origin: render.origin.extend(spacing).to_array(),
                    eye: eye.extend(outer).to_array(),
                    grid: [first[0], first[1], cells as i32, candidates as i32],
                    counts: [directory, render.pyramid.levels(), palette, 0],
                    scale: [
                        settings.density,
                        tried,
                        colors as f32,
                        terrain.cover_tallest() * lushest(),
                    ],
                    tier: [inner, law.span, scale, 0.0],
                    planes,
                    foliage: [f32::from(u8::from(grounded)), 0.0, 0.0, 0.0],
                    view: [pixels_per_meter, 0.0, 0.0, 0.0],
                };
                self.growths.push(Growth {
                    offset: (uniforms.len() as u64 * SLOT) as u32,
                    cells,
                });
                uniforms.push(uniform);
            }
            most += law.area() * density;
        }
        if self.growths.is_empty() {
            self.grow_uniforms = uniforms;
            return false;
        }
        let room = (self.blades.room() / BLADE_BYTES).min(u64::from(u32::MAX)) as u32;
        let seen = (most * turn_share(culling)).ceil() as u32;
        let capacity = settings
            .max_blades
            .map_or(seen, |budget| budget.min(seen))
            .max(1)
            .min(room);
        if self.blades.grow(device, u64::from(capacity) * BLADE_BYTES) {
            self.blades_generation += 1;
        }
        self.capacity = capacity;
        let mut bytes = std::mem::take(&mut self.grow_bytes);
        bytes.clear();
        bytes.resize(uniforms.len() * SLOT as usize, 0);
        for (index, uniform) in uniforms.iter_mut().enumerate() {
            uniform.counts[3] = capacity;
            let at = index * SLOT as usize;
            bytes[at..at + size_of::<GrowUniform>()].copy_from_slice(bytemuck::bytes_of(uniform));
        }
        if bytes.len() as u64 > self.uniforms.size() {
            self.uniforms = engine_render::buffers::uniform(
                device,
                "engine/cover-grow",
                (bytes.len() as u64).next_power_of_two(),
            );
            self.blades_generation += 1;
        }
        queue.write_buffer(&self.uniforms, 0, &bytes);
        self.grow_uniforms = uniforms;
        self.grow_bytes = bytes;
        let frame = FrameUniform {
            wind: [wind[0], wind[1], time, previous],
            ground: [f32::from(u8::from(grounded)), 0.0, 0.0, 0.0],
            blades: [self.capacity, 0, 0, 0],
        };
        queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&frame));
        self.compute.get_or_insert_with(|| compute(device));
        let key = (
            sources.generation.0,
            sources.generation.1,
            self.blades_generation,
            sources.generation.2,
        );
        if self.grow_group.stale(&key) {
            let group = engine_render::pipelines::group(
                device,
                "engine/cover-grow",
                &COVER_GROW,
                &self.grow_layout,
                &[
                    ("heights", engine_render::pipelines::view(sources.heights)),
                    ("control", engine_render::pipelines::view(sources.control)),
                    engine_render::buffers::entry("directory", sources.directory),
                    engine_render::buffers::entry("palette", sources.palette),
                    engine_render::buffers::entry("covers", sources.covers),
                    (
                        "grow",
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.uniforms,
                            offset: 0,
                            size: wgpu::BufferSize::new(GROW_BYTES),
                        }),
                    ),
                    engine_render::buffers::entry("blades", &self.blades),
                    engine_render::buffers::entry("counts", &self.counts),
                    engine_render::buffers::entry("args", &self.args),
                    (
                        "macro_maps",
                        engine_render::pipelines::view(sources.macro_maps),
                    ),
                    ("normals", engine_render::pipelines::view(sources.normals)),
                ],
            );
            self.grow_group.set(key, group);
        }
        if self.draw_group.stale(&self.blades_generation) {
            let group = engine_render::pipelines::group(
                device,
                "engine/cover",
                &COVER_DRAW,
                &self.draw_layout,
                &[
                    engine_render::buffers::entry("cover_blades", &self.blades),
                    engine_render::buffers::entry("cover_frame", &self.frame),
                ],
            );
            self.draw_group.set(self.blades_generation, group);
        }
        true
    }

    fn encode_growth(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &engine_render::profiling::GpuProfiler,
    ) {
        let (Some((grow, finish)), Some(group)) = (&self.compute, self.grow_group.get()) else {
            return;
        };
        if self.growths.is_empty() {
            return;
        }
        encoder.clear_buffer(&self.counts, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/cover-grow"),
                timestamp_writes: profiler.span("ground_cover").map(|span| span.compute()),
            });
            pass.set_pipeline(grow);
            for growth in &self.growths {
                pass.set_bind_group(0, group, &[growth.offset]);
                pass.dispatch_workgroups(growth.cells, growth.cells, 1);
            }
            pass.set_pipeline(finish);
            pass.set_bind_group(0, group, &[self.growths[0].offset]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.readbacks
            .record(device, encoder, &self.counts, 16, self.capacity);
    }

    fn growing(&self) -> bool {
        self.compute.is_some() && self.grow_group.get().is_some() && !self.growths.is_empty()
    }

    fn warm(&mut self, device: &wgpu::Device) {
        self.compute.get_or_insert_with(|| compute(device));
    }

    fn specialize(&mut self, shading: &[(&'static str, f64)]) {
        match self
            .shadings
            .iter()
            .position(|known| known.as_slice() == shading)
        {
            Some(known) => self.shading = known,
            None => {
                self.shading = self.shadings.len();
                self.shadings.push(shading.to_vec());
            }
        }
    }

    fn warm_cover(&mut self, core: &Core, snapshot: &RenderSnapshot, targets: Option<Targets>) {
        let terrains = &engine_terrain::RenderTerrains::of(snapshot).terrains;
        if terrains.is_empty() {
            return;
        }
        let terrain_environment = engine_terrain::TerrainEnvironment::of(snapshot);
        let cover = terrain_environment.ground_cover.sanitized();
        let grows = cover.enabled
            && cover.density > 0.0
            && terrains.iter().any(|terrain| {
                !terrain.surroundings && terrain.surface.iter().any(|look| look.cover.is_some())
            });
        if grows {
            self.warm(&core.device);
            if let Some(targets) = targets {
                self.ensure_pipeline(
                    &core.device,
                    targets,
                    core.display.format(),
                    engine_render::renderer::DEPTH_FORMAT,
                );
            }
        }
    }

    fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        targets: Targets,
        canvas: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
    ) {
        if self.pipelines.contains_key(&(targets, self.shading)) {
            return;
        }
        let layout = engine_render::pipelines::layout_of(
            device,
            &crate::pipeline_layouts::LIT_COVER_PIPELINES,
        );
        let (vertex, fragment) = crate::shaders::cover(targets);
        let color_targets = targets.color_targets(canvas);
        let pipeline = engine_render::pipelines::geometry(
            device,
            &engine_render::pipelines::Geometry {
                label: "engine/cover",
                layout: &layout,
                vertex_constants: &[],
                vertex,
                buffers: &[],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    cull_mode: None,
                    ..Default::default()
                },
                depth: Some(wgpu::DepthStencilState {
                    format: depth,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                fragment: Some(engine_render::pipelines::Fragment {
                    entry: fragment,
                    constants: &self.shadings[self.shading],
                    targets: &color_targets,
                }),
            },
        );
        self.pipelines.insert((targets, self.shading), pipeline);
    }

    fn draw_cover(&self, cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        let Some(passes) = cx.frame.passes else {
            return 0;
        };
        let kind = passes.opaque_kind;
        let Some(targets) = kind.targets().filter(|_| self.drawn) else {
            return 0;
        };
        pass.set_bind_group(0, cx.core.shared_group("frame", &["refraction"]), &[]);
        pass.set_bind_group(2, &cx.core.mesh_pipelines.empty_group, &[]);
        pass.set_bind_group(3, cx.core.shared_group("lights", &[]), &[]);
        self.draw_blades(pass, targets)
    }

    fn draw_blades(&self, pass: &mut wgpu::RenderPass<'_>, targets: Targets) -> u32 {
        let (Some(pipeline), Some(group)) = (
            self.pipelines.get(&(targets, self.shading)),
            self.draw_group.get(),
        ) else {
            return 0;
        };
        if self.growths.is_empty() {
            return 0;
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(1, group, &[]);
        pass.draw_indirect(&self.args, 0);
        pass.draw_indirect(&self.args, DRAW_ARGS);
        2
    }
}

fn turn_share(culling: Mat4) -> f32 {
    let across = culling.row(0).truncate().length();
    let depth = culling.row(3).truncate().length();
    if across <= 0.0 {
        return 1.0;
    }
    engine_core::math::min(
        engine_math::atan(depth / across) / std::f32::consts::PI,
        1.0,
    )
}

fn lushest() -> f32 {
    COVER_LUSH_RATIO.sqrt()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CoverLaw {
    pub(crate) span: f32,
}

impl CoverLaw {
    pub(crate) fn of(terms: CoverTerms, pixels_per_meter: f32) -> Self {
        Self {
            span: terms.span(pixels_per_meter),
        }
    }

    fn reach(self) -> f32 {
        2.0 * COVER_LEAVES as f32 * self.span
    }

    fn tiers(self) -> ([(f32, f32, f32); MAX_TIERS], usize) {
        let reach = self.reach();
        let mut tiers = [(0.0, self.span, 1.0); MAX_TIERS];
        let mut count = 1;
        while count < MAX_TIERS {
            let (_, outer, scale) = tiers[count - 1];
            if outer >= reach {
                break;
            }
            tiers[count] = (
                outer,
                engine_core::math::min(2.0 * outer, reach),
                2.0 * scale,
            );
            count += 1;
        }
        (tiers, count)
    }

    fn widening(self, away: f32) -> f32 {
        let blades = away / engine_core::math::max(self.span, 1.0e-6);
        engine_core::math::clamp(blades, 1.0, COVER_LEAVES as f32)
    }

    fn share(self, away: f32) -> f32 {
        let widening = self.widening(away);
        let spanned = self.span * widening / engine_core::math::max(away, 1.0e-3);
        seen_share(spanned) / widening
    }

    fn area(self) -> f32 {
        const RINGS: u32 = 256;
        let width = self.reach() / RINGS as f32;
        (0..RINGS)
            .map(|ring| {
                let away = (ring as f32 + 0.5) * width;
                std::f32::consts::TAU * away * width * self.share(away)
            })
            .sum()
    }
}

fn seen_share(pixels: f32) -> f32 {
    let octaves =
        engine_core::math::ln(engine_core::math::max(pixels, 1.0e-6)) * std::f32::consts::LOG2_E;
    engine_core::math::clamp(1.0 + octaves, 0.0, 1.0)
}

fn compute(device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::ComputePipeline) {
    let pipeline_layout =
        engine_render::pipelines::layout_of(device, &crate::pipeline_layouts::COVER_GROW_PIPELINES);
    (
        engine_render::pipelines::compute_entry(device, &pipeline_layout, cover_grow::grow_cover),
        engine_render::pipelines::compute_entry(device, &pipeline_layout, cover_grow::finish_cover),
    )
}

impl GpuPart for GroundCoverPass {
    fn install(&mut self, table: &GraphTable) {
        self.pass = table.pass_named("ground_cover");
        self.cover = table.resource("cover");
    }

    fn prepare_draws(&mut self, cx: &mut Step<'_>, _encoder: &mut wgpu::CommandEncoder) {
        self.drawn = false;
        self.framed = false;
        let (Some(terrain), Some(passes)) =
            (cx.parts.face::<dyn HeightfieldFace>(), cx.frame.passes)
        else {
            return;
        };
        let core = &*cx.core;
        let snapshot = cx.frame.snapshot;
        let targets = passes.opaque_kind.targets();
        self.specialize(core.shading().constants());
        self.warm_cover(core, snapshot, targets);
        if !terrain.drawn() {
            return;
        }
        self.drawn = true;
        let terrains = engine_terrain::RenderTerrains::of(snapshot);
        let terrain_environment = engine_terrain::TerrainEnvironment::of(snapshot);
        let covered = self.prepare_growth(
            &core.device,
            &core.queue,
            terrain,
            terrains,
            (
                &terrain_environment.ground_cover,
                terrain_environment.wind.velocity,
                snapshot.view_projection,
                snapshot.detail_eye,
                snapshot.elapsed_seconds,
            ),
            (
                core.display.size().height as f32,
                snapshot.environment.foliage.sanitized().ground_blend > 0.0,
            ),
        );
        if let (true, Some(targets)) = (covered, targets) {
            self.ensure_pipeline(
                &core.device,
                targets,
                core.display.format(),
                engine_render::renderer::DEPTH_FORMAT,
            );
        }
        self.framed = self.growing();
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        if let (true, Some(pass)) = (self.framed, self.pass) {
            let _ = graph.add(cx.table, pass).write(self.cover);
        }
    }

    fn encode(
        &mut self,
        _pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        self.encode_growth(&cx.core.device, encoder, &cx.core.profiler);
        0
    }

    fn draw(&self, at: DrawAt<'_>, cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        if at.hook() == Hook::Opaque(OpaqueSlot::Cover) {
            self.draw_cover(cx, pass)
        } else {
            0
        }
    }

    fn after_submit(&mut self) {
        let totals = Rc::clone(&self.totals);
        self.readbacks.after_submit(move |words, capacity| {
            let grown = words.get(0);
            totals.set(Some([grown.min(capacity), grown.saturating_sub(capacity)]));
        });
    }

    fn capacity_bytes(&self) -> u64 {
        self.blades.size()
    }

    fn stats(&self, _cx: &View<'_>, stats: &mut RenderStats) {
        let (totals, (reach, tiers)) = (self.totals(), self.reach());
        stats.ground_cover = engine_core::GroundCoverStats {
            blade_cut: Default::default(),
            blades: totals[0],
            dropped: totals[1],
            reach: reach.round() as u32,
            tiers,
        };
    }
}
