use engine_core::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use engine_core::glam::Vec3;
use engine_core::math::max;
use engine_core::{Cut, DebugView, RenderSnapshot, RenderStats, WaterStats, Wind};
use engine_faces::heightfield::{
    HeightfieldFace, SubdivisionUniform, SurfaceDraw, TerrainUniform, ViewUniform, WaterDraws,
};
use engine_render::bindings::{Contribution, Sampling, Stages, sampler};
use engine_render::cached::Bound;
use engine_render::entries::PassKind;
use engine_render::graph::{
    Access, FrameGraph, GraphTable, PassDecl, PassId, ResourceId, Stage, Targets,
};
use engine_render::parts::{
    Begin, BeginOut, Create, Encode, GpuPart, MadeRecord, PartDecl, PartEntry, RecordStart,
    Shading, Step, View,
};
use engine_render::post::PostPass;
use engine_render::profiling::Span;
use engine_render::renderer::Core;
use engine_render::targets::{FrameSize, ScreenTarget};
use engine_terrain::{Fall, RenderTerrain, SettleStart, Water};

use crate::bindings::{WATER, WATER_BEHIND};
use crate::water_falls::{fall_layout, sheets};
use crate::water_particles::{ParticleStats, ParticleView, WaterParticles, particle_terrain};
use crate::water_settle::WaterSettle;
use crate::water_sim::{Live, LiveGround, WaterSim};

use crate::gpu_layouts::water::{FallVertex, WaterUniform};

pub(crate) const WATER_UNIFORM_BYTES: usize = size_of::<WaterUniform>();

pub static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "water",
        brick: "water",
        passes: &[PassDecl {
            name: "water",
            stage: Stage::Surfaces,
        }],
        resources: &[],
        hooks: &[],
        contributes: &[Contribution {
            group: "frame",
            set: "refraction",
            bindings: &[sampler(
                "refraction_sampler",
                Sampling::Filtering,
                Stages::FRAGMENT,
            )],
        }],
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
    Box::new(WaterPass::new(&cx.core.device))
}

#[derive(Clone, Copy, Default)]
struct Ids {
    pass: Option<PassId>,
    shadow_maps: Option<ResourceId>,
    leaves: Option<ResourceId>,
    probes: Option<ResourceId>,
    depth: Option<ResourceId>,
}

struct Placeholders {
    scene: wgpu::TextureView,
    splatted: wgpu::TextureView,
    live: wgpu::Buffer,
    storage: wgpu::Buffer,
    particles: wgpu::Buffer,
}

struct TileDraws {
    terrains: Vec<(u64, u32, u32, SurfaceDraw)>,
}

type BuiltFalls = (u64, Vec3, Arc<[Fall]>, Range<u32>);

struct FallDraws {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    built: Vec<BuiltFalls>,
}

struct BehindCopies {
    depth: wgpu::Texture,
    scene: Option<wgpu::Texture>,
    foam: wgpu::TextureView,
    group: wgpu::BindGroup,
    drawing: wgpu::BindGroup,
    particles: u64,
}

const FOAM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

struct WaterFrame {
    blocks: [u32; 2],
    subdivided: [u32; 2],
}

pub(crate) struct WaterPass {
    layout: engine_render::pipelines::GroupLayout,
    behind_layout: engine_render::pipelines::GroupLayout,
    behind: ScreenTarget<BehindCopies>,
    placeholders: Placeholders,
    refraction_sampler: wgpu::Sampler,
    settle: WaterSettle,
    sim: WaterSim,
    particles: WaterParticles,
    uniform: wgpu::Buffer,
    group: Option<(GroupGenerations, wgpu::BindGroup)>,
    tiles: Option<TileDraws>,
    pipelines: HashMap<Targets, wgpu::RenderPipeline>,
    foam_pipeline: Option<wgpu::RenderPipeline>,
    falls: Option<FallDraws>,
    fall_pipelines: HashMap<Targets, wgpu::RenderPipeline>,
    fall_vertices: Vec<FallVertex>,
    fall_indices: Vec<u32>,
    shading: Vec<(&'static str, f64)>,
    water_views: bool,
    frame: Option<WaterFrame>,
    ids: Ids,
}

impl WaterPass {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: engine_render::pipelines::layout(device, &WATER),
            behind_layout: engine_render::pipelines::layout(device, &WATER_BEHIND),
            behind: ScreenTarget::default(),
            placeholders: placeholders(device),
            refraction_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/refraction-sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            settle: WaterSettle::new(device),
            sim: WaterSim::new(device),
            particles: WaterParticles::new(device),
            uniform: engine_render::buffers::uniform(
                device,
                "engine/water",
                WATER_UNIFORM_BYTES as u64,
            ),
            group: None,
            tiles: None,
            pipelines: HashMap::default(),
            foam_pipeline: None,
            falls: None,
            fall_pipelines: HashMap::default(),
            fall_vertices: Vec::new(),
            fall_indices: Vec::new(),
            shading: Vec::new(),
            water_views: false,
            frame: None,
            ids: Ids::default(),
        }
    }

    fn specialize(&mut self, shading: Vec<(&'static str, f64)>, water_views: bool) {
        if self.shading != shading {
            self.shading = shading;
            self.pipelines.clear();
            self.fall_pipelines.clear();
        }
        if self.water_views != water_views {
            self.water_views = water_views;
            self.pipelines.remove(&Targets::Debug);
        }
    }

    fn prepare_falls(&mut self, device: &wgpu::Device, terrains: &[RenderTerrain]) {
        let falling = || terrains.iter().filter(|terrain| !terrain.falls.is_empty());
        let same = self.falls.as_ref().is_some_and(|built| {
            built.built.len() == falling().count()
                && built
                    .built
                    .iter()
                    .zip(falling())
                    .all(|(a, b)| a.0 == b.id && a.1 == b.origin && Arc::ptr_eq(&a.2, &b.falls))
        });
        if same || (self.falls.is_none() && falling().next().is_none()) {
            return;
        }
        let runs = sheets(
            falling().map(|terrain| (terrain.origin, &*terrain.falls)),
            &mut self.fall_vertices,
            &mut self.fall_indices,
        );
        if self.fall_indices.is_empty() {
            self.falls = None;
            return;
        }
        let buffer = |label, contents: &[u8], usage| {
            engine_render::memory::buffer_init(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                },
            )
        };
        self.falls = Some(FallDraws {
            vertices: buffer(
                "engine/water-falls",
                bytemuck::cast_slice(&self.fall_vertices),
                wgpu::BufferUsages::VERTEX,
            ),
            indices: buffer(
                "engine/water-fall-indices",
                bytemuck::cast_slice(&self.fall_indices),
                wgpu::BufferUsages::INDEX,
            ),
            built: falling()
                .zip(runs)
                .map(|(terrain, run)| (terrain.id, terrain.origin, terrain.falls.clone(), run))
                .collect(),
        });
    }

    fn copy_behind(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::Texture,
        scene: Option<&wgpu::Texture>,
    ) {
        let Some(copies) = self.behind.get() else {
            return;
        };
        let pairs = [
            Some((depth, &copies.depth)),
            scene.zip(copies.scene.as_ref()),
        ];
        for (from, to) in pairs.into_iter().flatten() {
            if from.size() == to.size() {
                encoder.copy_texture_to_texture(
                    from.as_image_copy(),
                    to.as_image_copy(),
                    from.size(),
                );
            }
        }
    }

    fn prepare_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        (tiles, terrains, heightfield): (
            Option<WaterDraws<'_>>,
            &[RenderTerrain],
            &dyn HeightfieldFace,
        ),
        settings: Water,
        (seen, wind, seconds): (ParticleView, Wind, f64),
        (depth, refracts): (&wgpu::Texture, bool),
    ) -> Option<WaterFrame> {
        self.tiles = None;
        let blocks = tiles.as_ref().map_or([0; 2], |tiles| tiles.blocks);
        let subdivided = tiles.as_ref().map_or([0; 2], |tiles| tiles.subdivided);
        let tiles = tiles.filter(|tiles| settings.enabled && !tiles.terrains.is_empty());
        self.prepare_falls(device, if settings.enabled { terrains } else { &[] });
        let tiles = tiles?;
        let live = terrains.iter().find_map(|terrain| {
            let window = terrain.water_window.as_ref()?;
            let ground = tiles
                .terrains
                .iter()
                .find(|drawn| drawn.0 == terrain.id)
                .and_then(|drawn| {
                    let ground = heightfield.ground(terrain.id)?;
                    Some(LiveGround {
                        group: ground.group,
                        uniform: drawn.1,
                        terrain: terrain.id,
                        held: heightfield,
                    })
                });
            Some(Live {
                window,
                touches: &terrain.water_touches[..],
                beyond: terrain.water_touches_beyond,
                ground,
            })
        });
        self.sim.prepare(device, queue, live, seconds);
        let (running, sim_generation) = self.sim.drawn();
        let running = running.map(|drawn| {
            let window = drawn.window;
            let square = [
                window.origin[0],
                window.origin[1],
                window.cell,
                window.cells as f32,
            ];
            let [wrap_x, wrap_z] = drawn.wrap;
            (
                square,
                [
                    window.datum,
                    wrap_x as f32,
                    wrap_z as f32,
                    drawn.pool as f32,
                ],
                [window.pixels, 0.0, 0.0, 0.0],
            )
        });
        let carried = particle_terrain(&tiles, terrains, seen.detail);
        let ridden = running.map_or(settings.simulation_cell, |(square, _, _)| square[2]);
        self.particles.prepare(
            (device, queue),
            carried,
            seen,
            (&settings, ridden, wind.velocity, seconds),
        );
        self.prepare_tiles(device, &tiles, sim_generation);
        let [red, green, blue] = settings.color;
        let sediment = settings.sediment_color;
        let uniform = WaterUniform {
            color: [red, green, blue, settings.clarity],
            absorption: absorption(settings.color, settings.clarity),
            foam: self.particles.shading(),
            sim: running.map_or([0.0; 4], |(square, _, _)| square),
            sim_datum: running.map_or([0.0; 4], |(_, datum, _)| datum),
            sim_pixels: running.map_or([0.0; 4], |(_, _, pixels)| pixels),
            sediment: [sediment[0], sediment[1], sediment[2], settings.sediment],
            lift: [tiles.targets[0], tiles.targets[1], 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        let size = FrameSize {
            width: depth.width(),
            height: depth.height(),
        };
        let particles = self.particles.generation();
        let stale = self.behind.get().is_some_and(|copies| {
            copies.scene.is_some() != refracts || copies.particles != particles
        });
        if stale {
            self.behind.release();
        }
        if self.behind.resizes(size) {
            let copies = self.behind_copies(device, size, depth.format(), refracts);
            self.behind.set(size, copies);
        }
        Some(WaterFrame { blocks, subdivided })
    }

    fn prepare_tiles(
        &mut self,
        device: &wgpu::Device,
        tiles: &WaterDraws<'_>,
        sim_generation: u64,
    ) {
        let generations = (tiles.generation, sim_generation);
        if self
            .group
            .as_ref()
            .is_none_or(|(bound, _)| *bound != generations)
        {
            let group = self.tile_group(device, tiles);
            self.group = Some((generations, group));
        }
        self.tiles = Some(TileDraws {
            terrains: tiles
                .terrains
                .iter()
                .map(|(id, uniform, tree, leaves)| (*id, *uniform, *tree, leaves.clone()))
                .collect(),
        });
    }

    fn behind_copies(
        &self,
        device: &wgpu::Device,
        size: FrameSize,
        format: wgpu::TextureFormat,
        refracts: bool,
    ) -> BehindCopies {
        let copy = |label, format| {
            engine_render::targets::texture_2d(
                device,
                label,
                size.pixels(),
                1,
                1,
                format,
                wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            )
        };
        let depth = copy("engine/water-depth", format);
        let scene = refracts.then(|| copy("engine/water-scene", engine_render::post::SCENE_FORMAT));
        let foam = engine_render::targets::texture_2d(
            device,
            "engine/water-foam",
            size.pixels(),
            1,
            1,
            FOAM_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        )
        .create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let scene_view = scene
            .as_ref()
            .map(|scene| scene.create_view(&wgpu::TextureViewDescriptor::default()));
        let (frame, pool) = self
            .particles
            .foam_buffers()
            .unwrap_or((&self.placeholders.particles, &self.placeholders.storage));
        let group = |label, drawn: &wgpu::TextureView| {
            engine_render::pipelines::group(
                device,
                label,
                &WATER_BEHIND,
                &self.behind_layout,
                &[
                    ("water_depth", engine_render::pipelines::view(&depth_view)),
                    (
                        "water_scene",
                        engine_render::pipelines::view(
                            scene_view.as_ref().unwrap_or(&self.placeholders.scene),
                        ),
                    ),
                    (
                        "water_particles",
                        engine_render::pipelines::view(
                            self.particles.view().unwrap_or(&self.placeholders.splatted),
                        ),
                    ),
                    (
                        "water_particles_sampler",
                        engine_render::pipelines::sampler(self.particles.sampler()),
                    ),
                    ("particle_frame", frame.as_entire_binding()),
                    engine_render::buffers::entry("foam_pool", pool),
                    ("water_foam_drawn", engine_render::pipelines::view(drawn)),
                ],
            )
        };
        BehindCopies {
            group: group("engine/water-behind", &foam),
            drawing: group("engine/water-foam-draw", &self.placeholders.scene),
            depth,
            scene,
            foam,
            particles: self.particles.generation(),
        }
    }

    fn draw_foam(&self, encoder: &mut wgpu::CommandEncoder, camera: &wgpu::BindGroup) {
        let Some(behind) = self.behind.get() else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/water-foam"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &behind.foam,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if let (Some(pipeline), Some((_, group))) = (&self.foam_pipeline, &self.group) {
            self.particles
                .draw_foam(&mut pass, pipeline, (camera, group, &behind.drawing));
        }
    }

    fn tile_group(&self, device: &wgpu::Device, tiles: &WaterDraws<'_>) -> wgpu::BindGroup {
        let looks = self
            .sim
            .drawn()
            .0
            .map_or(&self.placeholders.live, |drawn| drawn.looks);
        engine_render::pipelines::group(
            device,
            "engine/water-group",
            &WATER,
            &self.layout,
            &[
                (
                    "terrain_heights",
                    engine_render::pipelines::view(tiles.heights),
                ),
                ("terrain_view", sized(tiles.views, size_of::<ViewUniform>())),
                (
                    "terrain",
                    sized(tiles.uniforms, size_of::<TerrainUniform>()),
                ),
                engine_render::buffers::entry("terrain_directory", tiles.directory),
                engine_render::buffers::entry("terrain_surface_leaves", tiles.leaves),
                (
                    "water_surface",
                    engine_render::pipelines::view(tiles.surfaces),
                ),
                ("water_info", engine_render::pipelines::view(tiles.info)),
                ("water", self.uniform.as_entire_binding()),
                engine_render::buffers::entry("water_live", looks),
                (
                    "water_tree",
                    sized(tiles.trees, size_of::<SubdivisionUniform>()),
                ),
            ],
        )
    }

    fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        (targets, coming): (Targets, bool),
        formats: (wgpu::TextureFormat, wgpu::TextureFormat),
    ) {
        if (coming || self.tiles.is_some()) && !self.pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, targets, formats, Surface::Tiles);
            self.pipelines.insert(targets, pipeline);
        }
        if (coming || self.tiles.is_some()) && self.foam_pipeline.is_none() {
            self.foam_pipeline = Some(Self::foam_pipeline(device));
        }
        if self.falls.is_some() && !self.fall_pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, targets, formats, Surface::Falls);
            self.fall_pipelines.insert(targets, pipeline);
        }
        self.particles.ensure_pipelines(device);
        self.sim.ensure_setup(device);
    }

    fn foam_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
        let layout =
            engine_render::pipelines::layout_of(device, &crate::pipeline_layouts::WATER_PIPELINES);
        let added = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let targets = [Some(wgpu::ColorTargetState {
            format: FOAM_FORMAT,
            blend: Some(wgpu::BlendState {
                color: added,
                alpha: added,
            }),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        engine_render::pipelines::geometry(
            device,
            &engine_render::pipelines::Geometry {
                label: "engine/water-foam",
                layout: &layout,
                vertex_constants: &[],
                vertex: crate::shaders::entry::meshes::vs_foam,
                buffers: &[],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth: None,
                fragment: Some(engine_render::pipelines::Fragment {
                    entry: crate::shaders::entry::meshes::fs_foam,
                    constants: &[],
                    targets: &targets,
                }),
            },
        )
    }

    fn pipeline(
        &self,
        device: &wgpu::Device,
        targets: Targets,
        (canvas, depth): (wgpu::TextureFormat, wgpu::TextureFormat),
        surface: Surface,
    ) -> wgpu::RenderPipeline {
        let layout =
            engine_render::pipelines::layout_of(device, &crate::pipeline_layouts::WATER_PIPELINES);
        let (vertex, fragment) = match surface {
            Surface::Tiles => crate::entries::water(targets),
            Surface::Falls => crate::entries::falls(targets),
        };
        let blended = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let mut color_targets = targets.color_targets(canvas);
        if let Some(Some(scene)) = color_targets.first_mut()
            && targets != Targets::Debug
        {
            scene.blend = Some(blended);
        }
        let sheets = [Some(fall_layout())];
        let constants = match (targets, surface) {
            (Targets::Debug, Surface::Tiles) => crate::debug::water_constants(self.water_views)
                .into_iter()
                .chain(self.shading.iter().copied())
                .collect(),
            _ => self.shading.clone(),
        };
        engine_render::pipelines::geometry(
            device,
            &engine_render::pipelines::Geometry {
                label: match surface {
                    Surface::Tiles => "engine/water",
                    Surface::Falls => "engine/water-falls",
                },
                layout: &layout,
                vertex_constants: &[],
                vertex,
                buffers: match surface {
                    Surface::Tiles => &[],
                    Surface::Falls => &sheets,
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
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
                    constants: &constants,
                    targets: &color_targets,
                }),
            },
        )
    }

    fn prepare_water(
        &mut self,
        core: &Core,
        snapshot: &RenderSnapshot,
        terrain: &dyn HeightfieldFace,
        opaque_kind: PassKind,
    ) -> Option<WaterFrame> {
        let targets = opaque_kind.targets()?;
        let terrains = engine_terrain::RenderTerrains::of(snapshot);
        let coming = terrains.terrains.iter().any(|terrain| terrain.hydrology);
        let device = &core.device;
        let formats = (core.display.format(), engine_render::renderer::DEPTH_FORMAT);
        let tiles = terrain.water_draws(snapshot);
        let display = core.display.region.display;
        let seen = ParticleView {
            detail: snapshot.detail_eye,
            pixel_angle: engine_render::gpu_data::pixel_angle(
                snapshot.view_projection,
                display.height,
            ),
            display,
        };
        if coming {
            self.ensure_pipeline(device, (targets, true), formats);
        }
        let prepared = self.prepare_frame(
            device,
            &core.queue,
            (tiles, &terrains.terrains, terrain),
            snapshot.section::<Water>(),
            (seen, snapshot.section::<Wind>(), snapshot.presented_seconds),
            (
                &core.depth.texture,
                matches!(targets, Targets::Post | Targets::Temporal),
            ),
        )?;
        self.ensure_pipeline(device, (targets, false), formats);
        Some(prepared)
    }

    fn draw_water(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        camera: &wgpu::BindGroup,
        targets: Targets,
    ) -> u32 {
        let Some(behind) = self.behind.get() else {
            return 0;
        };
        let mut draws = 0;
        let Some((_, group)) = &self.group else {
            return draws;
        };
        if let (Some(tiles), Some(pipeline)) = (&self.tiles, self.pipelines.get(&targets)) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(2, &behind.group, &[]);
            for (_, uniform, tree, leaves) in &tiles.terrains {
                pass.set_bind_group(1, group, &[0, *uniform, *tree]);
                leaves.record(pass);
                draws += 1;
            }
        }
        if let (Some(falls), Some(tiles), Some(pipeline)) =
            (&self.falls, &self.tiles, self.fall_pipelines.get(&targets))
        {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(2, &behind.group, &[]);
            pass.set_vertex_buffer(0, falls.vertices.slice(..));
            pass.set_index_buffer(falls.indices.slice(..), wgpu::IndexFormat::Uint32);
            for (id, _, _, run) in &falls.built {
                let Some(&(_, uniform, _, _)) =
                    tiles.terrains.iter().find(|terrain| terrain.0 == *id)
                else {
                    continue;
                };
                if run.is_empty() {
                    continue;
                }
                pass.set_bind_group(1, group, &[0, uniform, 0]);
                pass.draw_indexed(run.clone(), 0, 0..1);
                draws += 1;
            }
        }
        draws
    }
}

impl GpuPart for WaterPass {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            pass: table.pass_named("water"),
            shadow_maps: table.resource("shadow_maps"),
            leaves: table.resource("leaves"),
            probes: table.resource("probes"),
            depth: table.resource("depth"),
        };
    }

    fn begin(&mut self, cx: &mut Begin<'_>, _out: &mut BeginOut) {
        let water_views = crate::debug::shows_water(cx.snapshot.section::<DebugView>().view);
        let mut shading = Shading::default();
        for (_, part) in cx.parts.iter() {
            part.shading(cx.snapshot, &mut shading);
        }
        self.specialize(shading.values, water_views);
    }

    fn prepare_draws(&mut self, cx: &mut Step<'_>, encoder: &mut wgpu::CommandEncoder) {
        self.frame = None;
        let Some(terrain) = cx.parts.face::<dyn HeightfieldFace>() else {
            return;
        };
        let core = &*cx.core;
        self.settle.prepare(&core.device, &core.queue, terrain);
        self.settle.encode(encoder);
        if let Some(passes) = cx.frame.passes {
            self.frame = self.prepare_water(core, cx.frame.snapshot, terrain, passes.opaque_kind);
        }
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (Some(pass), Some(shape)) = (self.ids.pass, cx.frame.shape) else {
            return;
        };
        if self.frame.is_none() {
            return;
        }
        let mut water = graph
            .add(cx.table, pass)
            .read_written(self.ids.shadow_maps)
            .read_written(self.ids.leaves)
            .read(self.ids.probes);
        for (target, _) in engine_render::parts::main_targets(cx.table, &shape) {
            water = water.color(target, Access::Load);
        }
        if let Some(depth) = self.ids.depth {
            let _ = water.depth(depth, Access::Load);
        }
    }

    fn encode(
        &mut self,
        _pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let Some(shape) = cx.frame.shape else {
            return 0;
        };
        let core = &*cx.core;
        self.sim.encode(encoder, &core.profiler);
        let camera = core.shared_group("frame", &[]);
        if let (Some(_), Some((_, group))) = (&self.tiles, &self.group) {
            self.particles
                .encode(&core.device, encoder, (camera, group), &core.profiler);
        }
        let scene = matches!(shape.targets, Targets::Post | Targets::Temporal)
            .then(|| cx.parts.post())
            .flatten()
            .and_then(PostPass::scene_texture);
        self.copy_behind(encoder, &core.depth.texture, scene);
        self.draw_foam(encoder, camera);
        let mut pass = cx.render_pass(
            encoder,
            "engine/water",
            core.profiler.span("water").map(Span::render),
        );
        self.draw_water(&mut pass, camera, shape.targets)
    }

    fn contribute<'s>(&'s self, group: &str, out: &mut Vec<(&'static str, Bound<'s>)>) {
        if group == "frame" {
            out.push((
                "refraction_sampler",
                Bound::fixed(engine_render::pipelines::sampler(&self.refraction_sampler)),
            ));
        }
    }

    fn after_submit(&mut self) {
        self.settle.after_submit();
        self.particles.after_submit();
    }

    fn set_cooking(&mut self, cooking: bool) {
        self.settle.set_cooking(cooking);
    }

    fn start_records(&mut self, starts: &mut Vec<RecordStart>) {
        let mut settles = Vec::new();
        for (terrain, records, start) in std::mem::take(starts) {
            match start.downcast::<SettleStart>() {
                Ok(start) => settles.push((terrain, records, start)),
                Err(start) => starts.push((terrain, records, start)),
            }
        }
        self.settle.settle(settles);
    }

    fn take_records(&mut self) -> Vec<MadeRecord> {
        self.settle.take_made()
    }

    fn records_left(&self) -> Option<(&'static str, u64)> {
        Some(("water looks", self.settle.looks_left()))
    }

    fn capacity_bytes(&self) -> u64 {
        let copies = self.behind.size().map_or(0, |size| {
            let texels = u64::from(size.width) * u64::from(size.height);
            let scene = self
                .behind
                .get()
                .is_some_and(|copies| copies.scene.is_some());
            texels * (if scene { 8 } else { 4 } + 4)
        });
        let falls = self
            .falls
            .as_ref()
            .map_or(0, |falls| falls.vertices.size() + falls.indices.size());
        copies
            + falls
            + self.sim.capacity_bytes()
            + self.particles.capacity_bytes()
            + self.settle.capacity_bytes()
    }

    fn stats(&self, _cx: &View<'_>, stats: &mut RenderStats) {
        let frame = self.frame.as_ref();
        let (cell_cut, [live_cut, live_steps, live_behind, live_setup, live_waiting]) =
            frame.map_or((Cut::default(), [0; 5]), |_| self.sim.cells());
        let [touches, dropped_touches] = frame.map_or([0; 2], |_| self.sim.touches());
        let [leaves, refused] = frame.map_or([0; 2], |frame| frame.subdivided);
        let particles = frame.map_or_else(ParticleStats::default, |_| self.particles.stats());
        stats.water = WaterStats {
            cell_cut,
            foam_cut: particles.cut,
            leaf_cut: Cut::of(leaves.saturating_add(refused), leaves),
            blocks: frame.map_or(0, |frame| frame.blocks[0]),
            dropped_blocks: frame.map_or(0, |frame| frame.blocks[1]),
            live_cut,
            live_steps,
            live_behind,
            live_setup,
            live_waiting,
            touches,
            dropped_touches,
            particle_texels: particles.texels,
            foam_points: particles.points,
            foam_reach: particles.reach,
            foam_particles: particles.foam,
            wave_particles: particles.waves,
            particles_born: particles.born,
            dropped_particles: particles.dropped,
        };
    }
}

type GroupGenerations = ([u64; 6], u64);

#[derive(Clone, Copy)]
enum Surface {
    Tiles,
    Falls,
}

fn sized(buffer: &wgpu::Buffer, bytes: usize) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: wgpu::BufferSize::new(bytes as u64),
    })
}

fn absorption(color: [f32; 3], clarity: f32) -> [f32; 4] {
    const LUMINANCE: [f32; 3] = [0.2126, 0.7152, 0.0722];
    let luminance = max(
        color[0] * LUMINANCE[0] + color[1] * LUMINANCE[1] + color[2] * LUMINANCE[2],
        1.0e-4,
    );
    let weights = color.map(|channel| luminance / max(channel, luminance * 0.125));
    let mean = weights[0] * LUMINANCE[0] + weights[1] * LUMINANCE[1] + weights[2] * LUMINANCE[2];
    let rate = std::f32::consts::LN_2 / max(clarity, 0.01) / mean;
    [weights[0] * rate, weights[1] * rate, weights[2] * rate, 0.0]
}

fn placeholders(device: &wgpu::Device) -> Placeholders {
    let array = |label, format| {
        engine_render::memory::texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 2,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )
    };
    let scene = array(
        "engine/water-placeholder-scene",
        wgpu::TextureFormat::Rgba8Unorm,
    );
    Placeholders {
        scene: scene.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            array_layer_count: Some(1),
            ..Default::default()
        }),
        splatted: scene.create_view(&wgpu::TextureViewDescriptor::default()),
        live: engine_render::buffers::storage(
            device,
            "engine/water-placeholder-live",
            16,
            wgpu::BufferUsages::empty(),
        ),
        storage: engine_render::buffers::storage(
            device,
            "engine/water-placeholder-storage",
            256,
            wgpu::BufferUsages::empty(),
        ),
        particles: engine_render::buffers::uniform(
            device,
            "engine/water-placeholder-particles",
            size_of::<crate::gpu_layouts::water::ParticleFrame>() as u64,
        ),
    }
}
