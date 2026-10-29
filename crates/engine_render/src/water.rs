use engine_core::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use engine_core::glam::Vec3;
use engine_core::math::max;
use engine_core::{RenderSnapshot, WaterStats};
use engine_terrain::{Fall, RenderTerrain, Water, Wind};

use crate::bindings::{WATER, WATER_BEHIND};
use crate::gpu_layouts::terrain::{SubdivisionUniform, TerrainUniform, ViewUniform};
use crate::profiling::Span;
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::{Access, FrameShape, Node, Pass, RenderGraph, Resource, Targets};
use crate::scene_shading::SceneShading;
use crate::targets::{FrameSize, ScreenTarget};
use crate::terrain::{SurfaceDraw, TerrainWaterDraws};
use crate::water_falls::{fall_layout, sheets};
use crate::water_particles::{ParticleStats, ParticleView, WaterParticles, particle_terrain};
use crate::water_sim::{Live, LiveGround, WaterSim};

use crate::gpu_layouts::water::{FallVertex, WaterUniform};

pub(crate) const WATER_UNIFORM_BYTES: usize = size_of::<WaterUniform>();

struct Placeholders {
    scene: wgpu::TextureView,
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

pub(crate) struct WaterFrame {
    blocks: [u32; 2],
    subdivided: [u32; 2],
}

pub(crate) struct WaterPass {
    pub(crate) layout: crate::pipelines::GroupLayout,
    behind_layout: crate::pipelines::GroupLayout,
    behind: ScreenTarget<BehindCopies>,
    placeholders: Placeholders,
    sim: WaterSim,
    particles: WaterParticles,
    uniform: wgpu::Buffer,
    group: Option<(GroupGenerations, wgpu::BindGroup)>,
    tiles: Option<TileDraws>,
    pipeline_layout: Option<wgpu::PipelineLayout>,
    pipelines: HashMap<Targets, wgpu::RenderPipeline>,
    foam_pipeline: Option<wgpu::RenderPipeline>,
    falls: Option<FallDraws>,
    fall_pipelines: HashMap<Targets, wgpu::RenderPipeline>,
    fall_vertices: Vec<FallVertex>,
    fall_indices: Vec<u32>,
    shading: SceneShading,
    water_views: bool,
}

impl WaterPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let _owner = crate::memory::feature(crate::render_graph::FeaturePass::Water);
        Self {
            layout: crate::pipelines::layout(device, &WATER),
            behind_layout: crate::pipelines::layout(device, &WATER_BEHIND),
            behind: ScreenTarget::default(),
            placeholders: placeholders(device),
            sim: WaterSim::new(device),
            particles: WaterParticles::new(device),
            uniform: crate::buffers::uniform(device, "engine/water", WATER_UNIFORM_BYTES as u64),
            group: None,
            tiles: None,
            pipeline_layout: None,
            pipelines: HashMap::default(),
            foam_pipeline: None,
            falls: None,
            fall_pipelines: HashMap::default(),
            fall_vertices: Vec::new(),
            fall_indices: Vec::new(),
            shading: SceneShading::ALL,
            water_views: false,
        }
    }

    pub(crate) fn specialize(&mut self, shading: SceneShading, water_views: bool) {
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
            crate::memory::buffer_init(device, &wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
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

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        (tiles, terrains): (Option<TerrainWaterDraws<'_>>, &[RenderTerrain]),
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
            let ground = tiles.ground.and_then(|(layout, group)| {
                let drawn = tiles.terrains.iter().find(|drawn| drawn.0 == terrain.id)?;
                Some(LiveGround {
                    layout,
                    group,
                    uniform: drawn.1,
                    terrain: terrain.id,
                    held: tiles.held,
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
        tiles: &TerrainWaterDraws<'_>,
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
            terrains: tiles.terrains.clone(),
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
            crate::targets::texture_2d(
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
        let scene = refracts.then(|| copy("engine/water-scene", crate::post::SCENE_FORMAT));
        let foam = crate::targets::texture_2d(
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
            crate::pipelines::group(
                device,
                label,
                &WATER_BEHIND,
                &self.behind_layout,
                &[
                    ("water_depth", crate::pipelines::view(&depth_view)),
                    (
                        "water_scene",
                        crate::pipelines::view(
                            scene_view.as_ref().unwrap_or(&self.placeholders.scene),
                        ),
                    ),
                    (
                        "water_particles",
                        crate::pipelines::view(
                            self.particles.view().unwrap_or(&self.placeholders.scene),
                        ),
                    ),
                    (
                        "water_particles_sampler",
                        crate::pipelines::sampler(self.particles.sampler()),
                    ),
                    (
                        "water_particles_spread",
                        crate::pipelines::view(
                            self.particles
                                .spread_view()
                                .unwrap_or(&self.placeholders.scene),
                        ),
                    ),
                    ("particle_frame", frame.as_entire_binding()),
                    crate::buffers::entry("foam_pool", pool),
                    ("water_foam_drawn", crate::pipelines::view(drawn)),
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

    fn tile_group(&self, device: &wgpu::Device, tiles: &TerrainWaterDraws<'_>) -> wgpu::BindGroup {
        let looks = self
            .sim
            .drawn()
            .0
            .map_or(&self.placeholders.live, |drawn| drawn.looks);
        crate::pipelines::group(
            device,
            "engine/water-group",
            &WATER,
            &self.layout,
            &[
                ("terrain_heights", crate::pipelines::view(tiles.heights)),
                ("terrain_view", sized(tiles.views, size_of::<ViewUniform>())),
                (
                    "terrain",
                    sized(tiles.uniforms, size_of::<TerrainUniform>()),
                ),
                crate::buffers::entry("terrain_directory", tiles.directory),
                crate::buffers::entry("terrain_surface_leaves", tiles.leaves),
                ("water_surface", crate::pipelines::view(tiles.surfaces)),
                ("water_info", crate::pipelines::view(tiles.info)),
                ("water", self.uniform.as_entire_binding()),
                crate::buffers::entry("water_live", looks),
                (
                    "water_tree",
                    sized(tiles.trees, size_of::<SubdivisionUniform>()),
                ),
            ],
        )
    }

    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera: &crate::pipelines::GroupLayout,
        (targets, coming): (Targets, bool),
        formats: (wgpu::TextureFormat, wgpu::TextureFormat),
    ) {
        if (coming || self.tiles.is_some()) && !self.pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, shader, camera, targets, formats, Surface::Tiles);
            self.pipelines.insert(targets, pipeline);
        }
        if (coming || self.tiles.is_some()) && self.foam_pipeline.is_none() {
            self.foam_pipeline = Some(self.foam_pipeline(device, shader, camera));
        }
        if self.falls.is_some() && !self.fall_pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, shader, camera, targets, formats, Surface::Falls);
            self.fall_pipelines.insert(targets, pipeline);
        }
        self.particles
            .ensure_pipelines(device, shader, (camera, &self.layout));
        self.sim.ensure_setup(device, shader);
    }

    fn pipeline_layout(
        &mut self,
        device: &wgpu::Device,
        camera: &crate::pipelines::GroupLayout,
    ) -> wgpu::PipelineLayout {
        self.pipeline_layout
            .get_or_insert_with(|| {
                crate::pipelines::pipeline_layout(
                    device,
                    &crate::pipeline_layouts::WATER_PIPELINES,
                    &[camera, &self.layout, &self.behind_layout],
                )
            })
            .clone()
    }

    fn foam_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera: &crate::pipelines::GroupLayout,
    ) -> wgpu::RenderPipeline {
        let layout = self.pipeline_layout(device, camera);
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
        crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: "engine/water-foam",
                layout: &layout,
                module: shader,
                vertex_constants: &[],
                vertex: crate::shaders::entry::meshes::vs_foam,
                buffers: &[],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth: None,
                fragment: Some(crate::pipelines::Fragment {
                    entry: crate::shaders::entry::meshes::fs_foam,
                    constants: &[],
                    targets: &targets,
                }),
            },
        )
    }

    fn pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera: &crate::pipelines::GroupLayout,
        targets: Targets,
        (canvas, depth): (wgpu::TextureFormat, wgpu::TextureFormat),
        surface: Surface,
    ) -> wgpu::RenderPipeline {
        let layout = self.pipeline_layout(device, camera);
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
            (Targets::Debug, Surface::Tiles) => self
                .shading
                .with(&crate::debug_view::water_constants(self.water_views)),
            _ => self.shading.constants().to_vec(),
        };
        crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: match surface {
                    Surface::Tiles => "engine/water",
                    Surface::Falls => "engine/water-falls",
                },
                layout: &layout,
                module: shader,
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
                fragment: Some(crate::pipelines::Fragment {
                    entry: fragment,
                    constants: &constants,
                    targets: &color_targets,
                }),
            },
        )
    }

    fn draw(
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

impl RenderFeature for WaterPass {
    type Frame = WaterFrame;
    type Stats = WaterStats;

    fn declare(_frame: &WaterFrame, shape: &FrameShape, graph: &mut RenderGraph) {
        let mut water = graph
            .add(Pass::Feature(FeaturePass::Water))
            .read_if(shape.shadows, Resource::ShadowMaps)
            .read_if(shape.subdivision, Resource::Leaves)
            .read(Resource::Probes);
        for &target in shape.main_targets() {
            water = water.color(target, Access::Load);
        }
        water.depth(Resource::Depth, Access::Load);
    }

    fn encode(
        &self,
        _frame: &WaterFrame,
        node: &Node,
        encode: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        self.sim.encode(encoder, encode.profiler);
        if let (Some(_), Some((_, group))) = (&self.tiles, &self.group) {
            self.particles.encode(
                encode.device,
                encoder,
                (encode.water_camera, group),
                encode.profiler,
            );
        }
        self.copy_behind(encoder, encode.depth, encode.scene);
        self.draw_foam(encoder, encode.water_camera);
        let mut pass = encode.render_pass(
            encoder,
            node,
            "engine/water",
            encode.profiler.span("water").map(Span::render),
        );
        self.draw(&mut pass, encode.water_camera, encode.targets)
    }

    fn after_submit(&mut self, _frame: Option<&WaterFrame>) {
        self.particles.after_submit();
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
        copies + falls + self.sim.capacity_bytes() + self.particles.capacity_bytes()
    }

    fn stats(
        this: Option<&Self>,
        frame: Option<&WaterFrame>,
        _snapshot: &RenderSnapshot,
    ) -> WaterStats {
        let [
            live_cells,
            live_budget_cut,
            live_cut,
            live_steps,
            live_behind,
            live_setup,
            live_waiting,
        ] = frame.and(this).map_or([0; 7], |this| this.sim.cells());
        let [touches, dropped_touches] = frame.and(this).map_or([0; 2], |this| this.sim.touches());
        let [leaves, refused] = frame.map_or([0; 2], |frame| frame.subdivided);
        let particles = frame
            .and(this)
            .map_or_else(ParticleStats::default, |this| this.particles.stats());
        WaterStats {
            leaves,
            refused,
            blocks: frame.map_or(0, |frame| frame.blocks[0]),
            dropped_blocks: frame.map_or(0, |frame| frame.blocks[1]),
            live_cells,
            live_budget_cut,
            live_cut,
            live_steps,
            live_behind,
            live_setup,
            live_waiting,
            touches,
            dropped_touches,
            particle_texels: particles.texels,
            foam_points: particles.points,
            foam_points_cut: particles.cut,
            foam_reach: particles.reach,
            foam_particles: particles.foam,
            wave_particles: particles.waves,
            particles_born: particles.born,
            dropped_particles: particles.dropped,
        }
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
        crate::memory::texture(device, &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    };
    let scene = array(
        "engine/water-placeholder-scene",
        wgpu::TextureFormat::Rgba8Unorm,
    );
    Placeholders {
        scene: scene.create_view(&wgpu::TextureViewDescriptor::default()),
        live: crate::buffers::storage(
            device,
            "engine/water-placeholder-live",
            16,
            wgpu::BufferUsages::empty(),
        ),
        storage: crate::buffers::storage(
            device,
            "engine/water-placeholder-storage",
            256,
            wgpu::BufferUsages::empty(),
        ),
        particles: crate::buffers::uniform(
            device,
            "engine/water-placeholder-particles",
            size_of::<crate::gpu_layouts::water::ParticleFrame>() as u64,
        ),
    }
}
