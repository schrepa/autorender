//! Ground cover: the blades of every terrain's surface
//! materials' covers around the detail eye, grown each frame by a compute
//! pass (`cover_grow.wgsl`) into one buffer and drawn with one indirect draw
//! of strips in the opaque pass (`meshes/cover.wgsl`). Blades stand by their
//! size on screen alone (`terrain_variety.wgsl`'s `cover_share`): where
//! blades as wide as their roots would span fewer pixels than the settings
//! ask (by default the steady size, `engine_core::STEADY_PIXELS`), a share
//! of them stands by their fixed ranks, each widened by its inverse, so
//! they cover as much ground, their colors drawn toward their canopy's mean
//! by the root of that share, so they vary over a pixel as all of them
//! would (stochastic simplification of aggregate detail, Cook, Halstead,
//! Planck and Ryu 2007); and they retire in that order as their height
//! spans fewer, the ground showing the leaves they leave as its canopy.
//! Every blade stands in a first tier of cells of level-0 samples; past it
//! tiers of cells twice as wide each reach on to where the last retires.
//! Where the blades the view takes in would pass
//! the settings' budget, or what one storage binding holds, they widen
//! further, fewer covering as much ground, so a budget coarsens the grass,
//! never its cover; blades past the buffer's capacity still drop and count,
//! cells growing in order outward from the eye's corner of each grid, so
//! the farthest tend to drop first. Nothing here exists until a frame has
//! cover to grow.

use crate::shaders::entry::cover_grow;
use engine_core::collections::HashMap;
use std::cell::Cell;
use std::rc::Rc;

use engine_core::GroundCover;
use engine_core::glam::{Mat4, Vec3, Vec4};

use crate::bindings::{COVER_DRAW, COVER_GROW};
use crate::buffers::ReadbackRing;
use crate::cached::Cached;
use crate::gpu_layouts::constants::BLADE_BYTES;
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::Targets;
use crate::render_graph::{FrameShape, Node, Pass, RenderGraph, Resource};
use crate::scene_shading::SceneShading;
use crate::terrain::TerrainPass;

/// Uniform offset alignment between terrains' growth.
const SLOT: u64 = 512;
/// Bytes of the growth's uniform (`Grow`).
pub(crate) const GROW_BYTES: u64 = size_of::<GrowUniform>() as u64;

use crate::gpu_layouts::ground_cover::GrowUniform;

use crate::gpu_layouts::ground_cover::FrameUniform;

/// Bytes of one of the blades' indirect draws: whole blades', then
/// straight ones'.
const DRAW_ARGS: u64 = size_of::<wgpu::util::DrawIndirectArgs>() as u64;

/// One tier of one terrain's growth this frame: its uniform's offset and
/// its grid's cells along each side.
struct Growth {
    offset: u32,
    cells: u32,
}

/// Tiers a terrain's cover grows in at most: the first, where every blade
/// stands, and doublings past it, the last reaching all the way to where
/// the last blade retires. A validity bound (the first tier reaches as far
/// as blades stand as wide as their roots, the last where their height
/// retires them, a few doublings further, more only as the budget widens
/// them), keeping a tier's finest cells within what a workgroup counts, not
/// a budget.
const MAX_TIERS: usize = 12;

pub(crate) struct GroundCoverPass {
    grow_layout: crate::pipelines::GroupLayout,
    draw_layout: crate::pipelines::GroupLayout,
    // The growth's and the finish's pipelines, created with the first
    // frame that has cover.
    compute: Option<(wgpu::ComputePipeline, wgpu::ComputePipeline)>,
    // The blades' pipelines, and the scene features they are specialized
    // for.
    pipelines: HashMap<(Targets, bool), wgpu::RenderPipeline>,
    shading: SceneShading,
    blades: crate::buffers::Growable,
    capacity: u32,
    blades_generation: u64,
    counts: wgpu::Buffer,
    args: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    frame: wgpu::Buffer,
    grow_group: Cached<(u64, u64, u64, u64, u64), wgpu::BindGroup>,
    draw_group: Cached<u64, wgpu::BindGroup>,
    growths: Vec<Growth>,
    // The growths' uniforms, and their bytes as written, each at its slot,
    // kept from frame to frame.
    grow_uniforms: Vec<GrowUniform>,
    grow_bytes: Vec<u8>,
    // Last frame's presentation time, which blades swayed at then.
    previous_time: Option<f32>,
    readbacks: ReadbackRing<u32>,
    // Blades grown and those past capacity, read back a few frames late.
    totals: Rc<Cell<Option<[u32; 2]>>>,
    // This frame's farthest reach (meters), most tiers, and how many times
    // the settings' pixels the budget widens the blades that thin.
    reach: f32,
    tiers: u32,
    widened: f32,
}

impl GroundCoverPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            grow_layout: crate::pipelines::layout(device, &COVER_GROW),
            draw_layout: crate::pipelines::layout(device, &COVER_DRAW),
            compute: None,
            pipelines: HashMap::default(),
            shading: SceneShading::ALL,
            blades: crate::buffers::Growable::new(
                device,
                "engine/cover-blades",
                BLADE_BYTES,
                wgpu::BufferUsages::empty(),
            ),
            capacity: 0,
            blades_generation: 0,
            counts: crate::buffers::storage(
                device,
                "engine/cover-counts",
                16,
                wgpu::BufferUsages::COPY_SRC,
            ),
            args: crate::buffers::storage(
                device,
                "engine/cover-draw",
                2 * DRAW_ARGS,
                wgpu::BufferUsages::INDIRECT,
            ),
            uniforms: crate::buffers::uniform(device, "engine/cover-grow", SLOT),
            frame: crate::buffers::uniform(
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
            widened: 1.0,
        }
    }

    /// Meters from the eye this frame's cover reaches, its most tiers, and
    /// how many times the settings' pixels the budget widens the blades
    /// that thin.
    pub(crate) fn reach(&self) -> (f32, u32, f32) {
        (self.reach, self.tiers, self.widened)
    }

    /// Blades the last read-back frame drew, and those past capacity.
    pub(crate) fn totals(&self) -> [u32; 2] {
        self.totals.get().unwrap_or([0; 2])
    }

    /// Bytes of the blades' buffer.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.blades.size()
    }

    /// Plans this frame's growth for every terrain whose materials have
    /// cover, with `settings` on, seen through `culling` (world) on a view
    /// `height` pixels tall at presentation time `time`, its blades growing
    /// and retiring by their size on screen from the detail eye `eye`.
    /// Returns whether any grows.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrain: &TerrainPass,
        terrains: &engine_core::RenderTerrains,
        (settings, wind, culling, eye, time): (&GroundCover, [f32; 2], Mat4, Vec3, f32),
        (height, grounded): (f32, bool),
    ) -> bool {
        self.growths.clear();
        self.reach = 0.0;
        self.tiers = 0;
        self.widened = 1.0;
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
            // Nothing grows: none drawn, whatever the last read-back said.
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
        queue: &wgpu::Queue,
        terrain: &TerrainPass,
        terrains: &engine_core::RenderTerrains,
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
        let shading = sources.shading;
        // Blades one storage binding holds, within the settings' budget.
        let room = (self.blades.room() / BLADE_BYTES).min(u64::from(u32::MAX)) as u32;
        let budget = settings
            .max_blades
            .map_or(room, |budget| budget.min(room))
            .max(1);
        // Where the blades the view takes in at the settings' pixels would
        // pass the budget, those that thin widen further until they fit:
        // fewer, wider blades covering as much ground.
        let grown: f32 = (terrains.terrains.iter().enumerate())
            .filter_map(|(index, render)| {
                CoverLaw::of(terrain, (index, render), &settings, (pixels_per_meter, 1.0))
            })
            .map(|law| law.blades())
            .sum();
        let widened = engine_core::math::max(grown * view_share(culling) / budget as f32, 1.0);
        self.widened = widened;
        let mut uniforms = std::mem::take(&mut self.grow_uniforms);
        uniforms.clear();
        let mut most = 0.0;
        for (index, render) in terrains.terrains.iter().enumerate() {
            let view = (pixels_per_meter, widened);
            let Some(law) = CoverLaw::of(terrain, (index, render), &settings, view) else {
                continue;
            };
            let (directory, palette, _, colors) = terrain.cover_terrain(index);
            let spacing = render.pyramid.spacing(0);
            let candidates = engine_core::math::max((law.density * spacing * spacing).ceil(), 1.0);
            let local = eye - render.origin;
            let reach = cover_reach(law.retire);
            // Every blade stands within the first tier, of level-0 cells: as
            // far as none thins or retires yet (`cover_share`).
            let whole = engine_core::math::min(law.thin, 0.5 * law.retire);
            let mut tiers = [(0.0, whole, 1.0); MAX_TIERS];
            let mut count = 1;
            while count < MAX_TIERS {
                let (_, outer, scale) = tiers[count - 1];
                if outer >= reach {
                    break;
                }
                let last = count + 1 == MAX_TIERS;
                let next = if last {
                    reach
                } else {
                    engine_core::math::min(2.0 * outer, reach)
                };
                tiers[count] = (outer, next, 2.0 * scale);
                count += 1;
            }
            let reached = tiers[count - 1].1;
            self.reach = engine_core::math::max(self.reach, reached);
            self.tiers = self.tiers.max(count as u32);
            for &(inner, outer, scale) in &tiers[..count] {
                // Each tier past the first tries the finest cells'
                // candidates its nearest blades' share reaches.
                let tried = if scale > 1.0 {
                    engine_core::math::min(
                        (candidates * cover_share(inner, law.thin, law.retire)).ceil(),
                        candidates,
                    )
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
                        terrain.cover_tallest(),
                    ],
                    tier: [inner, law.thin, law.retire, scale],
                    planes,
                    deform_grid: shading.grid,
                    deform: shading.scale,
                    foliage: [f32::from(u8::from(grounded)), 0.0, 0.0, 0.0],
                    view: [pixels_per_meter, terrain.cover_widest(), 0.0, 0.0],
                };
                self.growths.push(Growth {
                    offset: (uniforms.len() as u64 * SLOT) as u32,
                    cells,
                });
                uniforms.push(uniform);
            }
            most += law.blades();
        }
        if self.growths.is_empty() {
            self.grow_uniforms = uniforms;
            return false;
        }
        // Room for the most blades the discs can grow, or the budget, within
        // what one storage binding holds: growth leaves the rest out and
        // counts them.
        let capacity = settings
            .max_blades
            .map_or(most.ceil() as u32, |budget| budget.min(most.ceil() as u32))
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
            self.uniforms = crate::buffers::uniform(
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
        self.compute
            .get_or_insert_with(|| compute(device, &self.grow_layout));
        let key = (
            sources.generation.0,
            sources.generation.1,
            sources.generation.2,
            self.blades_generation,
            sources.generation.3,
        );
        if self.grow_group.stale(&key) {
            let group = crate::pipelines::group(
                device,
                "engine/cover-grow",
                &COVER_GROW,
                &self.grow_layout,
                &[
                    ("heights", crate::pipelines::view(sources.heights)),
                    ("control", crate::pipelines::view(sources.control)),
                    crate::buffers::entry("directory", sources.directory),
                    crate::buffers::entry("palette", sources.palette),
                    crate::buffers::entry("covers", sources.covers),
                    (
                        "grow",
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.uniforms,
                            offset: 0,
                            size: wgpu::BufferSize::new(GROW_BYTES),
                        }),
                    ),
                    crate::buffers::entry("blades", &self.blades),
                    crate::buffers::entry("counts", &self.counts),
                    crate::buffers::entry("args", &self.args),
                    crate::buffers::entry("deformation", sources.deformation),
                    ("macro_maps", crate::pipelines::view(sources.macro_maps)),
                    ("normals", crate::pipelines::view(sources.normals)),
                ],
            );
            self.grow_group.set(key, group);
        }
        if self.draw_group.stale(&self.blades_generation) {
            let group = crate::pipelines::group(
                device,
                "engine/cover",
                &COVER_DRAW,
                &self.draw_layout,
                &[
                    crate::buffers::entry("cover_blades", &self.blades),
                    crate::buffers::entry("cover_frame", &self.frame),
                ],
            );
            self.draw_group.set(self.blades_generation, group);
        }
        true
    }

    /// Records this frame's growth and the indirect draw's arguments, and
    /// a copy of the counts to read back, timed by `profiler`
    /// (`ground_cover`).
    fn encode_growth(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
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

    /// Whether this frame grows blades: what its pass records.
    pub(crate) fn growing(&self) -> bool {
        self.compute.is_some() && self.grow_group.get().is_some() && !self.growths.is_empty()
    }

    /// Creates the growth's pipelines unless they exist.
    pub(crate) fn warm(&mut self, device: &wgpu::Device) {
        self.compute
            .get_or_insert_with(|| compute(device, &self.grow_layout));
    }

    /// Specializes the blades' pipelines for `shading`'s scene features:
    /// those made for others are dropped, to be created anew.
    pub(crate) fn specialize(&mut self, shading: SceneShading) {
        if self.shading != shading {
            self.shading = shading;
            self.pipelines.clear();
        }
    }

    /// Creates the pipeline drawing blades into `targets` (`lit` or not)
    /// unless it exists.
    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        (camera, empty, lights): (
            &crate::pipelines::GroupLayout,
            &crate::pipelines::GroupLayout,
            Option<&crate::pipelines::GroupLayout>,
        ),
        (targets, lit): (Targets, bool),
        canvas: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
    ) {
        if self.pipelines.contains_key(&(targets, lit)) {
            return;
        }
        let layout = if lit {
            let Some(lights) = lights else {
                return;
            };
            crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::LIT_COVER_PIPELINES,
                &[camera, &self.draw_layout, empty, lights],
            )
        } else {
            crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::COVER_PIPELINES,
                &[camera, &self.draw_layout],
            )
        };
        let (vertex, fragment) = crate::entries::cover(targets, lit);
        let color_targets = targets.color_targets(canvas);
        let pipeline = crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: "engine/cover",
                layout: &layout,
                module: shader,
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
                fragment: Some(crate::pipelines::Fragment {
                    entry: fragment,
                    constants: &self.shading.constants(),
                    targets: &color_targets,
                }),
            },
        );
        self.pipelines.insert((targets, lit), pipeline);
    }

    /// Draws this frame's blades into `targets` (`lit` or not), the camera
    /// and lights already bound; returns the draws.
    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass<'_>, targets: Targets, lit: bool) -> u32 {
        let (Some(pipeline), Some(group)) =
            (self.pipelines.get(&(targets, lit)), self.draw_group.get())
        else {
            return 0;
        };
        if self.growths.is_empty() {
            return 0;
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(1, group, &[]);
        // Whole blades, then straight ones.
        pass.draw_indirect(&self.args, 0);
        pass.draw_indirect(&self.args, DRAW_ARGS);
        2
    }
}

/// Pixels a meter spans a meter from the eye along a view `height` pixels
/// tall seen through `culling`.
pub(crate) fn pixels_per_meter(culling: Mat4, height: f32) -> f32 {
    culling.row(1).truncate().length() * height * 0.5
}

/// The share of a turn around the eye a view seen through `culling` takes
/// in across: its horizontal field of view over a whole turn, the share of
/// the cover around the eye it draws.
fn view_share(culling: Mat4) -> f32 {
    let across = engine_core::math::max(culling.row(0).truncate().length(), 1.0e-6);
    engine_core::math::min(2.0 * (1.0 / across).atan() / std::f32::consts::TAU, 1.0)
}

/// How cover grows on one terrain this frame: its densest cover's blades a
/// square meter, the settings' density applied, and the meters from the eye
/// past which its blades thin and widen and around which they retire
/// (`terrain_variety.wgsl`'s `cover_share`).
#[derive(Clone, Copy)]
struct CoverLaw {
    density: f32,
    thin: f32,
    retire: f32,
}

impl CoverLaw {
    /// The law of terrain `render`, the `index`th of the frame's, under
    /// `settings`, seen at `pixels_per_meter`, the blades that thin standing
    /// `widened` times the settings' pixels wide on screen; `None` where it
    /// grows none: surroundings, terrains not drawn and those without cover.
    fn of(
        terrain: &TerrainPass,
        (index, render): (usize, &engine_core::RenderTerrain),
        settings: &GroundCover,
        (pixels_per_meter, widened): (f32, f32),
    ) -> Option<Self> {
        let (_, _, [densest, width, height], _) = terrain.cover_terrain(index);
        let density = densest * settings.density;
        // Surroundings grow no blades.
        if density <= 0.0 || !render.drawn || render.surroundings {
            return None;
        }
        Some(Self {
            density,
            thin: width * pixels_per_meter / (settings.pixels * widened),
            retire: cover_retire(settings, height, pixels_per_meter),
        })
    }

    /// The blades it grows all around the eye: its density over the ground
    /// its blades stand on, by their share, ring by ring out to its reach.
    fn blades(&self) -> f32 {
        const RINGS: u32 = 256;
        let width = cover_reach(self.retire) / RINGS as f32;
        let area: f32 = (0..RINGS)
            .map(|ring| {
                let away = (ring as f32 + 0.5) * width;
                std::f32::consts::TAU * away * width * cover_share(away, self.thin, self.retire)
            })
            .sum();
        area * self.density
    }
}

/// The meters from the eye at which blades `height` meters tall, on a
/// terrain whose densest cover stands that tall on average, span
/// `settings.pixels` pixels on screen, seen at `pixels_per_meter`, around
/// which they retire (`terrain_variety.wgsl`'s `cover_shown`). Every cover
/// on the terrain retires by its densest's height, so the canopy, which
/// knows no blade's height, takes over by one law with them.
pub(crate) fn cover_retire(settings: &GroundCover, height: f32, pixels_per_meter: f32) -> f32 {
    height * pixels_per_meter / settings.pixels
}

/// Meters from the eye cover reaches, for blades retiring around `retire`
/// meters: twice that, where the last of them, by its rank, retires
/// spanning half as many pixels (`terrain_variety.wgsl`'s `seen_share`).
fn cover_reach(retire: f32) -> f32 {
    2.0 * retire
}

/// `terrain_variety.wgsl`'s `cover_share`: the share of the densest cover's
/// blades standing `away` meters from the eye, for blades thinning past
/// `thin` meters and retiring around `retire`.
fn cover_share(away: f32, thin: f32, retire: f32) -> f32 {
    let widening = engine_core::math::max(away / engine_core::math::max(thin, 1.0e-6), 1.0);
    seen_share(retire / engine_core::math::max(away, 1.0e-3)) / widening
}

/// `terrain_variety.wgsl`'s `seen_share`: the share of things retiring by
/// their size on screen still showing where each spans `pixels` pixels.
fn seen_share(pixels: f32) -> f32 {
    let octaves = engine_core::math::max(pixels, 1.0e-6).log2();
    engine_core::math::clamp(0.5 + 0.5 * octaves, 0.0, 1.0)
}

/// The view's frustum planes from its view-projection (world), each
/// normalized with its inside positive.
pub(crate) fn planes(view_projection: Mat4) -> [[f32; 4]; 6] {
    let rows = [
        view_projection.row(0),
        view_projection.row(1),
        view_projection.row(2),
        view_projection.row(3),
    ];
    // Reversed depth: the near plane is z = w, the far z = 0.
    [
        rows[3] + rows[0],
        rows[3] - rows[0],
        rows[3] + rows[1],
        rows[3] - rows[1],
        rows[2],
        rows[3] - rows[2],
    ]
    .map(|plane: Vec4| {
        (plane / engine_core::math::max(plane.truncate().length(), 1.0e-12)).to_array()
    })
}

fn compute(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
) -> (wgpu::ComputePipeline, wgpu::ComputePipeline) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/cover-grow.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::COVER_GROW_SHADER.into()),
    });
    let pipeline_layout = crate::pipelines::pipeline_layout(
        device,
        &crate::pipeline_layouts::COVER_GROW_PIPELINES,
        &[layout],
    );
    (
        crate::pipelines::compute_entry(device, &pipeline_layout, &module, cover_grow::grow_cover),
        crate::pipelines::compute_entry(
            device,
            &pipeline_layout,
            &module,
            cover_grow::finish_cover,
        ),
    )
}

/// A frame that grows blades.
pub(crate) struct CoverFrame;

impl RenderFeature for GroundCoverPass {
    type Frame = CoverFrame;
    type Stats = engine_core::GroundCoverStats;

    /// Blades grow over the tracks pressed into the ground.
    fn declare(_frame: &CoverFrame, _shape: &FrameShape, graph: &mut RenderGraph) {
        graph
            .add(Pass::Feature(FeaturePass::GroundCover))
            .read(Resource::Tracks)
            .write(Resource::Cover);
    }

    fn encode(
        &self,
        _frame: &CoverFrame,
        _node: &Node,
        encode: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        self.encode_growth(encode.device, encoder, encode.profiler);
        0
    }

    /// Requests the map of this frame's counts once it is submitted.
    fn after_submit(&mut self, _frame: Option<&CoverFrame>) {
        let totals = Rc::clone(&self.totals);
        self.readbacks.after_submit(move |words, capacity| {
            let grown = words.get(0);
            totals.set(Some([grown.min(capacity), grown.saturating_sub(capacity)]));
        });
    }

    fn capacity_bytes(&self) -> u64 {
        GroundCoverPass::capacity_bytes(self)
    }

    /// Blades the last read-back frame drew and those past capacity, how
    /// far and in how many tiers the cover reaches, and how far the budget
    /// widens the blades that thin.
    fn stats(
        this: Option<&Self>,
        _frame: Option<&CoverFrame>,
        _snapshot: &engine_core::RenderSnapshot,
    ) -> engine_core::GroundCoverStats {
        let (totals, (reach, tiers, widened)) = this.map_or(([0; 2], (0.0, 0, 1.0)), |cover| {
            (cover.totals(), cover.reach())
        });
        engine_core::GroundCoverStats {
            blades: totals[0],
            dropped: totals[1],
            reach: reach.round() as u32,
            tiers,
            widened: (100.0 * widened).round() as u32,
        }
    }
}
