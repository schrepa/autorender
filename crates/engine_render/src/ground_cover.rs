//! Ground cover: the blades of every terrain's surface
//! materials' covers around the detail eye, grown each frame by a compute
//! pass (`cover_grow.wgsl`) into one buffer and drawn with one indirect draw
//! of strips in the opaque pass (`meshes/cover.wgsl`). Within the radius
//! cells are level-0 samples; past it tiers of cells twice as wide each
//! reach on while any blade still stands: blades widen as they thin only as
//! far as their shape lets them stay grass leaves, retire in their ranks'
//! order as they span fewer pixels on screen than the settings ask, those no
//! longer widening narrowing as they recede, and the ground shows the leaves
//! they leave as its canopy, by one law ([`CoverLaw`]). Blades past the
//! buffer's capacity (the most the tiers and densities can grow on the
//! lushest ground, or the settings' budget) drop and count; cells grow in
//! order outward from the eye's corner of each grid, so the farthest tend to
//! drop first. Nothing here exists until a frame has cover to grow.

use crate::shaders::entry::cover_grow;
use engine_core::collections::HashMap;
use std::cell::Cell;
use std::rc::Rc;

use engine_core::Cover;
use engine_core::GroundCover;
use engine_core::glam::{Mat4, Vec3, Vec4};

use crate::bindings::{COVER_DRAW, COVER_GROW};
use crate::buffers::ReadbackRing;
use crate::cached::Cached;
use crate::gpu_layouts::constants::{BLADE_BYTES, COVER_CORE, COVER_LUSH_RATIO};
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

/// Tiers a terrain's cover grows in at most: the radius's and doublings
/// past it. A validity bound (the radius times 128 reaches past any
/// terrain), not a budget.
const MAX_TIERS: usize = 8;

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
    // This frame's farthest reach (meters) and most tiers.
    reach: f32,
    tiers: u32,
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
        }
    }

    /// Meters from the eye this frame's cover reaches, and its most tiers.
    pub(crate) fn reach(&self) -> (f32, u32) {
        (self.reach, self.tiers)
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
    /// and retiring by their distance from the detail eye `eye`. Returns
    /// whether any grows.
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
        let mut uniforms = std::mem::take(&mut self.grow_uniforms);
        uniforms.clear();
        let mut most = 0.0;
        for (index, render) in terrains.terrains.iter().enumerate() {
            let (directory, palette, terms, colors) = terrain.cover_terrain(index);
            // Blades a square meter of the densest cover grows on the
            // lushest ground.
            let density = terms.densest * settings.density * lushest();
            // Surroundings grow no blades.
            if density <= 0.0 || !render.drawn || render.surroundings {
                continue;
            }
            let spacing = render.pyramid.spacing(0);
            let candidates = engine_core::math::max((density * spacing * spacing).ceil(), 1.0);
            let local = eye - render.origin;
            let law = CoverLaw::of(&settings, terms, pixels_per_meter);
            let (tiers, count) = law.tiers();
            self.reach = engine_core::math::max(self.reach, tiers[count - 1].1);
            self.tiers = self.tiers.max(count as u32);
            for &(inner, outer, scale) in &tiers[..count] {
                // Each tier past the radius tries the finest cells'
                // candidates its nearest blades' share reaches.
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
                    tier: [inner, law.radius, law.span, scale],
                    planes,
                    deform_grid: shading.grid,
                    deform: shading.scale,
                    foliage: [f32::from(u8::from(grounded)), 0.0, 0.0, 0.0],
                    view: [pixels_per_meter, 0.0, 0.0, 0.0],
                    widening: [law.widest, 0.0, 0.0, 0.0],
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
        // Room for the most blades the discs can grow, or the budget, within
        // what one storage binding holds: growth leaves the rest out and
        // counts them.
        let room = (self.blades.room() / BLADE_BYTES).min(u64::from(u32::MAX)) as u32;
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

/// How many times its material's own height and density ground cover grows
/// on the lushest ground, and its inverse on the driest: geometrically about
/// its own, the lushest growing COVER_LUSH_RATIO times the driest
/// (`terrain_variety.wgsl`'s `cover_growth`).
fn lushest() -> f32 {
    COVER_LUSH_RATIO.sqrt()
}

/// The widest a widened blade grows for its height and still reads as a
/// grass leaf: the leaf blades of common meadow grasses are 3 to 10 mm wide
/// and 10 to 50 cm long (Hubbard 1984, "Grasses"), the widest of them a
/// tenth as wide as they are long.
const LEAF_ASPECT: f32 = 0.1;

/// How many times their roots' width `cover`'s blades widen at most and all
/// still read as grass leaves: until the shortest, as the driest ground
/// grows it, is LEAF_ASPECT as wide as it is tall; never narrower than
/// their roots.
fn leaf_widening(cover: &Cover) -> f32 {
    let shortest = engine_core::math::min(cover.height[0], cover.height[1]) / lushest();
    engine_core::math::max(LEAF_ASPECT * shortest / cover.width, 1.0)
}

/// A terrain's covers as ground cover's law takes them: its densest one's
/// blades per square meter and their root width (meters), and the most
/// times their roots' width every one's blades widen and still read as
/// grass leaves.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CoverTerms {
    pub(crate) densest: f32,
    pub(crate) width: f32,
    pub(crate) widest: f32,
}

impl CoverTerms {
    /// The terms of a terrain's `covers`: the first densest's, and the
    /// widening the most narrowly shaped allows, so no blade on the terrain
    /// widens past a leaf's shape.
    pub(crate) fn of<'a>(covers: impl IntoIterator<Item = &'a Cover>) -> Self {
        let start = Self {
            widest: 1.0 / COVER_CORE,
            ..Self::default()
        };
        covers.into_iter().fold(start, |terms, cover| {
            let widest = engine_core::math::min(terms.widest, leaf_widening(cover));
            if cover.density > terms.densest {
                Self {
                    densest: cover.density,
                    width: cover.width,
                    widest,
                }
            } else {
                Self { widest, ..terms }
            }
        })
    }
}

/// Ground cover's law on a terrain, which its blades and their canopy
/// follow alike (`terrain_variety.wgsl`'s `CoverLaw`): the radius it grows
/// densest within, the meters from the eye at which its blades past the
/// radius span the settings' pixels (0: cover within the radius alone), and
/// the most times their roots' width its blades widen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CoverLaw {
    pub(crate) radius: f32,
    pub(crate) span: f32,
    pub(crate) widest: f32,
}

impl CoverLaw {
    /// The law of `settings` on a terrain whose covers' terms are `terms`,
    /// seen at `pixels_per_meter`: blades widen as they thin within the
    /// radius as far as its covers' shapes let them all stay leaves, from 1
    /// to 1 / COVER_CORE times their roots', and past the radius keep that
    /// width. Every cover on the terrain retires by its densest's span, so
    /// the canopy, which knows no blade's width, thins by one law with them.
    pub(crate) fn of(settings: &GroundCover, terms: CoverTerms, pixels_per_meter: f32) -> Self {
        let widest = engine_core::math::clamp(terms.widest, 1.0, 1.0 / COVER_CORE);
        let span = if settings.pixels > 0.0 {
            terms.width * widest * pixels_per_meter / settings.pixels
        } else {
            0.0
        };
        Self {
            radius: settings.radius,
            span,
            widest,
        }
    }

    /// Meters from the eye its tiers reach: twice its span, where the last
    /// of its blades, by its rank, retires spanning half the settings'
    /// pixels (`terrain_variety.wgsl`'s `seen_share`); 0, no tiers, where
    /// that falls within the radius.
    fn reach(self) -> f32 {
        let reach = 2.0 * self.span;
        if reach > self.radius { reach } else { 0.0 }
    }

    /// Its tiers, each one's inner and outer meters and its cells' width in
    /// level-0 samples: the radius's, then out to its reach each reaching
    /// twice as far as the last with cells twice as wide; and how many.
    fn tiers(self) -> ([(f32, f32, f32); MAX_TIERS], usize) {
        let reach = self.reach();
        let mut tiers = [(0.0, self.radius, 1.0); MAX_TIERS];
        let mut count = 1;
        while reach > 0.0 && count < MAX_TIERS {
            let (_, outer, scale) = tiers[count - 1];
            if outer >= reach {
                break;
            }
            tiers[count] = (outer, engine_core::math::min(2.0 * outer, reach), 2.0 * scale);
            count += 1;
        }
        (tiers, count)
    }

    /// `terrain_variety.wgsl`'s `cover_widening`: how many times wider than
    /// their roots blades grow `away` meters from the eye.
    fn widening(self, away: f32) -> f32 {
        engine_core::math::clamp(away / (COVER_CORE * self.radius), 1.0, self.widest)
    }

    /// `terrain_variety.wgsl`'s `cover_share`: the most of the densest cover
    /// growing `away` meters from the eye.
    fn share(self, away: f32) -> f32 {
        let distance = engine_core::math::max(away, 1.0e-3);
        let mut share = engine_core::math::min(1.0, COVER_CORE * self.radius / distance);
        if self.span <= 0.0 {
            return share * (1.0 - smoothstep(0.5 * self.radius, self.radius, away));
        }
        if distance > self.radius {
            let ratio = self.radius / distance;
            share = COVER_CORE * ratio * ratio;
        }
        share * seen_share(self.span * self.widening(distance) / (self.widest * distance))
    }

    /// The square meters of densest cover its radius and tiers grow: their
    /// area weighted by its share, summed ring by ring.
    fn area(self) -> f32 {
        const RINGS: u32 = 256;
        let end = engine_core::math::max(self.radius, self.reach());
        let width = end / RINGS as f32;
        (0..RINGS)
            .map(|ring| {
                let away = (ring as f32 + 0.5) * width;
                std::f32::consts::TAU * away * width * self.share(away)
            })
            .sum()
    }
}

/// `terrain_variety.wgsl`'s `seen_share`: the share of things retiring by
/// their size on screen still showing where each spans `pixels` pixels.
fn seen_share(pixels: f32) -> f32 {
    let octaves = engine_core::math::max(pixels, 1.0e-6).log2();
    engine_core::math::clamp(0.5 + 0.5 * octaves, 0.0, 1.0)
}

fn smoothstep(from: f32, to: f32, value: f32) -> f32 {
    let t = engine_core::math::clamp((value - from) / (to - from), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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

    /// Blades the last read-back frame drew and those past capacity, and how
    /// far and in how many tiers the cover reaches.
    fn stats(
        this: Option<&Self>,
        _frame: Option<&CoverFrame>,
        _snapshot: &engine_core::RenderSnapshot,
    ) -> engine_core::GroundCoverStats {
        let (totals, (reach, tiers)) =
            this.map_or(([0; 2], (0.0, 0)), |cover| (cover.totals(), cover.reach()));
        engine_core::GroundCoverStats {
            blades: totals[0],
            dropped: totals[1],
            reach: reach.round() as u32,
            tiers,
        }
    }
}
