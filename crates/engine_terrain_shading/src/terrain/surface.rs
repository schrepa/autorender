use std::cell::Cell;
use std::rc::Rc;

use super::*;
use crate::bindings::{TERRAIN_SUBDIVISION, TERRAIN_SUBDIVISION_CAMERA};
use crate::gpu_layouts::constants::{
    SUBDIVISION_ARGS, SUBDIVISION_COUNTED, SUBDIVISION_DRAW, SUBDIVISION_LEAVES,
    SUBDIVISION_REFUSED, SURFACE_MESHLET_INDICES, SURFACE_MESHLET_LEVEL, SURFACE_MESHLET_VERTICES,
};
use crate::shaders::entry::terrain as draw;
use engine_faces::heightfield::{SubdivisionUniform, SurfaceDraw};
use engine_render::buffers::{Growable, ReadbackRing};
use engine_render::cached::{Bound, GroupCache, Versioned};

const LEAF_BYTES: u64 = 24;
const NODE_BYTES: u64 = 4;
const ARGS_BYTES: u64 = SUBDIVISION_ARGS as u64 * 4;
const DRAW_OFFSET: u64 = SUBDIVISION_DRAW as u64 * 4;
const SHALLOWEST: u32 = SUBDIVISION_COUNTED + 2;
const _: () = assert!(SURFACE_MESHLET_VERTICES <= 1 << 16, "meshlet_fits_u16");

type Spec = (&'static str, u64, wgpu::BufferUsages);

const LEAVES: Spec = (
    "engine/terrain-surface-leaves",
    LEAF_BYTES,
    wgpu::BufferUsages::empty(),
);
const NODES: Spec = (
    "engine/terrain-surface-nodes",
    NODE_BYTES,
    wgpu::BufferUsages::empty(),
);
const COUNTERS: Spec = (
    "engine/terrain-surface-counters",
    ARGS_BYTES,
    wgpu::BufferUsages::COPY_SRC,
);
const ARGS: Spec = (
    "engine/terrain-surface-args",
    ARGS_BYTES,
    wgpu::BufferUsages::INDIRECT,
);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Measure {
    Ground,
    Water,
}

impl Measure {
    fn code(self) -> f32 {
        match self {
            Self::Ground => 0.0,
            Self::Water => 1.0,
        }
    }
}

struct Tree {
    buffer: Versioned<wgpu::Buffer>,
    depth: u32,
    side: f32,
    room: u32,
    group: GroupCache<5>,
}

#[derive(Clone, Copy)]
struct Subdivided {
    id: u64,
    measure: Measure,
    terrain: usize,
    first: u32,
    finest: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SurfaceStats {
    pub leaves: u32,
    pub triangles: u32,
    pub refused: u32,
    pub shallow: u32,
}

pub(crate) struct SurfaceTrees {
    trees: HashMap<(u64, Measure), Tree>,
    frame: Vec<Option<Subdivided>>,
    grounds: usize,
    targets: [f32; 2],
    leaves: Growable,
    nodes: Growable,
    counters: Growable,
    args: Growable,
    indices: wgpu::Buffer,
    uniforms: Versioned<wgpu::Buffer>,
    uniforms_written: Vec<u8>,
    pub(super) stand_in: wgpu::Buffer,
    camera_layout: engine_render::pipelines::GroupLayout,
    tree_layout: engine_render::pipelines::GroupLayout,
    camera_group: Option<wgpu::BindGroup>,
    pipelines: Option<[wgpu::ComputePipeline; 3]>,
    frames: u64,
    readbacks: ReadbackRing<(u32, u32, u64)>,
    totals: Rc<Cell<[u32; 4]>>,
    rests: Rc<Cell<u64>>,
    shallow: u32,
    held: bool,
}

impl SurfaceTrees {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            trees: HashMap::default(),
            frame: Vec::new(),
            grounds: 0,
            targets: [0.0; 2],
            leaves: growable(device, LEAVES),
            nodes: growable(device, NODES),
            counters: growable(device, COUNTERS),
            args: growable(device, ARGS),
            indices: engine_render::memory::buffer_init(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("engine/terrain-surface-indices"),
                    contents: bytemuck::cast_slice(&meshlet_indices()),
                    usage: wgpu::BufferUsages::INDEX,
                },
            ),
            uniforms: Versioned::new(subdivision_uniforms(device, SLOT)),
            uniforms_written: Vec::new(),
            stand_in: engine_render::buffers::storage(
                device,
                "engine/terrain-surface-stand-in",
                LEAF_BYTES,
                wgpu::BufferUsages::empty(),
            ),
            camera_layout: engine_render::pipelines::layout(device, &TERRAIN_SUBDIVISION_CAMERA),
            tree_layout: engine_render::pipelines::layout(device, &TERRAIN_SUBDIVISION),
            camera_group: None,
            pipelines: None,
            frames: 0,
            readbacks: ReadbackRing::new("engine/terrain-surface-readback"),
            totals: Rc::new(Cell::new([0; 4])),
            rests: Rc::new(Cell::new(0)),
            shallow: 0,
            held: false,
        }
    }

    pub(super) fn leaves(&self) -> &Versioned<wgpu::Buffer> {
        &self.leaves
    }

    pub(super) fn uniforms(&self) -> &Versioned<wgpu::Buffer> {
        &self.uniforms
    }

    pub(super) fn capacity_bytes(&self) -> u64 {
        let trees: u64 = self.trees.values().map(|tree| tree.buffer.size()).sum();
        let frame = self.leaves.size() + self.nodes.size() + self.counters.size();
        trees + frame + self.args.size() + self.indices.size()
    }

    fn ground(&self, index: usize) -> Option<Subdivided> {
        (index < self.grounds)
            .then(|| self.frame.get(index).copied().flatten())
            .flatten()
    }

    pub(super) fn of(&self, index: usize) -> (u32, f32) {
        self.ground(index)
            .map_or((0, 0.0), |subdivided| (subdivided.first, subdivided.finest))
    }

    pub(super) fn any(&self) -> bool {
        self.frame.iter().any(Option::is_some)
    }

    pub(super) fn rest(&mut self, device: &wgpu::Device) {
        self.frame.clear();
        self.grounds = 0;
        self.shallow = 0;
        self.totals.set([0; 4]);
        self.rests.set(self.rests.get() + 1);
        if !std::mem::take(&mut self.held) {
            return;
        }
        for tree in self.trees.values() {
            engine_render::memory::release_buffer(&tree.buffer);
        }
        self.trees.clear();
        for (buffer, frame) in [
            (&mut self.leaves, LEAVES),
            (&mut self.nodes, NODES),
            (&mut self.counters, COUNTERS),
            (&mut self.args, ARGS),
        ] {
            engine_render::memory::release_buffer(buffer);
            *buffer = growable(device, frame);
        }
        self.uniforms.replace(subdivision_uniforms(device, SLOT));
        self.uniforms_written.clear();
        self.readbacks.clear();
        self.camera_group = None;
        self.pipelines = None;
    }

    pub(super) fn draw(&self, index: usize) -> Option<SurfaceDraw> {
        self.ground(index).map(|_| self.slot_draw(index))
    }

    pub(super) fn water(&self, index: usize) -> Option<(u32, SurfaceDraw)> {
        let slot = (self.grounds..self.frame.len()).find(|&slot| {
            self.frame
                .get(slot)
                .copied()
                .flatten()
                .is_some_and(|subdivided| subdivided.terrain == index)
        })?;
        Some(((slot as u64 * SLOT) as u32, self.slot_draw(slot)))
    }

    fn slot_draw(&self, slot: usize) -> SurfaceDraw {
        SurfaceDraw {
            indices: self.indices.clone(),
            args: wgpu::Buffer::clone(&self.args),
            offset: slot as u64 * ARGS_BYTES + DRAW_OFFSET,
        }
    }

    pub(super) fn targets(&self) -> [f32; 2] {
        self.targets
    }

    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        terrains: &RenderTerrains,
        views: &TerrainViews<'_>,
        holds_water: impl Fn(u64) -> bool,
    ) {
        self.frames += 1;
        self.shallow = 0;
        self.frame.clear();
        self.held = true;
        let (settings, water) = (&views.settings, &views.water);
        let subdivides_water = |terrain: &RenderTerrain| water.enabled && holds_water(terrain.id);
        self.trees.retain(|&(id, measure), _| {
            terrains.terrains.iter().any(|terrain| {
                terrain.id == id && (measure == Measure::Ground || subdivides_water(terrain))
            })
        });
        let watered: Vec<usize> = (terrains.terrains.iter().enumerate())
            .filter(|&(_, terrain)| terrain.drawn && subdivides_water(terrain))
            .map(|(index, _)| index)
            .collect();
        let count = terrains.terrains.len();
        let slots = (count + watered.len()) as u64;
        let share = self.leaves.room() / LEAF_BYTES / slots.max(1);
        let rooms = [
            share.min(u64::from(settings.subdivision_leaves)) as u32,
            share.min(u64::from(water.subdivision_leaves)) as u32,
        ];
        let leaves =
            count as u64 * u64::from(rooms[0]) + watered.len() as u64 * u64::from(rooms[1]);
        self.leaves.grow(device, leaves * LEAF_BYTES);
        self.nodes.grow(device, leaves * NODE_BYTES);
        self.counters.grow(device, slots * ARGS_BYTES);
        self.args.grow(device, slots * ARGS_BYTES);
        if self.uniforms_written.len() as u64 != slots * SLOT {
            self.readbacks.clear();
        }
        if slots * SLOT > self.uniforms.size() {
            let bytes = (slots * SLOT).next_power_of_two();
            self.uniforms.replace(subdivision_uniforms(device, bytes));
            self.uniforms_written.clear();
        }
        self.targets = [settings.subdivision_error, water.subdivision_error];
        let planes = world_planes(views.culling, views.eye);
        let mut bytes = vec![0_u8; (slots * SLOT) as usize];
        let depth = settings.subdivision_depth;
        for (index, terrain) in terrains.terrains.iter().enumerate() {
            if terrain.drawn {
                let leaves = (index as u32 * rooms[0], rooms[0]);
                let tree = (terrain, index, Measure::Ground);
                self.lay_out((device, queue), tree, (depth, leaves), planes, &mut bytes);
            } else {
                self.frame.push(None);
            }
        }
        self.grounds = count;
        let after_ground = count as u32 * rooms[0];
        for (place, &index) in watered.iter().enumerate() {
            let leaves = (after_ground + place as u32 * rooms[1], rooms[1]);
            let tree = (&terrains.terrains[index], index, Measure::Water);
            self.lay_out((device, queue), tree, (depth, leaves), planes, &mut bytes);
        }
        if bytes != self.uniforms_written {
            queue.write_buffer(&self.uniforms, 0, &bytes);
            self.uniforms_written = bytes;
        }
    }

    fn lay_out(
        &mut self,
        (device, queue): (&wgpu::Device, &engine_render::gpu_calls::Queue),
        (terrain, index, measure): (&RenderTerrain, usize, Measure),
        (depth, (first, room)): (u32, (u32, u32)),
        planes: [[f32; 4]; 6],
        bytes: &mut [u8],
    ) {
        let slot = self.frame.len();
        let (side, depth) = self.tree(device, queue, (terrain, measure), (depth, room));
        let target = match measure {
            Measure::Ground => self.targets[0],
            Measure::Water => self.targets[1],
        };
        let uniform = SubdivisionUniform {
            square: [side, target, measure.code(), 0.0],
            tree: [depth, counts_words(depth), first, room],
            frame: [(self.frames % 2) as u32, slot as u32, 0, 0],
            extent: {
                let extent = terrain.pyramid.extent();
                [0.0, 0.0, extent.x, extent.y]
            },
            cutout: terrain
                .cutout
                .map_or([0.0; 4], |(low, high)| [low.x, low.y, high.x, high.y]),
            planes,
        };
        let at = slot * SLOT as usize;
        bytes[at..at + size_of::<SubdivisionUniform>()]
            .copy_from_slice(bytemuck::bytes_of(&uniform));
        self.frame.push(Some(Subdivided {
            id: terrain.id,
            measure,
            terrain: index,
            first,
            finest: finest_spacing(side, depth),
        }));
    }

    fn tree(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        (terrain, measure): (&RenderTerrain, Measure),
        (wanted, capacity): (u32, u32),
    ) -> (f32, u32) {
        let limits = device.limits();
        let room = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let wanted = wanted.max(SHALLOWEST);
        let mut depth = wanted;
        while depth > SHALLOWEST && tree_bytes(depth) > room {
            depth -= 1;
        }
        self.shallow += u32::from(depth < wanted && measure == Measure::Ground);
        let pyramid = &terrain.pyramid;
        let [tiles_x, tiles_z] = pyramid.tiles(0);
        let side = pyramid.tile_size(0) * tiles_x.max(tiles_z).next_power_of_two() as f32;
        match self.trees.get_mut(&(terrain.id, measure)) {
            Some(tree) if tree.depth == depth && tree.side == side && tree.room <= capacity => {
                tree.room = capacity;
            }
            _ => {
                let buffer = engine_render::buffers::storage(
                    device,
                    "engine/terrain-subdivision-tree",
                    tree_bytes(depth),
                    wgpu::BufferUsages::empty(),
                );
                seed(queue, &buffer, depth);
                self.trees.insert(
                    (terrain.id, measure),
                    Tree {
                        buffer: Versioned::new(buffer),
                        depth,
                        side,
                        room: capacity,
                        group: GroupCache::default(),
                    },
                );
            }
        }
        (side, depth)
    }

    pub(super) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        camera: &wgpu::Buffer,
        terrain_group: &wgpu::BindGroup,
        profiler: &engine_render::profiling::GpuProfiler,
    ) {
        if !self.any() {
            return;
        }
        self.bind(device, camera);
        let (Some(pipelines), Some(camera_group)) = (&self.pipelines, &self.camera_group) else {
            return;
        };
        let (frame, trees) = (&self.frame, &self.trees);
        let bound = || {
            frame.iter().enumerate().filter_map(|(slot, subdivided)| {
                let subdivided = (*subdivided)?;
                let tree = trees.get(&(subdivided.id, subdivided.measure))?;
                Some((
                    (subdivided.terrain as u64 * SLOT) as u32,
                    (slot as u64 * SLOT) as u32,
                    slot as u64 * ARGS_BYTES,
                    tree.group.get()?,
                ))
            })
        };
        let bytes = frame.len() as u64 * ARGS_BYTES;
        encoder.clear_buffer(&self.counters, 0, Some(bytes));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/terrain-subdivision-arguments"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines[0]);
            pass.set_bind_group(0, camera_group, &[]);
            for (terrain, slot, _, group) in bound() {
                pass.set_bind_group(1, terrain_group, &[0, terrain]);
                pass.set_bind_group(2, group, &[slot]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        encoder.copy_buffer_to_buffer(&self.counters, 0, &self.args, 0, bytes);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/terrain-subdivision"),
                timestamp_writes: profiler
                    .span("terrain_subdivision")
                    .map(|span| span.compute()),
            });
            pass.set_bind_group(0, camera_group, &[]);
            for (terrain, slot, args, group) in bound() {
                pass.set_bind_group(1, terrain_group, &[0, terrain]);
                pass.set_bind_group(2, group, &[slot]);
                pass.set_pipeline(&pipelines[1]);
                pass.dispatch_workgroups_indirect(&self.args, args);
                pass.set_pipeline(&pipelines[2]);
                pass.dispatch_workgroups_indirect(&self.args, args);
            }
        }
        let tag = (self.grounds as u32, frame.len() as u32, self.rests.get());
        self.readbacks
            .record(device, encoder, &self.counters, bytes, tag);
    }

    fn bind(&mut self, device: &wgpu::Device, camera: &wgpu::Buffer) {
        let (camera_layout, tree_layout) = (&self.camera_layout, &self.tree_layout);
        self.pipelines.get_or_insert_with(|| {
            let layout = engine_render::pipelines::layout_of(
                device,
                &crate::pipeline_layouts::TERRAIN_SUBDIVISION_PIPELINES,
            );
            [
                draw::subdivide_begin,
                draw::subdivide_decide,
                draw::subdivide_apply,
            ]
            .map(|entry| engine_render::pipelines::compute_entry(device, &layout, entry))
        });
        self.camera_group.get_or_insert_with(|| {
            engine_render::pipelines::group(
                device,
                "engine/terrain-subdivision-camera",
                &TERRAIN_SUBDIVISION_CAMERA,
                camera_layout,
                &[engine_render::buffers::entry("camera", camera)],
            )
        });
        let uniform = Bound::of(
            &self.uniforms,
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.uniforms,
                offset: 0,
                size: wgpu::BufferSize::new(size_of::<SubdivisionUniform>() as u64),
            }),
        );
        for subdivided in self.frame.iter().flatten() {
            let Some(tree) = self.trees.get_mut(&(subdivided.id, subdivided.measure)) else {
                continue;
            };
            tree.group.bind(
                device,
                "engine/terrain-subdivision-group",
                (&TERRAIN_SUBDIVISION, tree_layout),
                [
                    ("subdivision", uniform.clone()),
                    ("subdivision_tree", tree.buffer.bound()),
                    ("subdivision_nodes", self.nodes.bound()),
                    ("subdivision_leaves", self.leaves.bound()),
                    ("subdivision_args", self.counters.bound()),
                ],
            );
        }
    }

    pub(crate) fn after_submit(&self) {
        let (totals, rests) = (Rc::clone(&self.totals), Rc::clone(&self.rests));
        self.readbacks
            .after_submit(move |words, (grounds, slots, rested)| {
                if rested != rests.get() {
                    return;
                }
                let mut sums = [0_u32; 4];
                for slot in 0..slots as usize {
                    let at = slot * SUBDIVISION_ARGS as usize;
                    let sum = if slot < grounds as usize { 0 } else { 2 };
                    let leaves = words.get(at + SUBDIVISION_LEAVES as usize);
                    let refused = words.get(at + SUBDIVISION_REFUSED as usize);
                    sums[sum] = sums[sum].saturating_add(leaves);
                    sums[sum + 1] = sums[sum + 1].saturating_add(refused);
                }
                totals.set(sums);
            });
    }

    pub(crate) fn stats(&self) -> SurfaceStats {
        let [leaves, refused, _, _] = self.totals.get();
        SurfaceStats {
            leaves,
            triangles: leaves.saturating_mul(SURFACE_MESHLET_INDICES / 3),
            refused,
            shallow: self.shallow,
        }
    }

    pub(super) fn water_stats(&self) -> [u32; 2] {
        let [_, _, leaves, refused] = self.totals.get();
        [leaves, refused]
    }
}

fn growable(device: &wgpu::Device, (label, bytes, usage): Spec) -> Growable {
    Growable::new(device, label, bytes, usage)
}

fn subdivision_uniforms(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
    engine_render::buffers::uniform(device, "engine/terrain-subdivision", bytes)
}

fn counts_words(depth: u32) -> u32 {
    1 << (depth + 1 - SUBDIVISION_COUNTED)
}

fn tree_bytes(depth: u32) -> u64 {
    (u64::from(counts_words(depth)) + (1_u64 << (depth - 5))) * 4
}

fn seed(queue: &engine_render::gpu_calls::Queue, buffer: &wgpu::Buffer, depth: u32) {
    let word = |at: u64, value: u32| queue.write_buffer(buffer, at * 4, &value.to_le_bytes());
    word(1, 2);
    for level in 1..=depth - SUBDIVISION_COUNTED {
        word(1 << level, 1);
        word((1 << level) + (1 << (level - 1)), 1);
    }
    let bitfield = u64::from(counts_words(depth));
    word(bitfield, 1);
    word(bitfield + (1 << (depth - 6)), 1);
}

fn finest_spacing(side: f32, depth: u32) -> f32 {
    let halvings = (depth - 1) / 2;
    let leg = side / (1_u64 << halvings) as f32;
    let leg = if (depth - 1).is_multiple_of(2) {
        leg
    } else {
        leg * std::f32::consts::FRAC_1_SQRT_2
    };
    leg / (1 << SURFACE_MESHLET_LEVEL) as f32
}

fn meshlet_indices() -> Vec<u16> {
    let parts = 1_u32 << SURFACE_MESHLET_LEVEL;
    (0..SURFACE_MESHLET_INDICES / 3)
        .flat_map(|piece| {
            let mut corners = [[parts, 0], [0, parts], [0, 0]];
            for bit in (0..2 * SURFACE_MESHLET_LEVEL).rev() {
                let [a, b, c] = corners;
                let middle = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
                corners = if ((piece >> bit) & 1) == 0 {
                    [c, a, middle]
                } else {
                    [b, c, middle]
                };
            }
            corners.map(|[i, j]| lattice_vertex(i, j))
        })
        .collect()
}

fn lattice_vertex(i: u32, j: u32) -> u16 {
    let row = i + j;
    (row * (row + 1) / 2 + j) as u16
}

fn world_planes(culling: Mat4, eye: Vec3) -> [[f32; 4]; 6] {
    let mut world = [[0.0; 4]; 6];
    for (plane, held) in planes(culling, false).iter().zip(&mut world) {
        let normal = plane.truncate().as_dvec3();
        let offset = f64::from(plane.w) - normal.dot(eye.as_dvec3());
        *held = [plane.x, plane.y, plane.z, offset as f32];
    }
    world
}
