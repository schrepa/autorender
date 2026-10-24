//! Each terrain's subdivision: the concurrent binary tree over its square
//! whose leaves the camera's ground draws (`meshes/terrain_subdivision.wgsl`,
//! `meshes/terrain_surface.wgsl`), kept on the GPU from frame to frame, and
//! what a frame binds and dispatches for it: each terrain's uniform and
//! tree, the leaves the frame lists for every terrain and their nodes, and
//! every terrain's counters, read back for the frame's stats, with the
//! indirect arguments copied from them.
//!
//! A terrain's tree is a buffer of its counts, then its bitfield, sized by
//! its depth: made when the terrain first draws, or anew when its square or
//! depth change or its room shrinks, holding the square's two halves as
//! leaves, which split from there over the frames that follow. Its depth is
//! the settings', or less where the device's storage bindings cannot hold
//! the tree (counted). Each frame terrain takes as many leaves as the
//! settings give one, or fewer where the leaves' binding cannot hold them
//! all, from where the ones before it end; its uniform, counters and
//! arguments sit at its index.
//!
//! A frame's subdivision is two passes: the first writes each terrain's
//! arguments among its counters, which the frame then copies to the buffer
//! the second pass's dispatches and the draws take them from, since no
//! dispatch may take its arguments from a buffer it binds to write.

use std::cell::Cell;
use std::rc::Rc;

use super::*;
use crate::bindings::{TERRAIN_SUBDIVISION, TERRAIN_VT_CAMERA};
use crate::buffers::{Growable, ReadbackRing};
use crate::cached::{Bound, GroupCache, Versioned};
use crate::gpu_layouts::constants::{
    SUBDIVISION_ARGS, SUBDIVISION_COUNTED, SURFACE_MESHLET_LEVEL, SURFACE_MESHLET_VERTICES,
};
use crate::gpu_layouts::terrain::SubdivisionUniform;
use crate::shaders::entry::meshes;

/// Bytes of a leaf the frame draws (`SurfaceTriangle`): three corners.
const LEAF_BYTES: u64 = 24;
/// Bytes of a leaf's node and decision.
const NODE_BYTES: u64 = 4;
/// Bytes of a terrain's counters and indirect arguments, and where its
/// draw's start in them.
const ARGS_BYTES: u64 = SUBDIVISION_ARGS as u64 * 4;
const DRAW_OFFSET: u64 = 16;
/// The shallowest tree a device's bindings are asked to hold: its counts
/// start two levels under the root.
const SHALLOWEST: u32 = SUBDIVISION_COUNTED + 2;

/// One terrain's tree: its buffer of counts and bitfield, its deepest
/// level, its square's side, the most leaves its room has held since it was
/// made, which it may hold, and the group binding it with the frame's
/// shared buffers.
struct Tree {
    buffer: Versioned<wgpu::Buffer>,
    depth: u32,
    side: f32,
    room: u32,
    group: GroupCache<5>,
}

/// A frame terrain's subdivision: its tree's terrain, where its leaves
/// start among the frame's, and the meters between its finest leaves'
/// meshlet vertices.
#[derive(Clone, Copy)]
struct Subdivided {
    id: u64,
    first: u32,
    finest: f32,
}

/// What the subdivisions drew as last read back: the leaves of every
/// terrain the camera drew and their triangles, and the splits their room
/// refused; and the frame's terrains whose trees the device's storage
/// bindings keep shallower than the settings ask.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SurfaceStats {
    pub leaves: u32,
    pub triangles: u32,
    pub refused: u32,
    pub shallow: u32,
}

pub(crate) struct SurfaceTrees {
    trees: HashMap<u64, Tree>,
    // Each frame terrain's subdivision, by index; none for those not
    // drawing.
    frame: Vec<Option<Subdivided>>,
    // The frame's leaves and nodes, every terrain's; their counters, which
    // the subdivision's passes write, and the indirect arguments copied from
    // them.
    leaves: Growable,
    nodes: Growable,
    counters: Growable,
    args: Growable,
    // Each frame terrain's uniform, a slot each, and the bytes written.
    uniforms: Versioned<wgpu::Buffer>,
    uniforms_written: Vec<u8>,
    /// What the terrain group binds for the leaves in the subdivision's own
    /// passes, which write them.
    pub(super) stand_in: wgpu::Buffer,
    camera_layout: crate::pipelines::GroupLayout,
    tree_layout: crate::pipelines::GroupLayout,
    camera_group: Option<wgpu::BindGroup>,
    pipelines: Option<[wgpu::ComputePipeline; 3]>,
    // Frames subdivided, whose parity picks splitting or merging.
    frames: u64,
    // Read-backs of the counters, each of its terrains and the rests before
    // it; the leaves and refused splits the last one summed; and the rests,
    // which leave read-backs recorded before them unsummed.
    readbacks: ReadbackRing<(u32, u64)>,
    totals: Rc<Cell<[u32; 2]>>,
    rests: Rc<Cell<u64>>,
    shallow: u32,
}

impl SurfaceTrees {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            trees: HashMap::default(),
            frame: Vec::new(),
            leaves: Growable::new(
                device,
                "engine/terrain-surface-leaves",
                LEAF_BYTES,
                wgpu::BufferUsages::empty(),
            ),
            nodes: Growable::new(
                device,
                "engine/terrain-surface-nodes",
                NODE_BYTES,
                wgpu::BufferUsages::empty(),
            ),
            counters: Growable::new(
                device,
                "engine/terrain-surface-counters",
                ARGS_BYTES,
                wgpu::BufferUsages::COPY_SRC,
            ),
            args: Growable::new(
                device,
                "engine/terrain-surface-args",
                ARGS_BYTES,
                wgpu::BufferUsages::INDIRECT,
            ),
            uniforms: Versioned::new(crate::buffers::uniform(
                device,
                "engine/terrain-subdivision",
                SLOT,
            )),
            uniforms_written: Vec::new(),
            stand_in: crate::buffers::storage(
                device,
                "engine/terrain-surface-stand-in",
                LEAF_BYTES,
                wgpu::BufferUsages::empty(),
            ),
            camera_layout: crate::pipelines::layout(device, &TERRAIN_VT_CAMERA),
            tree_layout: crate::pipelines::layout(device, &TERRAIN_SUBDIVISION),
            camera_group: None,
            pipelines: None,
            frames: 0,
            readbacks: ReadbackRing::new("engine/terrain-surface-readback"),
            totals: Rc::new(Cell::new([0; 2])),
            rests: Rc::new(Cell::new(0)),
            shallow: 0,
        }
    }

    /// The leaves the frame draws, every terrain's, which the terrain group
    /// binds.
    pub(super) fn leaves(&self) -> &Versioned<wgpu::Buffer> {
        &self.leaves
    }

    /// Bytes of the trees and of the frame's leaves, nodes, counters and
    /// arguments.
    pub(super) fn capacity_bytes(&self) -> u64 {
        let trees: u64 = self.trees.values().map(|tree| tree.buffer.size()).sum();
        trees + self.leaves.size() + self.nodes.size() + self.counters.size() + self.args.size()
    }

    /// Where frame terrain `index`'s leaves start among the frame's, and the
    /// meters between its finest leaves' meshlet vertices; zeroes where it
    /// does not draw.
    pub(super) fn of(&self, index: usize) -> (u32, f32) {
        self.frame
            .get(index)
            .copied()
            .flatten()
            .map_or((0, 0.0), |subdivided| (subdivided.first, subdivided.finest))
    }

    /// Whether any frame terrain subdivides.
    pub(super) fn any(&self) -> bool {
        self.frame.iter().any(Option::is_some)
    }

    /// Subdivides nothing this frame, as no terrain draws: its stats read
    /// none drawn.
    pub(super) fn rest(&mut self) {
        self.frame.clear();
        self.shallow = 0;
        self.totals.set([0; 2]);
        self.rests.set(self.rests.get() + 1);
    }

    /// The indirect draw arguments of frame terrain `index`'s leaves and
    /// their byte offset, once it subdivides this frame.
    pub(super) fn draw(&self, index: usize) -> Option<(&wgpu::Buffer, u64)> {
        self.frame
            .get(index)
            .copied()
            .flatten()
            .map(|_| (&**self.args, index as u64 * ARGS_BYTES + DRAW_OFFSET))
    }

    /// Lays out this frame's subdivisions of `terrains` drawing, as `views`
    /// see them: each one's tree (made or made anew where its square or
    /// depth changed or its room shrank), its leaves' room and its uniform,
    /// written where it changed.
    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        views: &TerrainViews<'_>,
    ) {
        self.frames += 1;
        self.shallow = 0;
        self.frame.clear();
        self.trees
            .retain(|id, _| terrains.terrains.iter().any(|terrain| terrain.id == *id));
        let count = terrains.terrains.len() as u64;
        let settings = &views.settings;
        // Every terrain's leaves within the leaves' binding.
        let room = self.leaves.room() / LEAF_BYTES / count.max(1);
        let capacity = room.min(u64::from(settings.subdivision_leaves)) as u32;
        let leaves = count * u64::from(capacity);
        self.leaves.grow(device, leaves * LEAF_BYTES);
        self.nodes.grow(device, leaves * NODE_BYTES);
        self.counters.grow(device, count * ARGS_BYTES);
        self.args.grow(device, count * ARGS_BYTES);
        // Read-backs hold the counters of as many terrains as were drawing
        // when they were made.
        if self.uniforms_written.len() as u64 != count * SLOT {
            self.readbacks.clear();
        }
        if count * SLOT > self.uniforms.size() {
            self.uniforms.replace(crate::buffers::uniform(
                device,
                "engine/terrain-subdivision",
                (count * SLOT).next_power_of_two(),
            ));
            self.uniforms_written.clear();
        }
        let planes = world_planes(views.culling, views.eye);
        let mut bytes = vec![0_u8; (count * SLOT) as usize];
        for (index, terrain) in terrains.terrains.iter().enumerate() {
            if !terrain.drawn {
                self.frame.push(None);
                continue;
            }
            let wanted = (settings.subdivision_depth, capacity);
            let (side, depth) = self.tree(device, queue, terrain, wanted);
            let first = index as u32 * capacity;
            let uniform = SubdivisionUniform {
                square: [side, settings.subdivision_error, 0.0, 0.0],
                tree: [depth, counts_words(depth), first, capacity],
                frame: [(self.frames % 2) as u32, index as u32, 0, 0],
                extent: {
                    let extent = terrain.pyramid.extent();
                    [0.0, 0.0, extent.x, extent.y]
                },
                cutout: terrain
                    .cutout
                    .map_or([0.0; 4], |(low, high)| [low.x, low.y, high.x, high.y]),
                planes,
            };
            let at = index * SLOT as usize;
            bytes[at..at + size_of::<SubdivisionUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&uniform));
            self.frame.push(Some(Subdivided {
                id: terrain.id,
                first,
                finest: finest_spacing(side, depth),
            }));
        }
        if bytes != self.uniforms_written {
            queue.write_buffer(&self.uniforms, 0, &bytes);
            self.uniforms_written = bytes;
        }
    }

    /// The side of `terrain`'s square and its tree's deepest level, `wanted`
    /// or as deep as the device's storage bindings hold (counted), for a
    /// room of `capacity` leaves: its tree made with the square's two halves
    /// as its leaves where it has none, or anew where its square or depth
    /// changed or the room shrank below the leaves it may hold.
    fn tree(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrain: &RenderTerrain,
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
        self.shallow += u32::from(depth < wanted);
        let pyramid = &terrain.pyramid;
        let [tiles_x, tiles_z] = pyramid.tiles(0);
        let side = pyramid.tile_size(0) * tiles_x.max(tiles_z).next_power_of_two() as f32;
        match self.trees.get_mut(&terrain.id) {
            Some(tree) if tree.depth == depth && tree.side == side && tree.room <= capacity => {
                tree.room = capacity;
            }
            _ => {
                let buffer = crate::buffers::storage(
                    device,
                    "engine/terrain-subdivision-tree",
                    tree_bytes(depth),
                    wgpu::BufferUsages::empty(),
                );
                seed(queue, &buffer, depth);
                self.trees.insert(
                    terrain.id,
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

    /// Records this frame's subdivisions: each terrain's arguments, then its
    /// leaves listed and its decisions applied, through the camera's uniform
    /// `camera` and the terrain group standing in for the leaves
    /// (`terrain_group`, with the terrain group's layout), timed by
    /// `profiler`; and a copy of every terrain's counters to read back.
    pub(super) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        shader: &wgpu::ShaderModule,
        camera: &wgpu::Buffer,
        (terrain_layout, terrain_group): (&crate::pipelines::GroupLayout, &wgpu::BindGroup),
        profiler: &crate::profiling::GpuProfiler,
    ) {
        if !self.any() {
            return;
        }
        self.bind(device, shader, camera, terrain_layout);
        let (Some(pipelines), Some(camera_group)) = (&self.pipelines, &self.camera_group) else {
            return;
        };
        let (frame, trees) = (&self.frame, &self.trees);
        // Each frame terrain subdividing: its uniforms' offset, its
        // arguments' and its tree's group.
        let bound = || {
            frame.iter().enumerate().filter_map(|(index, subdivided)| {
                let group = trees.get(&(*subdivided)?.id)?.group.get()?;
                Some((
                    (index as u64 * SLOT) as u32,
                    index as u64 * ARGS_BYTES,
                    group,
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
            for (slot, _, group) in bound() {
                pass.set_bind_group(1, terrain_group, &[0, slot]);
                pass.set_bind_group(2, group, &[slot]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        // The arguments the decisions' dispatches and the draws take.
        encoder.copy_buffer_to_buffer(&self.counters, 0, &self.args, 0, bytes);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/terrain-subdivision"),
                timestamp_writes: profiler
                    .span("terrain_subdivision")
                    .map(|span| span.compute()),
            });
            pass.set_bind_group(0, camera_group, &[]);
            for (slot, args, group) in bound() {
                pass.set_bind_group(1, terrain_group, &[0, slot]);
                pass.set_bind_group(2, group, &[slot]);
                pass.set_pipeline(&pipelines[1]);
                pass.dispatch_workgroups_indirect(&self.args, args);
                pass.set_pipeline(&pipelines[2]);
                pass.dispatch_workgroups_indirect(&self.args, args);
            }
        }
        let tag = (frame.len() as u32, self.rests.get());
        self.readbacks
            .record(device, encoder, &self.counters, bytes, tag);
    }

    /// Makes the passes' pipelines and the camera's group, through the
    /// camera's uniform `camera` and the terrain group's layout
    /// `terrain_layout`, unless they exist; and binds every frame terrain's
    /// tree with the frame's shared buffers.
    fn bind(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera: &wgpu::Buffer,
        terrain_layout: &crate::pipelines::GroupLayout,
    ) {
        let (camera_layout, tree_layout) = (&self.camera_layout, &self.tree_layout);
        self.pipelines.get_or_insert_with(|| {
            let layout = crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::TERRAIN_SUBDIVISION_PIPELINES,
                &[camera_layout, terrain_layout, tree_layout],
            );
            [
                meshes::subdivide_begin,
                meshes::subdivide_decide,
                meshes::subdivide_apply,
            ]
            .map(|entry| crate::pipelines::compute_entry(device, &layout, shader, entry))
        });
        self.camera_group.get_or_insert_with(|| {
            crate::pipelines::group(
                device,
                "engine/terrain-subdivision-camera",
                &TERRAIN_VT_CAMERA,
                camera_layout,
                &[crate::buffers::entry("camera", camera)],
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
            let Some(tree) = self.trees.get_mut(&subdivided.id) else {
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

    /// Requests the map of this frame's counters once it is submitted.
    pub(crate) fn after_submit(&self) {
        let (totals, rests) = (Rc::clone(&self.totals), Rc::clone(&self.rests));
        self.readbacks
            .after_submit(move |words, (terrains, rested)| {
                if rested != rests.get() {
                    return;
                }
                let mut sums = [0_u32; 2];
                for terrain in 0..terrains as usize {
                    let at = terrain * SUBDIVISION_ARGS as usize;
                    sums[0] = sums[0].saturating_add(words.get(at + 3));
                    sums[1] = sums[1].saturating_add(words.get(at + 9));
                }
                totals.set(sums);
            });
    }

    /// What the subdivisions drew, as last read back.
    pub(crate) fn stats(&self) -> SurfaceStats {
        let [leaves, refused] = self.totals.get();
        SurfaceStats {
            leaves,
            triangles: leaves.saturating_mul(SURFACE_MESHLET_VERTICES / 3),
            refused,
            shallow: self.shallow,
        }
    }
}

/// Words of a tree of deepest level `depth` before its bitfield: its counts
/// by node, from word 1, of every level down to SUBDIVISION_COUNTED above
/// the deepest.
fn counts_words(depth: u32) -> u32 {
    1 << (depth + 1 - SUBDIVISION_COUNTED)
}

/// Bytes of a tree of deepest level `depth`: its counts, then its bitfield
/// of a bit for each node of that level.
fn tree_bytes(depth: u32) -> u64 {
    (u64::from(counts_words(depth)) + (1_u64 << (depth - 5))) * 4
}

/// Writes the tree of deepest level `depth` in `buffer`, which holds zeroes,
/// with the square's two halves (nodes 2 and 3) as its leaves: their bits,
/// the first of the bitfield and the first of its second half, and a leaf
/// in every count over each, two at the root.
fn seed(queue: &wgpu::Queue, buffer: &wgpu::Buffer, depth: u32) {
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

/// The meters between the meshlet vertices of the deepest leaves of a tree
/// of deepest level `depth` over a square `side` meters wide: their right
/// sides, the square's side halved every two levels, cut into the
/// meshlet's parts.
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

/// The planes of the frustum of `culling`, a projection of positions
/// relative to `eye`, inward in the world: its sides, near and far.
fn world_planes(culling: Mat4, eye: Vec3) -> [[f32; 4]; 6] {
    let mut world = [[0.0; 4]; 6];
    for (plane, held) in planes(culling, false).iter().zip(&mut world) {
        let normal = plane.truncate().as_dvec3();
        let offset = f64::from(plane.w) - normal.dot(eye.as_dvec3());
        *held = [plane.x, plane.y, plane.z, offset as f32];
    }
    world
}
