//! Each terrain's subdivisions: the concurrent binary tree over its square
//! whose leaves the camera's ground draws (`meshes/terrain_subdivision.wgsl`,
//! `meshes/terrain_surface.wgsl`), and, where its tiles hold water, a second
//! tree over the same square, measured by the water's mesh, whose leaves its
//! water draws (`crate::water`); kept on the GPU from frame to frame,
//! and what a frame binds and dispatches for them: each tree's uniform, the
//! leaves the frame lists for every tree and their nodes, and every tree's
//! counters, read back for the frame's stats, with the indirect arguments
//! copied from them.
//!
//! A tree is a buffer of its counts, then its bitfield, sized by its depth:
//! made when it first subdivides, or anew when its square or depth change
//! or its room shrinks, holding the square's two halves as leaves, which
//! split from there over the frames that follow. Its depth is the
//! settings', or less where the device's storage bindings cannot hold the
//! tree (counted). Each frame the ground's trees take the first slots, each
//! frame terrain's at its index, and the water's trees the slots after them;
//! a tree's uniform, counters and arguments sit at its slot. Each tree
//! takes as many leaves as its settings give one, or fewer where the
//! leaves' binding cannot hold every tree's, from where the ones before it
//! end.
//!
//! A frame's subdivision is two passes: the first writes each tree's
//! arguments among its counters, which the frame then copies to the buffer
//! the second pass's dispatches and the draws take them from, since no
//! dispatch may take its arguments from a buffer it binds to write.
//!
//! Each tree's leaves draw in one indexed indirect draw, an instance a
//! leaf, through one index buffer made once: the meshlet's pieces, each
//! corner naming the point of its lattice it stands at
//! (`meshlet_indices`), so each point's vertex runs once.

use std::cell::Cell;
use std::rc::Rc;

use super::*;
use crate::bindings::{TERRAIN_SUBDIVISION, TERRAIN_SUBDIVISION_CAMERA};
use crate::buffers::{Growable, ReadbackRing};
use crate::cached::{Bound, GroupCache, Versioned};
use crate::gpu_layouts::constants::{
    SUBDIVISION_ARGS, SUBDIVISION_COUNTED, SUBDIVISION_DRAW, SUBDIVISION_LEAVES,
    SUBDIVISION_REFUSED, SURFACE_MESHLET_INDICES, SURFACE_MESHLET_LEVEL, SURFACE_MESHLET_VERTICES,
};
use crate::gpu_layouts::terrain::SubdivisionUniform;
use crate::shaders::entry::meshes;

/// Bytes of a leaf the frame draws (`SurfaceTriangle`): three corners.
const LEAF_BYTES: u64 = 24;
/// Bytes of a leaf's node and decision.
const NODE_BYTES: u64 = 4;
/// Bytes of a tree's counters and indirect arguments, and where its draw's
/// start in them.
const ARGS_BYTES: u64 = SUBDIVISION_ARGS as u64 * 4;
const DRAW_OFFSET: u64 = SUBDIVISION_DRAW as u64 * 4;
/// The shallowest tree a device's bindings are asked to hold: its counts
/// start two levels under the root.
const SHALLOWEST: u32 = SUBDIVISION_COUNTED + 2;
/// The format of the meshlet's indices (`meshlet_indices`), which name
/// every point of its lattice.
const INDEX_FORMAT: wgpu::IndexFormat = wgpu::IndexFormat::Uint16;
const _: () = assert!(
    SURFACE_MESHLET_VERTICES <= 1 << 16,
    "16-bit indices name every point of the meshlet's lattice"
);

/// A tree's draw of the frame's leaves: an instance a leaf, each the
/// meshlet's `SURFACE_MESHLET_INDICES` indices into its lattice
/// (`meshlet_indices`), by the tree's indexed indirect arguments
/// (`wgpu::util::DrawIndexedIndirectArgs`, which `subdivide_begin` writes)
/// at their byte offset.
#[derive(Clone)]
pub(crate) struct SurfaceDraw {
    indices: wgpu::Buffer,
    args: wgpu::Buffer,
    offset: u64,
}

impl SurfaceDraw {
    /// Records the draw into `pass`, through the pipeline and groups set
    /// there.
    pub(crate) fn record(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_index_buffer(self.indices.slice(..), INDEX_FORMAT);
        pass.draw_indexed_indirect(&self.args, self.offset);
    }
}

/// What a tree's leaves follow, which picks its measure
/// (`TerrainSubdivision::square.z`): a terrain's ground, or the water its
/// tiles hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Measure {
    Ground,
    Water,
}

impl Measure {
    /// `TerrainSubdivision::square.z` of its trees.
    fn code(self) -> f32 {
        match self {
            Self::Ground => 0.0,
            Self::Water => 1.0,
        }
    }
}

/// One tree: its buffer of counts and bitfield, its deepest level, its
/// square's side, the most leaves its room has held since it was made,
/// which it may hold, and the group binding it with the frame's shared
/// buffers.
struct Tree {
    buffer: Versioned<wgpu::Buffer>,
    depth: u32,
    side: f32,
    room: u32,
    group: GroupCache<5>,
}

/// A frame tree's subdivision: its terrain and what its leaves follow, the
/// terrain's index among the frame's, where its leaves start among the
/// frame's, and the meters between its finest leaves' meshlet vertices.
#[derive(Clone, Copy)]
struct Subdivided {
    id: u64,
    measure: Measure,
    terrain: usize,
    first: u32,
    finest: f32,
}

/// What the ground's subdivisions drew as last read back: the leaves of
/// every terrain the camera drew and their triangles, and the splits their
/// room refused; and the frame's terrains whose trees the device's storage
/// bindings keep shallower than the settings ask.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SurfaceStats {
    pub leaves: u32,
    pub triangles: u32,
    pub refused: u32,
    pub shallow: u32,
}

pub(crate) struct SurfaceTrees {
    trees: HashMap<(u64, Measure), Tree>,
    // Each frame tree's subdivision, by slot: each frame terrain's ground at
    // its index, none for those not drawing, then the water's trees; and the
    // ground's slots, one for each frame terrain.
    frame: Vec<Option<Subdivided>>,
    grounds: usize,
    // The pixels the ground's and the water's leaves may stray from their
    // surfaces this frame.
    targets: [f32; 2],
    // The frame's leaves and nodes, every tree's; their counters, which the
    // subdivision's passes write, and the indirect arguments copied from
    // them; and the meshlet's indices, every tree's draws'.
    leaves: Growable,
    nodes: Growable,
    counters: Growable,
    args: Growable,
    indices: wgpu::Buffer,
    // Each frame tree's uniform, a slot each, and the bytes written.
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
    // Read-backs of the counters, each of its slots, the ground's first of
    // them, and the rests before it; the leaves and refused splits the last
    // one summed, the ground's trees' then the water's; and the rests,
    // which leave read-backs recorded before them unsummed.
    readbacks: ReadbackRing<(u32, u32, u64)>,
    totals: Rc<Cell<[u32; 4]>>,
    rests: Rc<Cell<u64>>,
    shallow: u32,
}

impl SurfaceTrees {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            trees: HashMap::default(),
            frame: Vec::new(),
            grounds: 0,
            targets: [0.0; 2],
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
            indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("engine/terrain-surface-indices"),
                contents: bytemuck::cast_slice(&meshlet_indices()),
                usage: wgpu::BufferUsages::INDEX,
            }),
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
            camera_layout: crate::pipelines::layout(device, &TERRAIN_SUBDIVISION_CAMERA),
            tree_layout: crate::pipelines::layout(device, &TERRAIN_SUBDIVISION),
            camera_group: None,
            pipelines: None,
            frames: 0,
            readbacks: ReadbackRing::new("engine/terrain-surface-readback"),
            totals: Rc::new(Cell::new([0; 4])),
            rests: Rc::new(Cell::new(0)),
            shallow: 0,
        }
    }

    /// The leaves the frame draws, every tree's, which the terrain group and
    /// water's group bind.
    pub(super) fn leaves(&self) -> &Versioned<wgpu::Buffer> {
        &self.leaves
    }

    /// Each frame tree's uniform, a slot each, which water's draws bind at
    /// its trees' slots (`Self::water`) for where their leaves start.
    pub(super) fn uniforms(&self) -> &Versioned<wgpu::Buffer> {
        &self.uniforms
    }

    /// Bytes of the trees, of the frame's leaves, nodes, counters and
    /// arguments, and of the meshlet's indices.
    pub(super) fn capacity_bytes(&self) -> u64 {
        let trees: u64 = self.trees.values().map(|tree| tree.buffer.size()).sum();
        let frame = self.leaves.size() + self.nodes.size() + self.counters.size();
        trees + frame + self.args.size() + self.indices.size()
    }

    /// The subdivision of frame terrain `index`'s ground this frame, where
    /// it draws.
    fn ground(&self, index: usize) -> Option<Subdivided> {
        (index < self.grounds)
            .then(|| self.frame.get(index).copied().flatten())
            .flatten()
    }

    /// Where frame terrain `index`'s ground's leaves start among the
    /// frame's, and the meters between its finest leaves' meshlet vertices;
    /// zeroes where it does not draw.
    pub(super) fn of(&self, index: usize) -> (u32, f32) {
        self.ground(index)
            .map_or((0, 0.0), |subdivided| (subdivided.first, subdivided.finest))
    }

    /// Whether any frame tree subdivides.
    pub(super) fn any(&self) -> bool {
        self.frame.iter().any(Option::is_some)
    }

    /// Subdivides nothing this frame, as no terrain draws: its stats read
    /// none drawn.
    pub(super) fn rest(&mut self) {
        self.frame.clear();
        self.grounds = 0;
        self.shallow = 0;
        self.totals.set([0; 4]);
        self.rests.set(self.rests.get() + 1);
    }

    /// The draw of frame terrain `index`'s ground's leaves, once it
    /// subdivides this frame.
    pub(super) fn draw(&self, index: usize) -> Option<SurfaceDraw> {
        self.ground(index).map(|_| self.slot_draw(index))
    }

    /// The tree of frame terrain `index`'s water this frame: its uniform's
    /// offset among the trees' (`Self::uniforms`), and the draw of its
    /// leaves; `None` where its water does not subdivide.
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

    /// The draw of the leaves of the frame's tree at slot `slot`.
    fn slot_draw(&self, slot: usize) -> SurfaceDraw {
        SurfaceDraw {
            indices: self.indices.clone(),
            args: wgpu::Buffer::clone(&self.args),
            offset: slot as u64 * ARGS_BYTES + DRAW_OFFSET,
        }
    }

    /// The pixels the ground's leaves (x) and the water's (y) may stray from
    /// their surfaces seen from the detail eye, as this frame's trees take
    /// them.
    pub(super) fn targets(&self) -> [f32; 2] {
        self.targets
    }

    /// Lays out this frame's subdivisions of `terrains` drawing, as `views`
    /// see them: each one's ground's tree, and its water's where water's
    /// settings draw it and its tiles hold water (`holds_water`, by
    /// terrain), each made or made anew where its square or depth changed or
    /// its room shrank, with its leaves' room and its uniform, written where
    /// it changed.
    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        views: &TerrainViews<'_>,
        holds_water: impl Fn(u64) -> bool,
    ) {
        self.frames += 1;
        self.shallow = 0;
        self.frame.clear();
        let (settings, water) = (&views.settings, &views.water);
        let subdivides_water = |terrain: &RenderTerrain| water.enabled && holds_water(terrain.id);
        // A terrain's ground's tree stays while it does, its water's while
        // its tiles hold water.
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
        // Every tree's leaves within the leaves' binding.
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
        // Read-backs hold the counters of as many trees as were subdividing
        // when they were made.
        if self.uniforms_written.len() as u64 != slots * SLOT {
            self.readbacks.clear();
        }
        if slots * SLOT > self.uniforms.size() {
            self.uniforms.replace(crate::buffers::uniform(
                device,
                "engine/terrain-subdivision",
                (slots * SLOT).next_power_of_two(),
            ));
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

    /// Lays out the frame's next tree at the next slot: frame terrain
    /// `index`'s, `terrain`'s, following `measure`, of deepest level
    /// `depth` or as deep as the device's bindings hold, its `room` leaves
    /// from `first` on among the frame's; its uniform written into `bytes`
    /// at its slot, culling through the camera's frustum `planes`.
    fn lay_out(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
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

    /// The side of `terrain`'s square and the deepest level of its tree
    /// following `measure`, `wanted` or as deep as the device's storage
    /// bindings hold (counted for its ground's, which its water's is as
    /// deep as), for a room of `capacity` leaves: the tree made with the
    /// square's two halves as its leaves where there is none, or anew where
    /// its square or depth changed or the room shrank below the leaves it
    /// may hold.
    fn tree(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
                let buffer = crate::buffers::storage(
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

    /// Records this frame's subdivisions: each tree's arguments, then its
    /// leaves listed and its decisions applied, through the camera's uniform
    /// `camera` and the terrain group standing in for the leaves
    /// (`terrain_group`, with the terrain group's layout), timed by
    /// `profiler`; and a copy of every tree's counters to read back.
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
        // Each frame tree subdividing: its terrain's uniform's offset, its
        // own uniform's, its arguments' and its group.
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

    /// Makes the passes' pipelines and the camera's group, through the
    /// camera's uniform `camera` and the terrain group's layout
    /// `terrain_layout`, unless they exist; and binds every frame tree with
    /// the frame's shared buffers.
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
                &TERRAIN_SUBDIVISION_CAMERA,
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

    /// Requests the map of this frame's counters once it is submitted.
    pub(crate) fn after_submit(&self) {
        let (totals, rests) = (Rc::clone(&self.totals), Rc::clone(&self.rests));
        self.readbacks
            .after_submit(move |words, (grounds, slots, rested)| {
                if rested != rests.get() {
                    return;
                }
                // The ground's trees' leaves and refused splits, then the
                // water's.
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

    /// What the ground's subdivisions drew, as last read back.
    pub(crate) fn stats(&self) -> SurfaceStats {
        let [leaves, refused, _, _] = self.totals.get();
        SurfaceStats {
            leaves,
            triangles: leaves.saturating_mul(SURFACE_MESHLET_INDICES / 3),
            refused,
            shallow: self.shallow,
        }
    }

    /// The leaves the water's subdivisions drew and the splits their room
    /// refused, as last read back.
    pub(super) fn water_stats(&self) -> [u32; 2] {
        let [_, _, leaves, refused] = self.totals.get();
        [leaves, refused]
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

/// The meshlet's indices: three for each of its pieces, in the order and
/// turning the bisections cutting a leaf into them give
/// (`meshes/terrain_surface.wgsl`: the first from the leaf's first
/// bisection on, each piece's corners as `surface_half` turns them), each
/// the point of its lattice its corner stands at (`lattice_vertex`). The
/// bisections are replayed on the lattice's own coordinates, a leg's parts
/// from the leaf's right angle toward its longest edge's first end and
/// toward its last, where every middle they take is whole.
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

/// The number of the point of the meshlet's lattice `i` of a leaf's legs'
/// parts from its right angle toward its longest edge's first end and `j`
/// toward its last: rows of `i + j` from the right angle out, each a point
/// longer than the one before, and by `j` along each, as
/// `meshes/terrain_surface.wgsl`'s `surface_vertex` reads it back.
fn lattice_vertex(i: u32, j: u32) -> u16 {
    let row = i + j;
    (row * (row + 1) / 2 + j) as u16
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
