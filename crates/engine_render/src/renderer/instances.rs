//! Instance storage: retained slots' poses and
//! colors at their core slot positions, the slot IDs CPU-prepared frames
//! draw, joint palettes, and the pose buffer temporal frames
//! alternate with. Storage grows with the frame's slots up to
//! what one binding holds.

use super::*;
use crate::bindings::RETAINED_POSES;
use crate::cached::GroupCache;

pub(super) struct Instances {
    // Most slots this GPU's pose and shadow buffers hold.
    pub(super) limit: usize,
    pub(super) capacity: usize,
    // Slot poses, which upload straight from `RenderFrame::pose_rows`.
    pub(super) buffer: Versioned<wgpu::Buffer>,
    colors: Versioned<wgpu::Buffer>,
    pub(super) color_staging: Vec<ColorData>,
    // Group 2 of retained mesh pipelines, one per pose buffer: its poses,
    // the colors, its palette, and grounds none of the slots stand on.
    pub(super) pose_layout: crate::pipelines::GroupLayout,
    pose_groups: [GroupCache<4>; 2],
    grounds: wgpu::Buffer,
    // Temporal frames cull into this buffer and `buffer` in turn; released
    // (a new generation, so culling stops binding it) when they stop.
    alternate: Versioned<Option<wgpu::Buffer>>,
    // Slot IDs of CPU-prepared retained frames in drawing order, staged for
    // the frame identity and structure revision `slot_revision`.
    slots: Option<wgpu::Buffer>,
    pub(super) slot_capacity: usize,
    pub(super) slot_staging: Vec<u32>,
    pub(super) slot_revision: Option<(u64, u64)>,
    // Joint palettes of skinned slots, one per pose buffer,
    // with room for `palette_capacity` joints.
    pub(super) palettes: [Versioned<wgpu::Buffer>; 2],
    palette_capacity: usize,
    pub(super) palette_staging: Vec<engine_core::PoseRows>,
    pub(super) retained_uploads: RetainedUploads,
    pub(super) pose_uploads: StagingRing,
}

impl Instances {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        // Slots: what the largest per-slot storage (poses, colors, six
        // shadow ID lists) holds within one binding, below the moving flag
        // of visible IDs.
        let binding = device
            .limits()
            .max_storage_buffer_binding_size
            .min(device.limits().max_buffer_size);
        let record = size_of::<PoseData>().max(6 * size_of::<u32>()) as u64;
        let limit = (binding / record).min(u64::from(crate::gpu_data::MOVED_SLOT)) as usize;
        let pose_layout = crate::pipelines::layout(device, &RETAINED_POSES);
        let storage =
            |capacity, stride| Versioned::new(create_storage_buffer(device, capacity, stride));
        let mut instances = Self {
            limit,
            capacity: INITIAL_CAPACITY,
            buffer: storage(INITIAL_CAPACITY, size_of::<PoseData>()),
            colors: storage(INITIAL_CAPACITY, size_of::<ColorData>()),
            color_staging: Vec::new(),
            pose_layout,
            pose_groups: Default::default(),
            grounds: create_storage_buffer(device, 1, size_of::<GroundData>()),
            alternate: Versioned::new(None),
            slots: None,
            slot_capacity: 0,
            slot_staging: Vec::new(),
            slot_revision: None,
            palettes: [0, 1].map(|_| storage(1, size_of::<PoseData>())),
            palette_capacity: 1,
            palette_staging: Vec::new(),
            retained_uploads: RetainedUploads::default(),
            pose_uploads: StagingRing::default(),
        };
        instances.prepare_pose_group(device, 0);
        instances
    }

    /// Sizes storage for a frame of `payload_count` slots and `joints`
    /// palette entries, with the alternate pose buffer of `temporal` frames.
    pub(super) fn reserve(
        &mut self,
        device: &wgpu::Device,
        payload_count: usize,
        joints: usize,
        temporal: bool,
    ) {
        let grew = payload_count > self.capacity;
        if grew {
            self.capacity = payload_count.next_power_of_two().min(self.limit);
            engine_profile::counter("render.instance_buffer_growth", 1.0);
            self.buffer.replace(create_storage_buffer(
                device,
                self.capacity,
                size_of::<PoseData>(),
            ));
            self.colors.replace(create_storage_buffer(
                device,
                self.capacity,
                size_of::<ColorData>(),
            ));
            self.retained_uploads.invalidate();
            self.alternate.replace(None);
        }
        // Joint palettes grow with the frame's joints.
        if joints > self.palette_capacity {
            self.palette_capacity = joints.next_power_of_two();
            for palette in &mut self.palettes {
                palette.replace(create_storage_buffer(
                    device,
                    self.palette_capacity,
                    size_of::<PoseData>(),
                ));
            }
        }
        if temporal && self.alternate.is_none() {
            self.alternate.replace(Some(create_storage_buffer(
                device,
                self.capacity,
                size_of::<PoseData>(),
            )));
        }
        self.prepare_pose_group(device, 0);
    }

    /// Advances whenever pose storage is replaced: either pose buffer, a
    /// palette or the colors. Generations only grow, so their greatest
    /// does.
    pub(super) fn generation(&self) -> u64 {
        self.frame_poses(0).generation()
    }

    /// Stages the slot IDs of a CPU-prepared retained frame drawing
    /// `ordered`, unless already staged for `key` (its identity and
    /// structure revision); returns whether they need writing. Virtual
    /// geometry's selection reads them too.
    pub(super) fn stage_slots(
        &mut self,
        device: &wgpu::Device,
        ordered: &[usize],
        key: Option<(u64, u64)>,
    ) -> bool {
        let count = ordered.len();
        if self.slots.is_none() || count > self.slot_capacity {
            self.slot_capacity = count
                .max(INITIAL_CAPACITY)
                .next_power_of_two()
                .min(self.limit);
            self.slots = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/retained-slots"),
                size: (self.slot_capacity * size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.slot_revision = None;
        }
        if self.slot_revision == key {
            return false;
        }
        self.slot_staging.clear();
        self.slot_staging.reserve_exact(count);
        self.slot_staging
            .extend(ordered.iter().map(|&slot| slot as u32));
        !self.slot_staging.is_empty()
    }

    /// The pose buffer of frames of temporal `parity` (0 without temporal
    /// antialiasing).
    pub(super) fn poses(&self, parity: usize) -> &wgpu::Buffer {
        if parity == 1 {
            Option::as_ref(&self.alternate).expect("temporal frames create it")
        } else {
            &self.buffer
        }
    }

    /// The pose storage of retained frames of temporal `parity`.
    pub(super) fn frame_poses(&self, parity: usize) -> FramePoses<'_> {
        FramePoses {
            first: &self.buffer,
            alternate: &self.alternate,
            palettes: [&self.palettes[0], &self.palettes[1]],
            colors: &self.colors,
            parity,
        }
    }

    /// GPU presentation's inputs for frames of temporal
    /// `parity`: the staging ring of pose uploads, and the pose storage.
    pub(super) fn presentation_inputs(
        &mut self,
        parity: usize,
    ) -> (&mut StagingRing, FramePoses<'_>) {
        let poses = FramePoses {
            first: &self.buffer,
            alternate: &self.alternate,
            palettes: [&self.palettes[0], &self.palettes[1]],
            colors: &self.colors,
            parity,
        };
        (&mut self.pose_uploads, poses)
    }

    /// Both pose buffers as groups bind them: the first again when there is
    /// no alternate one.
    pub(super) fn pose_bounds(&self) -> [Bound<'_>; 2] {
        let poses = self.frame_poses(0);
        let first = poses.bound(0).expect("the first pose buffer exists");
        [first.clone(), poses.bound(1).unwrap_or(first)]
    }

    /// The pose buffer odd temporal frames cull into, once created.
    pub(super) fn alternate(&self) -> Option<&wgpu::Buffer> {
        Option::as_ref(&self.alternate)
    }

    /// The alternate pose buffer's bytes, 0 without one.
    pub(super) fn alternate_bytes(&self) -> u64 {
        self.alternate().map_or(0, wgpu::Buffer::size)
    }

    /// Releases the alternate pose buffer when temporal antialiasing stops:
    /// culling must not keep writing it, so groups binding pose storage
    /// rebuild.
    pub(super) fn release_alternate(&mut self) {
        if self.alternate.is_some() {
            self.alternate.replace(None);
        }
    }

    /// Retained slots' colors.
    pub(super) fn colors(&self) -> &Versioned<wgpu::Buffer> {
        &self.colors
    }

    /// Grounds of retained slots: none stand on terrain.
    pub(super) fn grounds(&self) -> &wgpu::Buffer {
        &self.grounds
    }

    /// Slot IDs of CPU-prepared retained frames.
    pub(super) fn slots(&self) -> &wgpu::Buffer {
        self.slots.as_ref().expect("retained slot buffer")
    }

    /// Group 2 of retained mesh pipelines for frames of temporal `parity`;
    /// odd frames' after [`Self::prepare_pose_group`].
    pub(super) fn pose_group(&self, parity: usize) -> &wgpu::BindGroup {
        self.pose_groups[parity].get().expect("prepared this frame")
    }

    /// The pose group of frames of temporal `parity`: their poses, the
    /// colors, their palette and grounds.
    pub(super) fn prepare_pose_group(&mut self, device: &wgpu::Device, parity: usize) {
        let poses = FramePoses {
            first: &self.buffer,
            alternate: &self.alternate,
            palettes: [&self.palettes[0], &self.palettes[1]],
            colors: &self.colors,
            parity,
        };
        self.pose_groups[parity].bind(
            device,
            "engine/retained-pose-bind-group",
            (&RETAINED_POSES, &self.pose_layout),
            [
                (
                    "poses",
                    poses.bound(parity).expect("temporal frames create it"),
                ),
                ("colors", self.colors.bound()),
                ("palette", self.palettes[parity].bound()),
                ("grounds", Bound::fixed(self.grounds.as_entire_binding())),
            ],
        );
    }
}

/// The pose storage of a retained frame, which GPU presentation culls into and
/// every pass drawing or lighting slots reads: both pose buffers (the second
/// while temporal antialiasing alternates them), the joint palette written with
/// each, the slot colors and which buffer the frame's poses are in.
#[derive(Clone, Copy)]
pub(crate) struct FramePoses<'a> {
    first: &'a Versioned<wgpu::Buffer>,
    alternate: &'a Versioned<Option<wgpu::Buffer>>,
    pub palettes: [&'a Versioned<wgpu::Buffer>; 2],
    pub colors: &'a Versioned<wgpu::Buffer>,
    pub parity: usize,
}

impl<'a> FramePoses<'a> {
    /// Pose buffer `parity`: the second exists while temporal antialiasing
    /// alternates them.
    pub(crate) fn buffer(&self, parity: usize) -> Option<&'a wgpu::Buffer> {
        if parity == 1 {
            Option::as_ref(self.alternate)
        } else {
            Some(self.first)
        }
    }

    /// Pose buffer `parity` as a group binds it.
    pub(crate) fn bound(&self, parity: usize) -> Option<Bound<'a>> {
        if parity == 1 {
            Option::as_ref(self.alternate)
                .map(|buffer| Bound::of(self.alternate, buffer.as_entire_binding()))
        } else {
            Some(self.first.bound())
        }
    }

    /// The frame's pose buffer.
    pub(crate) fn current(&self) -> &'a wgpu::Buffer {
        self.buffer(self.parity)
            .expect("temporal frames create the alternate pose buffer")
    }

    /// The joint palette written with the frame's poses.
    pub(crate) fn palette(&self) -> &'a wgpu::Buffer {
        self.palettes[self.parity]
    }

    /// Advances whenever pose storage is replaced: either pose buffer, a
    /// palette or the colors. Generations only grow, so their greatest
    /// does.
    pub(crate) fn generation(&self) -> u64 {
        [
            self.first.generation(),
            self.alternate.generation(),
            self.palettes[0].generation(),
            self.palettes[1].generation(),
            self.colors.generation(),
        ]
        .into_iter()
        .max()
        .unwrap_or_default()
    }

    /// A key naming the frame's pose buffer: its generation and parity.
    pub(crate) fn key(&self) -> u64 {
        self.generation() * 2 + self.parity as u64
    }
}

fn create_storage_buffer(device: &wgpu::Device, capacity: usize, stride: usize) -> wgpu::Buffer {
    crate::buffers::storage(
        device,
        "engine/slot-storage",
        (capacity * stride) as u64,
        wgpu::BufferUsages::empty(),
    )
}
