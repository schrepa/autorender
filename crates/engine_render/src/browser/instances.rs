//! Instance storage (decisions 013 and 014): retained slots' poses and
//! colors at their core slot positions, the slot IDs CPU-prepared frames
//! draw, joint palettes (decision 036), and the pose buffer temporal frames
//! alternate with (decision 017). Storage grows with the frame's slots up to
//! what one binding holds.

use super::*;
use crate::bindings::RETAINED_POSES;

pub(super) struct Instances {
    // Most slots this GPU's pose and shadow buffers hold.
    pub(super) limit: usize,
    pub(super) capacity: usize,
    // Slot poses, which upload straight from `RenderFrame::pose_rows`.
    pub(super) buffer: wgpu::Buffer,
    colors: wgpu::Buffer,
    pub(super) color_staging: Vec<ColorData>,
    // Group 2 of retained mesh pipelines: poses, colors, palettes, and
    // grounds (decision 096) none of the slots stand on.
    pub(super) pose_layout: wgpu::BindGroupLayout,
    pose_group: wgpu::BindGroup,
    grounds: wgpu::Buffer,
    // Temporal frames cull into this buffer and `buffer` in turn; odd
    // frames' pose group, for pose storage generation `.0`.
    alternate: Option<wgpu::Buffer>,
    alternate_group: Cached<u64, wgpu::BindGroup>,
    // Advances when pose storage is replaced (compute bindings follow).
    pub(super) generation: u64,
    // Slot IDs of CPU-prepared retained frames in drawing order, staged for
    // the frame identity and structure revision `slot_revision`.
    slots: Option<wgpu::Buffer>,
    pub(super) slot_capacity: usize,
    pub(super) slot_staging: Vec<u32>,
    pub(super) slot_revision: Option<(u64, u64)>,
    // Joint palettes of skinned slots (decision 036), one per pose buffer,
    // with room for `palette_capacity` joints.
    pub(super) palettes: [wgpu::Buffer; 2],
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
        let buffer = create_storage_buffer(device, INITIAL_CAPACITY, size_of::<PoseData>());
        let colors = create_storage_buffer(device, INITIAL_CAPACITY, size_of::<ColorData>());
        let palettes = [0, 1].map(|_| create_storage_buffer(device, 1, size_of::<PoseData>()));
        let grounds = create_storage_buffer(device, 1, size_of::<GroundData>());
        let pose_group = pose_group(
            device,
            &pose_layout,
            [&buffer, &colors, &palettes[0], &grounds],
        );
        Self {
            limit,
            capacity: INITIAL_CAPACITY,
            buffer,
            colors,
            color_staging: Vec::new(),
            pose_layout,
            pose_group,
            grounds,
            alternate: None,
            alternate_group: Cached::default(),
            generation: 0,
            slots: None,
            slot_capacity: 0,
            slot_staging: Vec::new(),
            slot_revision: None,
            palettes,
            palette_capacity: 1,
            palette_staging: Vec::new(),
            retained_uploads: RetainedUploads::default(),
            pose_uploads: StagingRing::default(),
        }
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
            self.buffer = create_storage_buffer(device, self.capacity, size_of::<PoseData>());
            self.colors = create_storage_buffer(device, self.capacity, size_of::<ColorData>());
            self.retained_uploads.invalidate();
            self.alternate = None;
        }
        // Joint palettes (decision 036) grow with the frame's joints.
        let palettes_grew = joints > self.palette_capacity;
        if palettes_grew {
            self.palette_capacity = joints.next_power_of_two();
            self.palettes = [0, 1].map(|_| {
                create_storage_buffer(device, self.palette_capacity, size_of::<PoseData>())
            });
        }
        if grew || palettes_grew {
            self.pose_group = pose_group(
                device,
                &self.pose_layout,
                [&self.buffer, &self.colors, &self.palettes[0], &self.grounds],
            );
            self.generation += 1;
        }
        if temporal && self.alternate.is_none() {
            self.alternate = Some(create_storage_buffer(
                device,
                self.capacity,
                size_of::<PoseData>(),
            ));
            self.generation += 1;
        }
    }

    /// Stages the slot IDs of a CPU-prepared retained frame drawing
    /// `ordered`, unless already staged for `key` (its identity and
    /// structure revision); returns whether they need writing. Virtual
    /// geometry's selection (decision 044) reads them too.
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
            self.alternate.as_ref().expect("temporal frames create it")
        } else {
            &self.buffer
        }
    }

    /// The pose storage of retained frames of temporal `parity`.
    pub(super) fn frame_poses(&self, parity: usize) -> FramePoses<'_> {
        FramePoses {
            buffers: [Some(&self.buffer), self.alternate.as_ref()],
            palettes: [&self.palettes[0], &self.palettes[1]],
            colors: self.colors(),
            parity,
            generation: self.generation,
        }
    }

    /// GPU presentation's inputs (decision 014) for frames of temporal
    /// `parity`: the staging ring of pose uploads, and the pose storage.
    pub(super) fn presentation_inputs(
        &mut self,
        parity: usize,
    ) -> (&mut StagingRing, FramePoses<'_>) {
        let poses = FramePoses {
            buffers: [Some(&self.buffer), self.alternate.as_ref()],
            palettes: [&self.palettes[0], &self.palettes[1]],
            colors: &self.colors,
            parity,
            generation: self.generation,
        };
        (&mut self.pose_uploads, poses)
    }

    /// Both pose buffers: the first again when there is no alternate one.
    pub(super) fn pose_buffers(&self) -> [&wgpu::Buffer; 2] {
        [
            &self.buffer,
            self.alternate.as_ref().unwrap_or(&self.buffer),
        ]
    }

    /// The pose buffer odd temporal frames cull into, once created.
    pub(super) fn alternate(&self) -> Option<&wgpu::Buffer> {
        self.alternate.as_ref()
    }

    /// The alternate pose buffer's bytes, 0 without one.
    pub(super) fn alternate_bytes(&self) -> u64 {
        self.alternate.as_ref().map_or(0, wgpu::Buffer::size)
    }

    /// Releases the alternate pose buffer when temporal antialiasing stops.
    pub(super) fn release_alternate(&mut self) {
        if self.alternate.take().is_some() {
            // Culling must not keep writing the released buffer.
            self.generation += 1;
        }
    }

    /// Retained slots' colors.
    pub(super) fn colors(&self) -> &wgpu::Buffer {
        &self.colors
    }

    /// Grounds of retained slots (decision 096): none stand on terrain.
    pub(super) fn grounds(&self) -> &wgpu::Buffer {
        &self.grounds
    }

    /// Slot IDs of CPU-prepared retained frames.
    pub(super) fn slots(&self) -> &wgpu::Buffer {
        self.slots.as_ref().expect("retained slot buffer")
    }

    /// Group 2 of retained mesh pipelines for frames of temporal `parity`;
    /// odd frames' after [`Self::prepare_alternate_group`].
    pub(super) fn pose_group(&self, parity: usize) -> &wgpu::BindGroup {
        if parity == 1 {
            self.alternate_group.get().expect("prepared this frame")
        } else {
            &self.pose_group
        }
    }

    /// The pose group of odd temporal frames: their poses, the colors and
    /// the second palette.
    pub(super) fn prepare_alternate_group(&mut self, device: &wgpu::Device) {
        if self.alternate_group.stale(&self.generation) {
            let group = pose_group(
                device,
                &self.pose_layout,
                [
                    self.poses(1),
                    self.colors(),
                    &self.palettes[1],
                    &self.grounds,
                ],
            );
            self.alternate_group.set(self.generation, group);
        }
    }
}

/// The pose storage of a retained frame, which GPU presentation (decision
/// 014) culls into and every pass drawing or lighting slots reads: both
/// pose buffers (the second while temporal antialiasing alternates them,
/// decision 017), the joint palette written with each (decision 036), the
/// slot colors, which buffer the frame's poses are in, and the generation
/// that advances when any of them is replaced.
#[derive(Clone, Copy)]
pub(crate) struct FramePoses<'a> {
    pub buffers: [Option<&'a wgpu::Buffer>; 2],
    pub palettes: [&'a wgpu::Buffer; 2],
    pub colors: &'a wgpu::Buffer,
    pub parity: usize,
    pub generation: u64,
}

impl<'a> FramePoses<'a> {
    /// The frame's pose buffer.
    pub(crate) fn current(&self) -> &'a wgpu::Buffer {
        self.buffers[self.parity].expect("temporal frames create the alternate pose buffer")
    }

    /// The joint palette written with the frame's poses.
    pub(crate) fn palette(&self) -> &'a wgpu::Buffer {
        self.palettes[self.parity]
    }

    /// A key naming the frame's pose buffer: its generation and parity.
    pub(crate) fn key(&self) -> u64 {
        self.generation * 2 + self.parity as u64
    }
}

/// Group 2 of retained mesh pipelines: slot poses, colors, the joint
/// palettes of skinned slots (decision 036) and grounds (decision 096).
fn pose_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    [poses, colors, palette, grounds]: [&wgpu::Buffer; 4],
) -> wgpu::BindGroup {
    crate::pipelines::group(
        device,
        "engine/retained-pose-bind-group",
        &RETAINED_POSES,
        layout,
        &[
            ("poses", poses.as_entire_binding()),
            ("colors", colors.as_entire_binding()),
            ("palette", palette.as_entire_binding()),
            ("grounds", grounds.as_entire_binding()),
        ],
    )
}

fn create_storage_buffer(device: &wgpu::Device, capacity: usize, stride: usize) -> wgpu::Buffer {
    crate::buffers::storage(
        device,
        "engine/slot-storage",
        (capacity * stride) as u64,
        wgpu::BufferUsages::empty(),
    )
}
