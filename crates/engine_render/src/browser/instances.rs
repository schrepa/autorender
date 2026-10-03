//! Instance storage (decisions 013 and 014): public snapshots' instance
//! records, or retained slots' poses and colors at their core slot
//! positions, the slot IDs CPU-prepared frames draw, joint palettes
//! (decision 036), and the pose buffer temporal frames alternate with
//! (decision 017). Storage grows with the frame's instances up to what one
//! binding holds.

use super::*;

pub(super) struct Instances {
    // Most instances this GPU's instance, pose and shadow buffers hold.
    pub(super) limit: usize,
    pub(super) capacity: usize,
    // Public instance records, or retained slots' poses.
    pub(super) buffer: wgpu::Buffer,
    pub(super) staging: Vec<InstanceData>,
    colors: Option<wgpu::Buffer>,
    // Retained poses upload straight from `RenderFrame::pose_rows`.
    pub(super) color_staging: Vec<ColorData>,
    // Group 2 of retained mesh pipelines: poses, colors and palettes.
    pub(super) pose_layout: wgpu::BindGroupLayout,
    pose_group: Option<wgpu::BindGroup>,
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
    // Whether storage holds retained slots rather than public records.
    affine: bool,
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
        // Instances: what the largest per-instance storage (poses, colors,
        // six shadow ID lists) holds within one binding, below the moving
        // flag of visible IDs.
        let binding = device
            .limits()
            .max_storage_buffer_binding_size
            .min(device.limits().max_buffer_size);
        let record = size_of::<InstanceData>()
            .max(size_of::<PoseData>())
            .max(6 * size_of::<u32>()) as u64;
        let limit = (binding / record).min(u64::from(crate::gpu_data::MOVED_SLOT)) as usize;
        let vertex = wgpu::ShaderStages::VERTEX;
        let pose_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/retained-pose-layout"),
            entries: &[
                storage_entry(0, size_of::<PoseData>() as u64, vertex),
                storage_entry(1, size_of::<ColorData>() as u64, vertex),
                // Joint palettes of skinned slots (decision 036).
                storage_entry(3, size_of::<PoseData>() as u64, vertex),
            ],
        });
        Self {
            limit,
            capacity: INITIAL_CAPACITY,
            buffer: create_instance_buffer(device, INITIAL_CAPACITY, size_of::<InstanceData>()),
            staging: Vec::with_capacity(INITIAL_CAPACITY),
            colors: None,
            color_staging: Vec::new(),
            pose_layout,
            pose_group: None,
            alternate: None,
            alternate_group: Cached::default(),
            generation: 0,
            slots: None,
            slot_capacity: 0,
            slot_staging: Vec::new(),
            slot_revision: None,
            affine: false,
            palettes: [0, 1].map(|_| create_storage_buffer(device, 1, size_of::<PoseData>())),
            palette_capacity: 1,
            palette_staging: Vec::new(),
            retained_uploads: RetainedUploads::default(),
            pose_uploads: StagingRing::default(),
        }
    }

    /// Sizes storage for a frame of `payload_count` records (retained slots
    /// when `affine`) and `joints` palette entries, with the alternate pose
    /// buffer of `temporal` frames. Returns whether the record buffer was
    /// replaced, so uploads must write it whole.
    pub(super) fn reserve(
        &mut self,
        device: &wgpu::Device,
        affine: bool,
        payload_count: usize,
        joints: usize,
        temporal: bool,
    ) -> bool {
        let layout_changed = self.affine != affine;
        let grew = payload_count > self.capacity;
        if layout_changed {
            self.affine = affine;
            // A caller switching contracts replaces the old staging layout;
            // do not retain a second full population of instance payloads.
            self.staging = Vec::new();
            self.color_staging = Vec::new();
            self.retained_uploads = RetainedUploads::default();
            self.pose_uploads.clear();
            self.slots = None;
            self.slot_capacity = 0;
            self.slot_staging = Vec::new();
            self.slot_revision = None;
        }
        if grew {
            self.capacity = payload_count.next_power_of_two().min(self.limit);
            engine_profile::counter("render.instance_buffer_growth", 1.0);
        }
        // Joint palettes (decision 036) grow with the frame's joints.
        let palettes_grew = joints > self.palette_capacity;
        if palettes_grew {
            self.palette_capacity = joints.next_power_of_two();
            self.palettes = [0, 1].map(|_| {
                create_storage_buffer(device, self.palette_capacity, size_of::<PoseData>())
            });
        }
        let replaced = grew || layout_changed;
        if replaced {
            self.buffer = if affine {
                create_storage_buffer(device, self.capacity, size_of::<PoseData>())
            } else {
                create_instance_buffer(device, self.capacity, size_of::<InstanceData>())
            };
            self.colors = affine
                .then(|| create_storage_buffer(device, self.capacity, size_of::<ColorData>()));
            self.retained_uploads.invalidate();
            self.alternate = None;
        }
        if replaced || palettes_grew {
            self.pose_group = self.colors.as_ref().map(|colors| {
                pose_group(
                    device,
                    &self.pose_layout,
                    &self.buffer,
                    colors,
                    &self.palettes[0],
                )
            });
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
        replaced
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

    /// GPU presentation's inputs (decision 014) for frames of temporal
    /// `parity`: the staging ring of pose uploads, and the frame's pose
    /// buffer, joint palette and the colors.
    pub(super) fn presentation_inputs(
        &mut self,
        parity: usize,
    ) -> (&mut StagingRing, [&wgpu::Buffer; 3]) {
        let poses = if parity == 1 {
            self.alternate.as_ref().expect("temporal frames create it")
        } else {
            &self.buffer
        };
        let colors = self.colors.as_ref().expect("affine color buffer");
        (
            &mut self.pose_uploads,
            [poses, &self.palettes[parity], colors],
        )
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
        self.colors.as_ref().expect("affine color buffer")
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
            self.pose_group.as_ref().expect("retained pose bindings")
        }
    }

    /// The pose group of odd temporal frames: their poses, the colors and
    /// the second palette.
    pub(super) fn prepare_alternate_group(&mut self, device: &wgpu::Device) {
        if self.alternate_group.stale(&self.generation) {
            let group = pose_group(
                device,
                &self.pose_layout,
                self.poses(1),
                self.colors(),
                &self.palettes[1],
            );
            self.alternate_group.set(self.generation, group);
        }
    }
}

/// Group 2 of retained mesh pipelines: slot poses, colors and the joint
/// palettes of skinned slots (decision 036).
fn pose_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    poses: &wgpu::Buffer,
    colors: &wgpu::Buffer,
    palette: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("engine/retained-pose-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: poses.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: colors.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: palette.as_entire_binding(),
            },
        ],
    })
}

fn create_storage_buffer(device: &wgpu::Device, capacity: usize, stride: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/retained-instance-storage"),
        size: (capacity * stride) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_instance_buffer(device: &wgpu::Device, capacity: usize, stride: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/instance-buffer"),
        size: (capacity * stride) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
