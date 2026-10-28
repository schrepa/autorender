#![expect(
    clippy::expect_used,
    reason = "pose buffers exist in the frames that bind them"
)]

use super::*;
use crate::bindings::RETAINED_POSES;
use crate::cached::GroupCache;

pub(super) struct Instances {
    pub(super) limit: usize,
    pub(super) capacity: usize,
    pub(super) buffer: Versioned<wgpu::Buffer>,
    colors: Versioned<wgpu::Buffer>,
    pub(super) color_staging: Vec<ColorData>,
    pub(super) pose_layout: crate::pipelines::GroupLayout,
    pose_groups: [GroupCache<3>; 2],
    grounds: wgpu::Buffer,
    alternate: Versioned<Option<wgpu::Buffer>>,
    slots: Option<wgpu::Buffer>,
    pub(super) slot_capacity: usize,
    pub(super) slot_staging: Vec<u32>,
    pub(super) slot_revision: Option<(u64, u64)>,
    pub(super) retained_uploads: RetainedUploads,
    pub(super) pose_uploads: StagingRing,
}

impl Instances {
    pub(super) fn new(device: &wgpu::Device) -> Self {
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
            retained_uploads: RetainedUploads::default(),
            pose_uploads: StagingRing::default(),
        };
        instances.prepare_pose_group(device, 0);
        instances
    }

    pub(super) fn reserve(&mut self, device: &wgpu::Device, payload_count: usize, temporal: bool) {
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
        if temporal && self.alternate.is_none() {
            self.alternate.replace(Some(create_storage_buffer(
                device,
                self.capacity,
                size_of::<PoseData>(),
            )));
        }
        self.prepare_pose_group(device, 0);
    }

    pub(super) fn generation(&self) -> u64 {
        self.frame_poses(0).generation()
    }

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

    pub(super) fn poses(&self, parity: usize) -> &wgpu::Buffer {
        if parity == 1 {
            Option::as_ref(&self.alternate).expect("temporal frames create it")
        } else {
            &self.buffer
        }
    }

    pub(super) fn frame_poses(&self, parity: usize) -> FramePoses<'_> {
        FramePoses {
            first: &self.buffer,
            alternate: &self.alternate,
            colors: &self.colors,
            parity,
        }
    }

    pub(super) fn presentation_inputs(
        &mut self,
        parity: usize,
    ) -> (&mut StagingRing, FramePoses<'_>) {
        let poses = FramePoses {
            first: &self.buffer,
            alternate: &self.alternate,
            colors: &self.colors,
            parity,
        };
        (&mut self.pose_uploads, poses)
    }

    pub(super) fn alternate(&self) -> Option<&wgpu::Buffer> {
        Option::as_ref(&self.alternate)
    }

    pub(super) fn alternate_bytes(&self) -> u64 {
        self.alternate().map_or(0, wgpu::Buffer::size)
    }

    pub(super) fn release_alternate(&mut self) {
        if self.alternate.is_some() {
            self.alternate.replace(None);
        }
    }

    pub(super) fn colors(&self) -> &Versioned<wgpu::Buffer> {
        &self.colors
    }

    pub(super) fn grounds(&self) -> &wgpu::Buffer {
        &self.grounds
    }

    pub(super) fn slots(&self) -> &wgpu::Buffer {
        self.slots.as_ref().expect("retained slot buffer")
    }

    pub(super) fn pose_group(&self, parity: usize) -> &wgpu::BindGroup {
        self.pose_groups[parity].get().expect("prepared this frame")
    }

    pub(super) fn prepare_pose_group(&mut self, device: &wgpu::Device, parity: usize) {
        let poses = FramePoses {
            first: &self.buffer,
            alternate: &self.alternate,
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
                ("grounds", Bound::fixed(self.grounds.as_entire_binding())),
            ],
        );
    }
}

#[derive(Clone, Copy)]
pub(crate) struct FramePoses<'a> {
    first: &'a Versioned<wgpu::Buffer>,
    alternate: &'a Versioned<Option<wgpu::Buffer>>,
    pub colors: &'a Versioned<wgpu::Buffer>,
    pub parity: usize,
}

impl<'a> FramePoses<'a> {
    pub(crate) fn buffer(&self, parity: usize) -> Option<&'a wgpu::Buffer> {
        if parity == 1 {
            Option::as_ref(self.alternate)
        } else {
            Some(self.first)
        }
    }

    pub(crate) fn bound(&self, parity: usize) -> Option<Bound<'a>> {
        if parity == 1 {
            Option::as_ref(self.alternate)
                .map(|buffer| Bound::of(self.alternate, buffer.as_entire_binding()))
        } else {
            Some(self.first.bound())
        }
    }

    pub(crate) fn current(&self) -> &'a wgpu::Buffer {
        self.buffer(self.parity)
            .expect("temporal frames create the alternate pose buffer")
    }

    pub(crate) fn generation(&self) -> u64 {
        [
            self.first.generation(),
            self.alternate.generation(),
            self.colors.generation(),
        ]
        .into_iter()
        .max()
        .unwrap_or_default()
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
