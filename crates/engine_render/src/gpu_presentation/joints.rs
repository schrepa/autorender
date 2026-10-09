//! Skinned slots' joints: each joint's node and inverse bind
//! matrix, uploaded when the frame's joints changed, from which the palette
//! pass composes joint palettes every frame.

use engine_core::{PoseRows, RenderJoint};

use super::{grow, storage};

pub(super) struct Joints {
    capacity: usize,
    pub(super) nodes: wgpu::Buffer,
    pub(super) inverse_binds: wgpu::Buffer,
    // The joints as last uploaded.
    uploaded_nodes: Vec<u32>,
    uploaded_binds: Vec<PoseRows>,
    // Joints this frame's palette pass writes.
    count: u32,
    // The retained stream and joints revision the buffers hold.
    revision: Option<(u64, u64)>,
    // Advances when joint storage is replaced.
    generation: u64,
}

impl Joints {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let empty = |label| storage(device, label, 0, wgpu::BufferUsages::empty());
        Self {
            capacity: 0,
            nodes: empty("engine/presentation-joint-nodes"),
            inverse_binds: empty("engine/presentation-inverse-binds"),
            uploaded_nodes: Vec::new(),
            uploaded_binds: Vec::new(),
            count: 0,
            revision: None,
            generation: 0,
        }
    }

    /// Joints this frame's palette pass writes.
    pub(super) fn count(&self) -> u32 {
        self.count
    }

    /// Advances when joint storage is replaced.
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn capacity_bytes(&self) -> u64 {
        self.nodes.size() + self.inverse_binds.size()
    }

    /// Uploads the frame's `joints` when they changed; `revision` (the
    /// frame's retained stream and joints revision) names them.
    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        joints: &[RenderJoint],
        revision: (u64, u64),
    ) {
        self.count = joints.len() as u32;
        if self.revision == Some(revision) && joints.len() <= self.capacity {
            return;
        }
        if joints.len() > self.capacity {
            self.capacity = grow(joints.len());
            let capacity = self.capacity as u64;
            self.nodes = storage(
                device,
                "engine/presentation-joint-nodes",
                capacity * 4,
                wgpu::BufferUsages::empty(),
            );
            self.inverse_binds = storage(
                device,
                "engine/presentation-inverse-binds",
                capacity * size_of::<PoseRows>() as u64,
                wgpu::BufferUsages::empty(),
            );
            self.uploaded_nodes.clear();
            self.uploaded_binds.clear();
            self.generation += 1;
        }
        self.revision = Some(revision);
        let unchanged = self.uploaded_nodes.len() == joints.len()
            && joints
                .iter()
                .zip(&self.uploaded_nodes)
                .zip(&self.uploaded_binds)
                .all(|((joint, node), bind)| joint.node == *node && joint.inverse_bind == *bind);
        if unchanged || joints.is_empty() {
            return;
        }
        self.uploaded_nodes.clear();
        self.uploaded_nodes
            .extend(joints.iter().map(|joint| joint.node));
        self.uploaded_binds.clear();
        self.uploaded_binds
            .extend(joints.iter().map(|joint| joint.inverse_bind));
        queue.write_buffer(&self.nodes, 0, bytemuck::cast_slice(&self.uploaded_nodes));
        queue.write_buffer(
            &self.inverse_binds,
            0,
            bytemuck::cast_slice(&self.uploaded_binds),
        );
    }
}
