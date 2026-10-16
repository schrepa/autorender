//! Optional view fog appended to the existing cloud-shadow binding. The
//! original prefix remains the cloud pass's target; enabled frames copy it
//! into this allocation before any view shading. Off owns no allocation.

use engine_core::math;
use engine_core::{BasinFog, RenderBasinFog, RenderSnapshot};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

const PREFIX: usize = crate::gpu_layouts::clouds::CLOUD_SHADOW_BYTES / 4;
const HEADER: usize = 4;
const NODE: usize = 12;

struct MapKey {
    entity: u64,
    revision: u64,
    origin: [u32; 3],
    uploaded: Arc<AtomicU64>,
}

impl PartialEq for MapKey {
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity
            && self.revision == other.revision
            && self.origin == other.origin
            // A replacement world may reuse every numeric key. Retaining
            // the acknowledgement also prevents allocator-address reuse.
            && Arc::ptr_eq(&self.uploaded, &other.uploaded)
    }
}

#[derive(Default)]
pub(crate) struct BasinPass {
    pub(crate) enabled: bool,
    pub(crate) buffer: Option<wgpu::Buffer>,
    pub(crate) generation: u64,
    pub(crate) bytes: u64,
    pub(crate) upload_bytes: u64,
    pub(crate) nodes: u64,
    key: Vec<MapKey>,
    settings: Option<BasinFog>,
}

pub(crate) fn capacity(device: &wgpu::Device) -> u64 {
    let limits = device.limits();
    limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size)
        .saturating_sub((PREFIX * 4) as u64)
}

impl BasinPass {
    /// Returns whether scene pipelines must switch specialization.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        snapshot: &RenderSnapshot,
        cloud_shadows: &wgpu::Buffer,
    ) -> Result<bool, crate::RenderError> {
        self.upload_bytes = 0;
        let settings = snapshot.environment.basin_fog;
        let enabled = settings.enabled && settings.budget_m > 0.0;
        let switched = self.enabled != enabled;
        self.enabled = enabled;
        if !enabled {
            if self.buffer.take().is_some() {
                self.generation = self.generation.wrapping_add(1);
            }
            self.key = Vec::new();
            self.settings = None;
            self.bytes = 0;
            self.nodes = 0;
            return Ok(switched);
        }
        let key: Vec<_> = snapshot
            .terrain
            .basin_fog
            .iter()
            .map(|t| MapKey {
                entity: t.entity.to_bits(),
                revision: t.revision,
                origin: t.origin.to_array().map(f32::to_bits),
                uploaded: t.uploaded.clone(),
            })
            .collect();
        if self.key == key && self.settings == Some(settings) {
            return Ok(switched);
        }
        let _profile = engine_profile::span("cpu", "basin_fog_upload");
        if let Some(buffer) = &self.buffer
            && self.key == key
        {
            let header = [
                (PREFIX + HEADER + self.nodes as usize * NODE) as u32,
                self.nodes as u32,
                settings.extinction_per_m.to_bits(),
                settings.top_fade_m.to_bits(),
            ];
            queue.write_buffer(buffer, (PREFIX * 4) as u64, bytemuck::cast_slice(&header));
            self.settings = Some(settings);
            self.upload_bytes = 16;
            return Ok(switched);
        }
        let error = || crate::RenderError("basin fog exceeds checked storage capacity".into());
        let cells = snapshot
            .terrain
            .basin_fog
            .iter()
            .try_fold(0usize, |n, t| n.checked_add(t.grid.floors.len()))
            .ok_or_else(error)?;
        let maximum = cells
            .checked_mul(2 * NODE)
            .and_then(|n| n.checked_add(HEADER))
            .ok_or_else(error)?;
        if maximum as u64 > capacity(device) / 4 || maximum > u32::MAX as usize - PREFIX {
            return Err(error());
        }
        let mut words = Vec::new();
        words.try_reserve_exact(maximum).map_err(|_| error())?;
        words.extend([
            0,
            0,
            settings.extinction_per_m.to_bits(),
            settings.top_fade_m.to_bits(),
        ]);
        for terrain in &snapshot.terrain.basin_fog {
            let _ = append(&mut words, terrain, [0, 0], terrain.grid.size).ok_or_else(error)?;
        }
        words[0] = (PREFIX + words.len()) as u32;
        words[1] = ((words.len() - HEADER) / NODE) as u32;
        if words.len() == HEADER {
            if self.buffer.take().is_some() {
                self.generation = self.generation.wrapping_add(1);
            }
            self.key = key;
            self.settings = Some(settings);
            self.bytes = 0;
            self.nodes = 0;
            for terrain in &snapshot.terrain.basin_fog {
                terrain.uploaded.store(terrain.revision, Ordering::Relaxed);
            }
            return Ok(switched);
        }
        let bytes = ((PREFIX + words.len()) * 4) as u64;
        if self
            .buffer
            .as_ref()
            .is_none_or(|b| b.size() != bytes.next_multiple_of(16))
        {
            self.buffer = Some(crate::buffers::storage(
                device,
                "engine/basin-fog",
                bytes,
                wgpu::BufferUsages::empty(),
            ));
            self.generation = self.generation.wrapping_add(1);
            // Probe/environment submissions precede the view. Give a newly
            // bound allocation the existing shadow prefix before they read it.
            let mut prefix = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/basin-cloud-prefix"),
            });
            self.copy_prefix(&mut prefix, cloud_shadows);
            queue.submit_now([prefix.finish()]);
        }
        let buffer = self.buffer.as_ref().ok_or_else(error)?;
        queue.write_buffer(buffer, (PREFIX * 4) as u64, bytemuck::cast_slice(&words));
        self.bytes = buffer.size();
        self.upload_bytes = (words.len() * 4) as u64;
        self.nodes = u64::from(words[1]);
        self.key = key;
        self.settings = Some(settings);
        // The queue write is ordered before this frame's submission. Published
        // maps are immutable, so the revision acknowledged is exactly this one.
        for terrain in &snapshot.terrain.basin_fog {
            terrain.uploaded.store(terrain.revision, Ordering::Relaxed);
        }
        Ok(switched)
    }

    pub(crate) fn copy_prefix(&self, encoder: &mut wgpu::CommandEncoder, source: &wgpu::Buffer) {
        if let Some(buffer) = &self.buffer {
            encoder.copy_buffer_to_buffer(source, 0, buffer, 0, (PREFIX * 4) as u64);
        }
    }
}

/// Preorder binary spatial tree, with forward escape addresses. Empty
/// subtrees disappear. Every nonempty leaf is one complete fixed DEM cell.
type NodeBounds = ([f32; 3], [f32; 3]);

fn append(
    words: &mut Vec<u32>,
    terrain: &RenderBasinFog,
    low: [u32; 2],
    high: [u32; 2],
) -> Option<Option<NodeBounds>> {
    let grid = &terrain.grid;
    let start = words.len();
    words.resize(start + NODE, 0);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let leaf = high[0] - low[0] == 1 && high[1] - low[1] == 1;
    if leaf {
        let at = low[1] as usize * grid.size[0] as usize + low[0] as usize;
        if grid.pools[at] == 0 || grid.tops[at] <= grid.floors[at] {
            words.truncate(start);
            return Some(None);
        }
        min = [
            low[0] as f32 * grid.spacing + terrain.origin.x,
            grid.floors[at] + terrain.origin.y,
            low[1] as f32 * grid.spacing + terrain.origin.z,
        ];
        max = [
            math::min(high[0] as f32 * grid.spacing, grid.extent[0]) + terrain.origin.x,
            grid.tops[at] + terrain.origin.y,
            math::min(high[1] as f32 * grid.spacing, grid.extent[1]) + terrain.origin.z,
        ];
    } else {
        let axis = usize::from(high[1] - low[1] > high[0] - low[0]);
        let middle = low[axis] + (high[axis] - low[axis]) / 2;
        let mut left = high;
        left[axis] = middle;
        let mut right = low;
        right[axis] = middle;
        for (a, b) in [(low, left), (right, high)] {
            if let Some((child_min, child_max)) = append(words, terrain, a, b)? {
                for i in 0..3 {
                    min[i] = math::min(min[i], child_min[i]);
                    max[i] = math::max(max[i], child_max[i]);
                }
            }
        }
        if min[0] == f32::INFINITY {
            words.truncate(start);
            return Some(None);
        }
    }
    if min.iter().chain(&max).any(|v| !v.is_finite()) {
        return None;
    }
    let escape = u32::try_from(PREFIX + words.len()).ok()?;
    words[start..start + NODE].copy_from_slice(&[
        min[0].to_bits(),
        min[1].to_bits(),
        min[2].to_bits(),
        escape,
        max[0].to_bits(),
        max[1].to_bits(),
        max[2].to_bits(),
        u32::from(leaf),
        min[1].to_bits(),
        max[1].to_bits(),
        0,
        0,
    ]);
    Some(Some((min, max)))
}
