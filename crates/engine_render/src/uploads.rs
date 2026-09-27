//! Exact CPU upload reuse. Staging remains the only retained instance copy;
//! neither source pointers nor gameplay revisions establish GPU contents.

use engine_core::RenderInstance;

use crate::gpu_data::{CameraUniform, InstanceData};

const MAX_WRITE_RANGES: usize = 32;
// A heuristic byte allowance per additional queue call, not a timing claim.
// Include small clean gaps to limit browser/GPU staging operations.
const WRITE_CALL_BYTES: usize = 16 * 1024;
const MERGE_GAP_RECORDS: usize = WRITE_CALL_BYTES.div_ceil(size_of::<InstanceData>());

#[derive(Clone, Copy, Default)]
pub(crate) struct UploadRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Default)]
pub(crate) struct UploadCache {
    camera: Option<CameraUniform>,
    instances_committed: bool,
    ranges: [UploadRange; MAX_WRITE_RANGES],
    range_count: usize,
}

pub(crate) struct PreparedInstances {
    pub write: bool,
    /// Records copied into staging, independently of bytes needing upload.
    pub repacked: usize,
    pub full_upload: bool,
    /// Fragmentation/density exceeded the plan's byte/call thresholds.
    pub fallback: bool,
}

impl UploadCache {
    pub fn camera_needs_write(&self, camera: &CameraUniform) -> bool {
        self.camera
            .as_ref()
            .is_none_or(|previous| bytemuck::bytes_of(previous) != bytemuck::bytes_of(camera))
    }

    /// Call only after the queue accepted the camera bytes.
    pub fn commit_camera(&mut self, camera: CameraUniform) {
        self.camera = Some(camera);
    }

    /// A replacement GPU buffer has no previously uploaded instance contents.
    pub fn invalidate_instances(&mut self) {
        self.instances_committed = false;
    }

    /// The validated draw plan supplies a complete, bounded ordering. Compare
    /// source matrix/color bytes directly; unchanged records need no repacking.
    /// Changed records update staging in place. Nearby changes share a write;
    /// fragmented/dense changes fall back to one full write. An uncommitted
    /// earlier preparation may have overwritten staging, so retries write all.
    pub fn prepare_instances(
        &mut self,
        instances: &[RenderInstance],
        order: &[usize],
        staging: &mut Vec<InstanceData>,
    ) -> PreparedInstances {
        let mut repacked = 0;
        self.range_count = 0;
        if staging.len() != order.len() {
            self.instances_committed = false;
            staging.truncate(order.len());
            staging.reserve_exact(order.len().saturating_sub(staging.len()));
        }
        let force_full = !self.instances_committed;
        // Preparation cannot certify GPU contents, even if a later queue call
        // aborts after writing only a prefix of the selected ranges.
        self.instances_committed = false;
        let mut fragmented = false;
        for (position, &index) in order.iter().enumerate() {
            let source = &instances[index];
            if let Some(previous) = staging.get_mut(position) {
                let record = InstanceData::from(source);
                if bytemuck::bytes_of(previous) == bytemuck::bytes_of(&record) {
                    continue;
                }
                *previous = record;
            } else {
                staging.push(InstanceData::from(source));
            }
            repacked += 1;
            if !force_full && !fragmented {
                fragmented = !self.add_changed_record(position);
            }
        }
        let mut fallback = fragmented;
        if !force_full && !fragmented && self.range_count != 0 {
            let covered: usize = self
                .instance_ranges()
                .iter()
                .map(|range| range.end - range.start)
                .sum();
            let bytes = covered * size_of::<InstanceData>();
            let full_bytes = staging.len() * size_of::<InstanceData>();
            fallback = covered * 4 >= staging.len() * 3
                || bytes + (self.range_count - 1) * WRITE_CALL_BYTES >= full_bytes;
        }
        let full_upload = !staging.is_empty() && (force_full || fallback);
        if full_upload {
            self.ranges[0] = UploadRange {
                start: 0,
                end: staging.len(),
            };
            self.range_count = 1;
        }
        let write = self.range_count != 0;
        if !write && !force_full {
            self.instances_committed = true;
        }
        PreparedInstances {
            write,
            repacked,
            full_upload,
            fallback,
        }
    }

    fn add_changed_record(&mut self, position: usize) -> bool {
        if self.range_count != 0 {
            let previous = &mut self.ranges[self.range_count - 1];
            if position - previous.end <= MERGE_GAP_RECORDS {
                previous.end = position + 1;
                return true;
            }
        }
        if self.range_count == MAX_WRITE_RANGES {
            return false;
        }
        self.ranges[self.range_count] = UploadRange {
            start: position,
            end: position + 1,
        };
        self.range_count += 1;
        true
    }

    pub fn instance_ranges(&self) -> &[UploadRange] {
        &self.ranges[..self.range_count]
    }

    pub const fn range_scratch_bytes() -> usize {
        size_of::<[UploadRange; MAX_WRITE_RANGES]>()
    }

    /// Call only after the queue accepted all current staging bytes.
    pub fn commit_instances(&mut self) {
        self.instances_committed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{AssetLibrary, Mesh};
    use glam::{Mat4, Vec3};

    fn instances() -> Vec<RenderInstance> {
        let mut assets = AssetLibrary::default();
        let mesh = assets.insert_mesh(Mesh::cube()).unwrap();
        [Vec3::ZERO, Vec3::X, Vec3::Y]
            .map(|position| RenderInstance {
                mesh,
                material: None,
                model: Mat4::from_translation(position),
                color: [1.0; 4],
                surface: engine_core::Surface::MATTE,
            })
            .to_vec()
    }

    #[test]
    fn static_cache_camera_reuse_requires_committed_exact_bytes() {
        let mut cache = UploadCache::default();
        let camera = CameraUniform::exact(Mat4::IDENTITY, glam::Vec3::ZERO, 0.0);
        assert!(cache.camera_needs_write(&camera));
        // Merely inspecting or preparing a skipped frame does not commit it.
        assert!(cache.camera_needs_write(&camera));
        cache.commit_camera(camera);
        assert!(!cache.camera_needs_write(&camera));
        let mut changed = camera;
        changed.view_projection[3][0] = -0.0;
        assert!(cache.camera_needs_write(&changed));
        cache.commit_camera(changed);
        assert!(!cache.camera_needs_write(&changed));
        assert!(cache.camera_needs_write(&camera));
        // Temporal jitter and history are camera bytes too.
        let mut jittered = changed;
        jittered.jitter[0] = 0.001;
        assert!(cache.camera_needs_write(&jittered));
    }

    #[test]
    fn static_cache_changed_pose_color_and_order_repack_without_a_second_instance_cache() {
        let mut cache = UploadCache::default();
        let mut source = instances();
        let mut staging = Vec::with_capacity(3);
        let first = cache.prepare_instances(&source, &[0, 1, 2], &mut staging);
        assert!(first.write);
        assert_eq!(first.repacked, 3);
        cache.commit_instances();
        let capacity = staging.capacity();
        let same = cache.prepare_instances(&source, &[0, 1, 2], &mut staging);
        assert!(!same.write);
        assert_eq!(same.repacked, 0);
        source[1].color[2] = 0.25;
        let changed = cache.prepare_instances(&source, &[0, 1, 2], &mut staging);
        assert!(changed.write);
        assert_eq!(changed.repacked, 1);
        assert_eq!(
            staging[1].color,
            crate::gpu_data::pack_color(source[1].color)
        );
        cache.commit_instances();
        source[1].model = Mat4::from_translation(Vec3::Z);
        let moved = cache.prepare_instances(&source, &[0, 1, 2], &mut staging);
        assert!(moved.write);
        assert_eq!(moved.repacked, 1);
        assert_eq!(staging[1].model[3][2], 1.0);
        cache.commit_instances();
        let reordered = cache.prepare_instances(&source, &[2, 1, 0], &mut staging);
        assert!(reordered.write);
        assert_eq!(reordered.repacked, 2);
        assert_eq!(staging[0].model[3][1], 1.0);
        assert_eq!(staging.capacity(), capacity);
    }

    #[test]
    fn static_cache_retries_empty_frames_and_replaced_buffers_do_not_reuse_unwritten_data() {
        let mut cache = UploadCache::default();
        let mut source = instances();
        let mut staging = Vec::new();
        assert!(
            cache
                .prepare_instances(&source, &[0, 1, 2], &mut staging)
                .write
        );
        // The staging bytes match, but an aborted write still needs retrying.
        let retry = cache.prepare_instances(&source, &[0, 1, 2], &mut staging);
        assert!(retry.write);
        assert_eq!(retry.repacked, 0);
        cache.commit_instances();
        cache.invalidate_instances();
        assert!(
            cache
                .prepare_instances(&source, &[0, 1, 2], &mut staging)
                .write
        );
        cache.commit_instances();
        source.truncate(1);
        assert!(cache.prepare_instances(&source, &[0], &mut staging).write);
        cache.commit_instances();
        source.clear();
        assert!(!cache.prepare_instances(&source, &[], &mut staging).write);
        source = instances();
        assert!(
            cache
                .prepare_instances(&source, &[0, 1, 2], &mut staging)
                .write
        );
        cache.commit_instances();
        // Bitwise comparison remains stable even for client-supplied NaN color.
        source[0].color[0] = f32::from_bits(0x7fc0_0001);
        assert!(
            cache
                .prepare_instances(&source, &[0, 1, 2], &mut staging)
                .write
        );
        cache.commit_instances();
        assert!(
            !cache
                .prepare_instances(&source, &[0, 1, 2], &mut staging)
                .write
        );
        // Another NaN payload packs to the same record: nothing to write.
        source[0].color[0] = f32::from_bits(0x7fc0_0002);
        assert!(
            !cache
                .prepare_instances(&source, &[0, 1, 2], &mut staging)
                .write
        );
        source[0].color[0] = 0.5;
        assert!(
            cache
                .prepare_instances(&source, &[0, 1, 2], &mut staging)
                .write
        );
    }

    #[test]
    fn partial_writes_reconstruct_all_bytes_and_aborted_ranges_retry_in_full() {
        let mut source = vec![instances()[0]; 2048];
        let order: Vec<_> = (0..source.len()).collect();
        let mut cache = UploadCache::default();
        let mut staging = Vec::new();
        cache.prepare_instances(&source, &order, &mut staging);
        let mut gpu = staging.clone();
        cache.commit_instances();
        for range in [50..60, 90..100, 500..540, 1500..1530] {
            for record in &mut source[range] {
                record.color[0] = 0.25;
            }
        }
        let prepared = cache.prepare_instances(&source, &order, &mut staging);
        assert_eq!(prepared.repacked, 90);
        assert!(!prepared.full_upload);
        assert!(!prepared.fallback);
        let ranges: Vec<_> = cache
            .instance_ranges()
            .iter()
            .map(|range| range.start..range.end)
            .collect();
        assert_eq!(ranges, [50..100, 500..540, 1500..1530]);
        for range in &ranges {
            gpu[range.clone()].copy_from_slice(&staging[range.clone()]);
        }
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&gpu),
            bytemuck::cast_slice::<_, u8>(&staging)
        );
        cache.commit_instances();
        assert!(!cache.prepare_instances(&source, &order, &mut staging).write);

        source[70].model = Mat4::from_translation(Vec3::Z);
        source[1700].color[2] = 0.5;
        assert!(
            !cache
                .prepare_instances(&source, &order, &mut staging)
                .full_upload
        );
        // Only the first write reached the queue before the caller aborted.
        let first = cache.instance_ranges()[0];
        gpu[first.start..first.end].copy_from_slice(&staging[first.start..first.end]);
        source[1900].color[1] = 0.75;
        let retry = cache.prepare_instances(&source, &order, &mut staging);
        assert!(retry.full_upload);
        assert!(!retry.fallback);
        assert_eq!(retry.repacked, 1);
        let range = cache.instance_ranges()[0];
        gpu[range.start..range.end].copy_from_slice(&staging[range.start..range.end]);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&gpu),
            bytemuck::cast_slice::<_, u8>(&staging)
        );
    }

    #[test]
    fn fragmented_and_dense_changes_use_one_bounded_full_write() {
        assert!(UploadCache::range_scratch_bytes() <= 512);
        let mut source = vec![instances()[0]; 16_000];
        let order: Vec<_> = (0..source.len()).collect();
        let mut cache = UploadCache::default();
        let mut staging = Vec::new();
        cache.prepare_instances(&source, &order, &mut staging);
        cache.commit_instances();
        for index in (0..source.len())
            .step_by(MERGE_GAP_RECORDS + 2)
            .take(MAX_WRITE_RANGES + 1)
        {
            source[index].color[0] = 0.0;
        }
        let fragmented = cache.prepare_instances(&source, &order, &mut staging);
        assert!(fragmented.full_upload && fragmented.fallback);
        assert_eq!(cache.instance_ranges().len(), 1);
        assert_eq!(cache.instance_ranges()[0].end, source.len());
        cache.commit_instances();
        // Interleaved motion coalesces into a dense range instead of thousands
        // of individual queue calls, then falls back to the complete buffer.
        for index in (0..source.len()).step_by(4) {
            source[index].color[1] = 0.0;
        }
        let dense = cache.prepare_instances(&source, &order, &mut staging);
        assert!(dense.full_upload && dense.fallback);
        assert_eq!(cache.instance_ranges().len(), 1);
        assert_eq!(cache.instance_ranges()[0].end, source.len());
    }
}
