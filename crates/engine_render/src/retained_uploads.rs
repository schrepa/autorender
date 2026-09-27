//! Revision-driven uploads for core-owned affine frames. Public snapshots use
//! the independent exact-byte cache because callers may mutate any field.
//!
//! Poses need no packing here: core keeps every slot's pose in the upload
//! layout (`RenderFrame::pose_rows`) and this cache only selects changed slots.

#[cfg(target_arch = "wasm32")]
use engine_core::RenderFrame;
use engine_core::RenderInstance;

use crate::gpu_data::{ColorData, PoseData};
use crate::uploads::UploadRange;

const MAX_WRITE_RANGES: usize = 32;
const WRITE_CALL_BYTES: usize = 16 * 1024;

#[derive(Default)]
struct WritePlan {
    ranges: [UploadRange; MAX_WRITE_RANGES],
    count: usize,
    fragmented: bool,
}

impl WritePlan {
    fn clear(&mut self) {
        self.count = 0;
        self.fragmented = false;
    }

    fn changed(&mut self, position: usize, stride: usize) {
        if self.fragmented {
            return;
        }
        if self.count != 0 {
            let range = &mut self.ranges[self.count - 1];
            if position - range.end <= WRITE_CALL_BYTES.div_ceil(stride) {
                range.end = position + 1;
                return;
            }
        }
        if self.count == MAX_WRITE_RANGES {
            self.fragmented = true;
            return;
        }
        self.ranges[self.count] = UploadRange {
            start: position,
            end: position + 1,
        };
        self.count += 1;
    }

    fn finish(&mut self, len: usize, stride: usize, force_full: bool) -> (bool, bool) {
        let covered: usize = self.ranges().iter().map(|r| r.end - r.start).sum();
        let fallback = !force_full
            && (self.fragmented
                || (self.count != 0
                    && (covered * 4 >= len * 3
                        || covered * stride + (self.count - 1) * WRITE_CALL_BYTES
                            >= len * stride)));
        let full = len != 0 && (force_full || fallback);
        if full {
            self.ranges[0] = UploadRange { start: 0, end: len };
            self.count = 1;
        }
        (full, fallback)
    }

    fn ranges(&self) -> &[UploadRange] {
        &self.ranges[..self.count]
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct FrameKey {
    identity: u64,
    poses: u64,
    colors: u64,
}

#[derive(Default)]
pub(crate) struct RetainedUploads {
    committed: Option<FrameKey>,
    pending: Option<FrameKey>,
    stamps: Vec<(u64, u64)>,
    poses: WritePlan,
    colors: WritePlan,
}

#[derive(Default)]
pub(crate) struct PreparedRetained {
    pub repacked: usize,
    pub poses_repacked: usize,
    pub colors_repacked: usize,
    pub full_upload: bool,
    pub fallback: bool,
}

impl RetainedUploads {
    /// A new buffer, layout, or non-retained submission invalidates GPU contents.
    #[cfg(target_arch = "wasm32")]
    pub fn invalidate(&mut self) {
        self.committed = None;
        self.pending = None;
    }

    #[cfg(target_arch = "wasm32")]
    pub fn prepare(
        &mut self,
        frame: &RenderFrame<'_>,
        colors: &mut Vec<ColorData>,
    ) -> PreparedRetained {
        self.prepare_records(
            &frame.snapshot().instances,
            FrameKey {
                identity: frame.identity(),
                poses: frame.pose_revision(),
                colors: frame.color_revision(),
            },
            frame.pose_changes(),
            frame.color_changes(),
            colors,
        )
    }

    /// Selects changed slots of both channels. `instances` supplies slot
    /// colors, which are packed here; changed pose slots upload directly from
    /// the frame's pose rows, so poses are never read or repacked.
    fn prepare_records(
        &mut self,
        instances: &[RenderInstance],
        key: FrameKey,
        pose_changes: &[u64],
        color_changes: &[u64],
        colors: &mut Vec<ColorData>,
    ) -> PreparedRetained {
        self.poses.clear();
        self.colors.clear();
        let previous = self.committed.take();
        self.pending = Some(key);
        let full = previous.is_none_or(|old| old.identity != key.identity);
        let pose_work = full || previous.is_none_or(|old| old.poses != key.poses);
        let color_work = full || previous.is_none_or(|old| old.colors != key.colors);
        let mut prepared = PreparedRetained::default();
        if full {
            colors.clear();
            self.stamps.clear();
            colors.reserve_exact(instances.len());
            self.stamps.reserve_exact(colors.capacity());
            for (index, source) in instances.iter().enumerate() {
                colors.push(ColorData::from(source));
                self.stamps
                    .push((pose_changes[index], color_changes[index]));
            }
            prepared.repacked = instances.len();
            prepared.poses_repacked = instances.len();
            prepared.colors_repacked = instances.len();
        } else {
            // Buffer capacity can outlive shrinking or expanding populations.
            // Existing slots retain GPU contents; only appended slots need both
            // channels initialized when the GPU allocation itself still fits.
            colors.truncate(instances.len());
            self.stamps.truncate(instances.len());
            self.stamps
                .reserve_exact(colors.capacity().saturating_sub(self.stamps.len()));
            if pose_work || color_work {
                for (index, stamps) in self.stamps.iter_mut().enumerate() {
                    let pose_changed = pose_work && stamps.0 != pose_changes[index];
                    let color_changed = color_work && stamps.1 != color_changes[index];
                    if pose_changed {
                        stamps.0 = pose_changes[index];
                        self.poses.changed(index, size_of::<PoseData>());
                        prepared.poses_repacked += 1;
                    }
                    if color_changed {
                        colors[index] = ColorData::from(&instances[index]);
                        stamps.1 = color_changes[index];
                        self.colors.changed(index, size_of::<ColorData>());
                        prepared.colors_repacked += 1;
                    }
                    prepared.repacked += usize::from(pose_changed || color_changed);
                }
            }
            for (index, source) in instances.iter().enumerate().skip(self.stamps.len()) {
                colors.push(ColorData::from(source));
                self.stamps
                    .push((pose_changes[index], color_changes[index]));
                self.poses.changed(index, size_of::<PoseData>());
                self.colors.changed(index, size_of::<ColorData>());
                prepared.repacked += 1;
                prepared.poses_repacked += 1;
                prepared.colors_repacked += 1;
            }
        }
        let (pose_full, pose_fallback) =
            self.poses
                .finish(instances.len(), size_of::<PoseData>(), full);
        let (color_full, color_fallback) =
            self.colors
                .finish(instances.len(), size_of::<ColorData>(), full);
        prepared.full_upload = pose_full || color_full;
        prepared.fallback = pose_fallback || color_fallback;
        prepared
    }

    pub fn pose_ranges(&self) -> &[UploadRange] {
        self.poses.ranges()
    }

    pub fn color_ranges(&self) -> &[UploadRange] {
        self.colors.ranges()
    }

    /// Commit only after every selected pose and color range reaches the queue.
    pub fn commit(&mut self) {
        self.committed = self.pending.take();
    }

    #[cfg(target_arch = "wasm32")]
    pub fn capacity_bytes(&self) -> usize {
        size_of::<Self>() + self.stamps.capacity() * size_of::<(u64, u64)>()
    }

    #[cfg(target_arch = "wasm32")]
    pub const fn range_scratch_bytes() -> usize {
        2 * size_of::<[UploadRange; MAX_WRITE_RANGES]>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{AssetLibrary, Mesh};
    use glam::{Mat4, Vec3};

    fn records(count: usize) -> Vec<RenderInstance> {
        let mut assets = AssetLibrary::default();
        let mesh = assets.insert_mesh(Mesh::cube()).unwrap();
        vec![
            RenderInstance {
                mesh,
                material: None,
                model: Mat4::IDENTITY,
                color: [1.0; 4],
                surface: engine_core::Surface::MATTE,
            };
            count
        ]
    }

    #[test]
    fn retained_channels_update_independently_across_skipped_frames() {
        let mut source = records(3_000);
        // Core's pose rows: rewritten exactly when a slot's pose stamp advances.
        let mut rows: Vec<PoseData> = source.iter().map(PoseData::from).collect();
        let mut pose_stamps = vec![1; source.len()];
        let mut color_stamps = vec![1; source.len()];
        let mut cache = RetainedUploads::default();
        let mut colors = Vec::new();
        let mut key = FrameKey {
            identity: 10,
            poses: 1,
            colors: 1,
        };
        let first = cache.prepare_records(&source, key, &pose_stamps, &color_stamps, &mut colors);
        assert!(first.full_upload);
        assert_eq!(
            (cache.pose_ranges()[0].start, cache.pose_ranges()[0].end),
            (0, source.len())
        );
        let mut gpu_poses = rows.clone();
        let mut gpu_colors = colors.clone();
        cache.commit();

        // Intermediate extractions were never rendered: revision equality,
        // not adjacency of frame sequence numbers, determines GPU dirtiness.
        source[2_900].model = Mat4::from_translation(Vec3::X);
        rows[2_900] = PoseData::from(&source[2_900]);
        pose_stamps[2_900] = 8;
        key.poses = 8;
        let change = cache.prepare_records(&source, key, &pose_stamps, &color_stamps, &mut colors);
        assert_eq!((change.poses_repacked, change.colors_repacked), (1, 0));
        assert!(cache.color_ranges().is_empty());
        assert!(!change.full_upload);
        for range in cache.pose_ranges() {
            gpu_poses[range.start..range.end].copy_from_slice(&rows[range.start..range.end]);
        }
        cache.commit();
        source[600].color = [0.2, 0.4, 0.6, 1.0];
        color_stamps[600] = 12;
        key.colors = 12;
        let change = cache.prepare_records(&source, key, &pose_stamps, &color_stamps, &mut colors);
        assert_eq!((change.poses_repacked, change.colors_repacked), (0, 1));
        assert!(cache.pose_ranges().is_empty());
        for range in cache.color_ranges() {
            gpu_colors[range.start..range.end].copy_from_slice(&colors[range.start..range.end]);
        }
        cache.commit();
        for (index, instance) in source.iter().enumerate() {
            assert_eq!(
                bytemuck::bytes_of(&gpu_poses[index]),
                bytemuck::bytes_of(&PoseData::from(instance))
            );
            assert_eq!(
                bytemuck::bytes_of(&gpu_colors[index]),
                bytemuck::bytes_of(&ColorData::from(instance))
            );
        }
        let unchanged =
            cache.prepare_records(&source, key, &pose_stamps, &color_stamps, &mut colors);
        assert_eq!(unchanged.repacked, 0);
        assert!(cache.pose_ranges().is_empty() && cache.color_ranges().is_empty());
    }

    #[test]
    fn retries_refresh_storage_but_reused_and_appended_slots_update_independently() {
        let mut source = records(8);
        let mut stamps = vec![1; source.len()];
        let mut colors = Vec::new();
        let mut cache = RetainedUploads::default();
        let mut key = FrameKey {
            identity: 1,
            poses: 1,
            colors: 1,
        };
        cache.prepare_records(&source, key, &stamps, &stamps, &mut colors);
        // Caller aborted after only part of the writes. Staging is not proof
        // of queued contents, so even an identical frame must write both streams.
        let retry = cache.prepare_records(&source, key, &stamps, &stamps, &mut colors);
        assert!(retry.full_upload);
        assert_eq!(
            (cache.pose_ranges()[0].end, cache.color_ranges()[0].end),
            (8, 8)
        );
        cache.commit();
        // Slot reuse advances both core stamps even if its payload bits match.
        stamps[2] = 2;
        key.poses = 2;
        key.colors = 2;
        let reuse = cache.prepare_records(&source, key, &stamps, &stamps, &mut colors);
        assert_eq!(reuse.repacked, 1);
        assert_eq!((reuse.poses_repacked, reuse.colors_repacked), (1, 1));
        assert_eq!(
            (cache.pose_ranges()[0].start, cache.pose_ranges()[0].end),
            (2, 3)
        );
        assert_eq!(
            (cache.color_ranges()[0].start, cache.color_ranges()[0].end),
            (2, 3)
        );
        cache.commit();
        source.push(source[0]);
        stamps.push(3);
        key.poses = 3;
        key.colors = 3;
        let appended = cache.prepare_records(&source, key, &stamps, &stamps, &mut colors);
        assert_eq!(appended.repacked, 1);
        assert_eq!(
            (cache.pose_ranges()[0].start, cache.pose_ranges()[0].end),
            (8, 9)
        );
        assert_eq!(
            (cache.color_ranges()[0].start, cache.color_ranges()[0].end),
            (8, 9)
        );
        cache.commit();
        key.identity = 2;
        let other_engine = cache.prepare_records(&source, key, &stamps, &stamps, &mut colors);
        assert!(other_engine.full_upload);
        cache.commit();
        cache.prepare_records(&[], key, &[], &[], &mut colors);
        assert!(cache.pose_ranges().is_empty() && cache.color_ranges().is_empty());
        assert!(colors.is_empty());
    }
}
