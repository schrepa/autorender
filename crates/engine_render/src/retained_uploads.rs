//! Revision-driven uploads of retained frames' slot poses and colors.
//!
//! Poses need no packing here: core keeps every slot's pose in the upload
//! layout (`RenderFrame::pose_rows`) and this cache only selects changed slots.

#[cfg(target_arch = "wasm32")]
use engine_core::RenderFrame;
use engine_core::RenderInstance;

use crate::gpu_data::{ColorData, PoseData};
use crate::uploads::{UploadRange, WRITE_CALL_BYTES, whole_stream};

const MAX_WRITE_RANGES: usize = 32;

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
                    && (whole_stream(covered, len)
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
