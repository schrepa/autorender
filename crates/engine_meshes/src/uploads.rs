use bytemuck::{Pod, Zeroable};
use engine_core::Stamp;
use engine_render::{pack_color, pack_surface};

use crate::retained::{Instance, PoseRows, RetainedRender};

pub(crate) type PoseData = PoseRows;

#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct ColorData {
    pub color: [u32; 2],
    pub surface: u32,
    pub flags: u32,
}

fn color_data(instance: &Instance) -> ColorData {
    ColorData {
        color: pack_color(instance.color),
        surface: pack_surface(instance.surface),
        flags: 0,
    }
}

pub(crate) const WRITE_CALL_BYTES: usize = 16 * 1024;

pub(crate) fn whole_stream(changed: usize, len: usize) -> bool {
    changed * 4 >= len * 3
}

#[derive(Clone, Copy, Default)]
pub(crate) struct UploadRange {
    pub start: usize,
    pub end: usize,
}

const MAX_RANGES: usize = 64;

pub(crate) fn changed_ranges(
    stamps: &[u64],
    since: u64,
    stride: usize,
    ranges: &mut Vec<UploadRange>,
) -> bool {
    ranges.clear();
    let gap = WRITE_CALL_BYTES.div_ceil(stride);
    let mut covered = 0;
    for (index, &stamp) in stamps.iter().enumerate() {
        if stamp <= since {
            continue;
        }
        match ranges.last_mut() {
            Some(range) if index - range.end <= gap => {
                covered += index + 1 - range.end;
                range.end = index + 1;
            }
            _ => {
                if ranges.len() == MAX_RANGES {
                    return full(stamps.len(), ranges);
                }
                ranges.push(UploadRange {
                    start: index,
                    end: index + 1,
                });
                covered += 1;
            }
        }
    }
    if whole_stream(covered, stamps.len()) && !ranges.is_empty() {
        return full(stamps.len(), ranges);
    }
    true
}

pub(crate) fn listed_ranges(
    nodes: &[u32],
    len: usize,
    stride: usize,
    bits: &mut Vec<u64>,
    ranges: &mut Vec<UploadRange>,
) -> bool {
    ranges.clear();
    let words = len.div_ceil(64);
    if bits.len() < words {
        bits.resize(words, 0);
    }
    for &node in nodes {
        let node = node as usize;
        debug_assert!(node < len, "changed_node_in_array");
        bits[node / 64] |= 1 << (node % 64);
    }
    let gap = WRITE_CALL_BYTES.div_ceil(stride);
    let mut covered = 0;
    let mut partial = true;
    'words: for (word, &value) in bits[..words].iter().enumerate() {
        let mut value = value;
        while value != 0 {
            let first = value.trailing_zeros() as usize;
            let run = (value >> first).trailing_ones() as usize;
            let (start, end) = (word * 64 + first, word * 64 + first + run);
            match ranges.last_mut() {
                Some(range) if start - range.end <= gap => {
                    covered += end - range.end;
                    range.end = end;
                }
                _ => {
                    if ranges.len() == MAX_RANGES {
                        partial = false;
                        break 'words;
                    }
                    ranges.push(UploadRange { start, end });
                    covered += end - start;
                }
            }
            value = if first + run == 64 {
                0
            } else {
                value & !(((1 << run) - 1) << first)
            };
        }
    }
    for &node in nodes {
        bits[node as usize / 64] = 0;
    }
    if !partial || (whole_stream(covered, len) && !ranges.is_empty()) {
        return full(len, ranges);
    }
    true
}

fn full(len: usize, ranges: &mut Vec<UploadRange>) -> bool {
    ranges.clear();
    if len != 0 {
        ranges.push(UploadRange { start: 0, end: len });
    }
    false
}

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
    identity: Stamp,
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
    pub fn capacity_bytes(&self) -> usize {
        size_of::<Self>() + self.stamps.capacity() * size_of::<(u64, u64)>()
    }

    pub const fn range_scratch_bytes() -> usize {
        2 * size_of::<[UploadRange; MAX_WRITE_RANGES]>()
    }

    pub fn invalidate(&mut self) {
        self.committed = None;
        self.pending = None;
    }

    pub fn prepare(
        &mut self,
        retained: &RetainedRender,
        colors: &mut Vec<ColorData>,
    ) -> PreparedRetained {
        self.prepare_records(
            &retained.snapshot().instances,
            FrameKey {
                identity: retained.identity(),
                poses: retained.pose_revision(),
                colors: retained.color_revision(),
            },
            retained.pose_changes(),
            retained.color_changes(),
            colors,
        )
    }

    fn prepare_records(
        &mut self,
        instances: &[Instance],
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
                colors.push(color_data(source));
                self.stamps
                    .push((pose_changes[index], color_changes[index]));
            }
            prepared.repacked = instances.len();
            prepared.poses_repacked = instances.len();
            prepared.colors_repacked = instances.len();
        } else {
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
                        colors[index] = color_data(&instances[index]);
                        stamps.1 = color_changes[index];
                        self.colors.changed(index, size_of::<ColorData>());
                        prepared.colors_repacked += 1;
                    }
                    prepared.repacked += usize::from(pose_changed || color_changed);
                }
            }
            for (index, source) in instances.iter().enumerate().skip(self.stamps.len()) {
                colors.push(color_data(source));
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

    pub fn commit(&mut self) {
        self.committed = self.pending.take();
    }
}
