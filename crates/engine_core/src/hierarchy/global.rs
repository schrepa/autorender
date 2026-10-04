use glam::{Mat4, Vec3};
use std::sync::atomic::{AtomicU64, Ordering};

// IDs belong to derived matrix values, not entities or Bevy change ticks.
// Never recycle an ID. Exhaustion uses an explicitly uncacheable sentinel.
static NEXT_VALUE: AtomicU64 = AtomicU64::new(1);
const UNSTAMPED: u64 = u64::MAX;
const BLOCK: u64 = 1024;

/// Read-only authoritative local-to-world pose from [`crate::GlobalTransforms`].
/// The full matrix preserves shear; invalid admitted poses have no matrix.
#[derive(Clone, Copy)]
pub struct GlobalTransform {
    matrix: Mat4,
    value: u64,
    valid: bool,
}

impl Default for GlobalTransform {
    fn default() -> Self {
        Self {
            matrix: Mat4::IDENTITY,
            value: 0,
            valid: false,
        }
    }
}

impl std::fmt::Debug for GlobalTransform {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("GlobalTransform")
            .field(&self.matrix())
            .finish()
    }
}

impl GlobalTransform {
    /// `None` means propagation has not run or this node cannot be transformed.
    pub fn matrix(&self) -> Option<Mat4> {
        self.valid.then_some(self.matrix)
    }

    pub fn translation(&self) -> Option<Vec3> {
        self.matrix().map(|matrix| matrix.w_axis.truncate())
    }

    /// Output revision; u64::MAX is uncacheable after stamp exhaustion.
    pub fn revision(&self) -> u64 {
        self.value
    }

    pub(crate) fn read_value(matrix: Option<Mat4>, revision: u64) -> Self {
        Self {
            matrix: matrix.unwrap_or(Mat4::IDENTITY),
            value: revision,
            valid: matrix.is_some(),
        }
    }
}

/// Small batches avoid a process-wide atomic operation for every changed node.
#[derive(Default)]
pub(crate) struct ValueIds {
    next: u64,
    end: u64,
}

impl ValueIds {
    pub(crate) fn next(&mut self) -> u64 {
        self.next_from(&NEXT_VALUE)
    }

    fn next_from(&mut self, source: &AtomicU64) -> u64 {
        if self.next == self.end {
            let Ok(start) = source.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(BLOCK)
            }) else {
                return UNSTAMPED;
            };
            self.next = start;
            self.end = start + BLOCK;
        }
        let value = self.next;
        self.next += 1;
        value
    }
}
