use glam::{Mat4, Vec3};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_VALUE: AtomicU64 = AtomicU64::new(1);
const BLOCK: u64 = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Revision(NonZeroU64);

#[derive(Clone, Copy)]
pub struct GlobalTransform {
    matrix: Mat4,
    valid: bool,
}

impl GlobalTransform {
    pub fn matrix(&self) -> Option<Mat4> {
        self.valid.then_some(self.matrix)
    }

    pub fn translation(&self) -> Option<Vec3> {
        self.matrix().map(|matrix| matrix.w_axis.truncate())
    }

    pub(crate) fn read_value(matrix: Option<Mat4>) -> Self {
        Self {
            matrix: matrix.unwrap_or(Mat4::IDENTITY),
            valid: matrix.is_some(),
        }
    }
}

#[derive(Default)]
pub(crate) struct ValueIds {
    next: u64,
    end: u64,
}

impl ValueIds {
    pub(crate) fn next(&mut self) -> Revision {
        if self.next == self.end {
            let start = NEXT_VALUE
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                    next.checked_add(BLOCK)
                })
                .expect("value_ids_left");
            self.next = start;
            self.end = start + BLOCK;
        }
        let value = self.next;
        self.next += 1;
        Revision(NonZeroU64::new(value).expect("value_ids_start_at_one"))
    }
}
