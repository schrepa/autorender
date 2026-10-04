//! Exact CPU upload reuse: the camera uniform is rewritten only when its
//! bytes change, and upload planners describe written regions as ranges
//! with the heuristics they share.

use crate::gpu_data::CameraUniform;

/// Clean bytes a range may span instead of another queue write.
pub(crate) const WRITE_CALL_BYTES: usize = 16 * 1024;

/// Whether a stream of `len` records with `changed` of them changed uploads
/// whole instead of by ranges: at least three quarters changed.
pub(crate) fn whole_stream(changed: usize, len: usize) -> bool {
    changed * 4 >= len * 3
}

#[derive(Clone, Copy, Default)]
pub(crate) struct UploadRange {
    pub start: usize,
    pub end: usize,
}

/// The camera uniform last accepted by the queue.
#[derive(Default)]
pub(crate) struct UploadCache {
    camera: Option<CameraUniform>,
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
}
