//! Exact CPU upload reuse: the camera uniform is rewritten only when its
//! bytes change, and upload planners describe written regions as ranges.

use crate::gpu_data::CameraUniform;

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

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

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
}
