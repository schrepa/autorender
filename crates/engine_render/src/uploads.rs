use crate::gpu_data::CameraUniform;

pub const WRITE_CALL_BYTES: usize = 16 * 1024;

pub fn whole_stream(changed: usize, len: usize) -> bool {
    changed * 4 >= len * 3
}

#[derive(Clone, Copy, Default)]
pub struct UploadRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Default)]
pub struct UploadCache {
    camera: Option<CameraUniform>,
}

impl UploadCache {
    pub fn camera_needs_write(&self, camera: &CameraUniform) -> bool {
        self.camera
            .as_ref()
            .is_none_or(|previous| bytemuck::bytes_of(previous) != bytemuck::bytes_of(camera))
    }

    pub fn commit_camera(&mut self, camera: CameraUniform) {
        self.camera = Some(camera);
    }
}
