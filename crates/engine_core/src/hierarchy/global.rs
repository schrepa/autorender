use glam::{Mat4, Vec3};

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
