pub const NOON_LIGHT: f32 = 1.1;
pub const ADAPTATION_POWER: f32 = 0.75;
pub const ADAPTATION_MOST: f32 = 16.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EnvironmentPlan {
    pub cubes: u32,
    pub most_cubes: u32,
    pub lost: bool,
    pub sky_changed: bool,
}

pub trait EnvironmentFace {
    fn plan(&self) -> EnvironmentPlan;

    fn size(&self) -> u32;

    fn radiance_face(&self, face: usize) -> &wgpu::TextureView;

    fn finish(&self, encoder: &mut wgpu::CommandEncoder, cube: u32);
}
