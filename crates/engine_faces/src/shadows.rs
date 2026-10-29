#[derive(Clone, Copy, Debug)]
pub struct KeptRefresh {
    pub planes: [[f32; 4]; 6],
    pub texel: f32,
}

pub trait ShadowCastersFace {
    fn region_views(&self) -> &[(usize, KeptRefresh)];
}
