pub const ROUGHNESS: &str = "roughness";
pub const METALNESS: &str = "metalness";
pub const OCCLUSION: &str = "occlusion";
pub const TRANSMISSION: &str = "transmission";
pub const OCCLUSION_STRENGTH: &str = "occlusion_strength";
pub const METALLIC_ROUGHNESS_MAP: &str = "metallic_roughness";
pub const OCCLUSION_MAP: &str = "occlusion";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceValues {
    pub roughness: f32,
    pub metalness: f32,
    pub occlusion_strength: f32,
    pub transmission: f32,
}

impl SurfaceValues {
    pub const MATTE: Self = Self {
        roughness: 0.8,
        metalness: 0.0,
        occlusion_strength: 1.0,
        transmission: 0.0,
    };

    pub fn read(
        surface: impl Fn(&str) -> Option<f32>,
        value: impl Fn(&str) -> Option<f32>,
    ) -> Self {
        Self {
            roughness: surface(ROUGHNESS).unwrap_or(Self::MATTE.roughness),
            metalness: surface(METALNESS).unwrap_or(Self::MATTE.metalness),
            occlusion_strength: value(OCCLUSION_STRENGTH).unwrap_or(Self::MATTE.occlusion_strength),
            transmission: value(TRANSMISSION).unwrap_or(Self::MATTE.transmission),
        }
    }

    pub const fn units(self) -> [f32; 2] {
        [self.roughness, self.metalness]
    }
}
