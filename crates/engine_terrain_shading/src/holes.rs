use bevy_ecs::prelude::*;
use engine_core::{AlphaMode, AssetLibrary, Error, Material, MaterialHandle, SurfaceSlot};
use engine_render::SlotModule;

use crate::shaders::terrain_shading::holes;

const HOLED_GROUND: &str = "holed_ground";
const HOLE_CUTOFF: f32 = 0.5;

#[derive(Resource, Default)]
pub(crate) struct HoledGround {
    pub(crate) material: Option<MaterialHandle>,
}

pub(crate) const SLOT_MODULE: SlotModule = SlotModule {
    slot: SurfaceSlot::Opacity.name(),
    name: HOLED_GROUND,
    module: holes::MODULE,
    maps: &[],
    values: &[],
    reach: 0.0,
};

fn holed_material() -> Result<Material, Error> {
    Material::new([1.0, 1.0, 1.0, 1.0])?
        .with_alpha_mode(AlphaMode::Mask {
            cutoff: HOLE_CUTOFF,
        })?
        .with_slot(SurfaceSlot::Opacity.name(), HOLED_GROUND)
}

pub(crate) fn hold_material(mut library: ResMut<AssetLibrary>, mut ground: ResMut<HoledGround>) {
    if ground
        .material
        .is_some_and(|handle| library.material(handle).is_some())
    {
        return;
    }
    let made = holed_material().and_then(|material| library.insert_material(material));
    ground.material = made.ok();
}
