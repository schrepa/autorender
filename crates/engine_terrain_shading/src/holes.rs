use bevy_ecs::prelude::*;
use engine_core::{
    AlphaMode, Asset, AssetLibrary, AssetResidency, AssetUses, Error, Material, MaterialHandle,
    SurfaceSlot,
};
use engine_render::SlotModule;

use crate::shaders::terrain_shading::holes;

const HOLED_GROUND: &str = "holed_ground";
const HOLED_MATERIAL: &str = "$.holed_ground.material";
const HOLE_CUTOFF: f32 = 0.5;

#[derive(Resource)]
pub(crate) struct HoledGround {
    made: Option<Result<MaterialHandle, Error>>,
    pub(crate) held: Result<Option<MaterialHandle>, Error>,
    uses: AssetUses,
}

impl Default for HoledGround {
    fn default() -> Self {
        Self {
            made: None,
            held: Ok(None),
            uses: AssetUses::default(),
        }
    }
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

pub(crate) fn hold_material(
    mut library: ResMut<AssetLibrary>,
    residency: Res<AssetResidency>,
    mut ground: ResMut<HoledGround>,
) {
    let kept = ground
        .made
        .as_ref()
        .and_then(|made| made.as_ref().ok().copied())
        .filter(|&handle| library.material(handle).is_some());
    let material = match kept {
        Some(handle) => Ok(handle),
        None => holed_material().and_then(|material| library.insert_material(material)),
    };
    let wanted: Vec<Asset> = material
        .as_ref()
        .ok()
        .copied()
        .map(Asset::Material)
        .into_iter()
        .collect();
    if ground.uses.assets() != wanted.as_slice() {
        ground.uses = AssetUses::new(&residency, wanted);
    }
    let held = residency.made_material(Some(&material), HOLED_MATERIAL);
    let made = Some(material);
    if ground.made != made {
        ground.made = made;
    }
    if ground.held != held {
        ground.held = held;
    }
}
