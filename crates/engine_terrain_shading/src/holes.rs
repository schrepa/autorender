use bevy_ecs::prelude::*;
use engine_core::{
    AlphaMode, Asset, AssetLibrary, AssetResidency, AssetUses, Error, Material, MaterialHandle,
    SurfaceSlot,
};
use engine_render::SlotModule;

use crate::shaders::terrain_shading::holes;

const HOLED_GROUND: &str = "holed_ground";
const HOLE_CUTOFF: f32 = 0.5;
const NOT_IN_LIBRARY: &str = "the holed ground material is not in the asset library";
const NOT_RESIDENT: &str = "the holed ground material is not resident yet";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum HoledState {
    #[default]
    Unmade,
    Waiting(MaterialHandle),
    Resident(MaterialHandle),
}

impl HoledState {
    fn material(self) -> Option<MaterialHandle> {
        match self {
            Self::Unmade => None,
            Self::Waiting(handle) | Self::Resident(handle) => Some(handle),
        }
    }

    pub(crate) fn held(self) -> Result<MaterialHandle, &'static str> {
        match self {
            Self::Unmade => Err(NOT_IN_LIBRARY),
            Self::Waiting(_) => Err(NOT_RESIDENT),
            Self::Resident(handle) => Ok(handle),
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct HoledGround {
    pub(crate) state: HoledState,
    uses: AssetUses,
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
        .state
        .material()
        .filter(|&handle| library.material(handle).is_some());
    let material = match kept {
        Some(handle) => Some(handle),
        None => holed_material()
            .and_then(|material| library.insert_material(material))
            .ok(),
    };
    let wanted: Vec<Asset> = material.map(Asset::Material).into_iter().collect();
    if ground.uses.assets() != wanted.as_slice() {
        ground.uses = AssetUses::new(&residency, wanted);
    }
    let state = match material {
        None => HoledState::Unmade,
        Some(handle) if residency.material_resident(handle) => HoledState::Resident(handle),
        Some(handle) => HoledState::Waiting(handle),
    };
    if ground.state != state {
        ground.state = state;
    }
}
