use bevy_ecs::prelude::*;
use engine_core::{
    AlphaMode, Asset, AssetLibrary, AssetResidency, AssetUses, Error, Material, MaterialHandle,
    Request, SurfaceSlot,
};
use engine_render::SlotModule;

use crate::shaders::terrain_shading::holes;

const HOLED_GROUND: &str = "holed_ground";
const HOLE_CUTOFF: f32 = 0.5;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) enum HoledState {
    #[default]
    Unmade,
    Waiting(MaterialHandle),
    Resident(MaterialHandle),
    Failed(MaterialHandle, Error),
    Refused(Error),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Unheld<'a> {
    Waiting,
    Refused(&'a Error),
}

impl HoledState {
    fn material(&self) -> Option<MaterialHandle> {
        match self {
            Self::Unmade | Self::Refused(_) => None,
            Self::Waiting(handle) | Self::Resident(handle) | Self::Failed(handle, _) => {
                Some(*handle)
            }
        }
    }

    pub(crate) fn held(&self) -> Result<MaterialHandle, Unheld<'_>> {
        match self {
            Self::Unmade | Self::Waiting(_) => Err(Unheld::Waiting),
            Self::Resident(handle) => Ok(*handle),
            Self::Failed(_, error) | Self::Refused(error) => Err(Unheld::Refused(error)),
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
    let state = match material {
        Err(error) => HoledState::Refused(error),
        Ok(handle) if residency.material_resident(handle) => HoledState::Resident(handle),
        Ok(handle) => match residency.state(Asset::Material(handle)) {
            Some(Request::Failed(error)) => HoledState::Failed(handle, error.clone()),
            Some(Request::Asked(_) | Request::Ready(_) | Request::Missing) | None => {
                HoledState::Waiting(handle)
            }
        },
    };
    if ground.state != state {
        ground.state = state;
    }
}
