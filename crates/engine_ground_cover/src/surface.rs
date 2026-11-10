use bevy_ecs::prelude::*;
use engine_core::{
    Asset, AssetLibrary, AssetResidency, AssetUses, Cull, Error, Material, MaterialHandle,
    SurfaceSlot,
};
use engine_places::surface::{OCCLUSION, ROUGHNESS, TRANSMISSION};
use engine_render::SlotModule;

use crate::shaders::cover::surface;

const COVER_SURFACE: &str = "cover";

const COVER_SLOTS: [&str; 7] = [
    SurfaceSlot::BaseColor.name(),
    SurfaceSlot::Opacity.name(),
    SurfaceSlot::Normal.name(),
    ROUGHNESS,
    OCCLUSION,
    TRANSMISSION,
    SurfaceSlot::Kind.name(),
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SurfaceState {
    #[default]
    Unmade,
    Waiting(MaterialHandle),
    Resident(MaterialHandle),
}

impl SurfaceState {
    fn material(self) -> Option<MaterialHandle> {
        match self {
            Self::Unmade => None,
            Self::Waiting(handle) | Self::Resident(handle) => Some(handle),
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct CoverSurface {
    pub(crate) state: SurfaceState,
    uses: AssetUses,
}

pub(crate) fn slot_modules() -> impl Iterator<Item = SlotModule> {
    COVER_SLOTS.into_iter().map(|slot| SlotModule {
        slot,
        name: COVER_SURFACE,
        module: surface::MODULE,
        maps: &[],
        values: &[],
        reach: 0.0,
    })
}

fn cover_material() -> Result<Material, Error> {
    COVER_SLOTS.into_iter().try_fold(
        Material::new([1.0, 1.0, 1.0, 1.0])?.with_cull(Cull::None),
        |material, slot| material.with_slot(slot, COVER_SURFACE),
    )
}

pub(crate) fn hold_surface(
    mut library: ResMut<AssetLibrary>,
    residency: Res<AssetResidency>,
    mut surface: ResMut<CoverSurface>,
) {
    let kept = surface
        .state
        .material()
        .filter(|&handle| library.material(handle).is_some());
    let material = match kept {
        Some(handle) => Some(handle),
        None => cover_material()
            .and_then(|material| library.insert_material(material))
            .ok(),
    };
    let wanted: Vec<Asset> = material.map(Asset::Material).into_iter().collect();
    if surface.uses.assets() != wanted.as_slice() {
        surface.uses = AssetUses::new(&residency, wanted);
    }
    let state = match material {
        None => SurfaceState::Unmade,
        Some(handle) if residency.material_resident(handle) => SurfaceState::Resident(handle),
        Some(handle) => SurfaceState::Waiting(handle),
    };
    if surface.state != state {
        surface.state = state;
    }
}
