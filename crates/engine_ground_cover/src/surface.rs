use bevy_ecs::prelude::*;
use engine_core::{
    Asset, AssetLibrary, AssetResidency, AssetUses, Cull, Error, Material, MaterialHandle,
    SurfaceSlot,
};
use engine_places::surface::{OCCLUSION, ROUGHNESS, TRANSMISSION};
use engine_render::SlotModule;

use crate::shaders::cover::surface;

const COVER_SURFACE: &str = "cover";
const COVER_MATERIAL: &str = "$.cover.material";

const COVER_SLOTS: [&str; 7] = [
    SurfaceSlot::BaseColor.name(),
    SurfaceSlot::Opacity.name(),
    SurfaceSlot::Normal.name(),
    ROUGHNESS,
    OCCLUSION,
    TRANSMISSION,
    SurfaceSlot::Kind.name(),
];

#[derive(Resource)]
pub(crate) struct CoverSurface {
    made: Option<Result<MaterialHandle, Error>>,
    pub(crate) held: Result<Option<MaterialHandle>, Error>,
    uses: AssetUses,
}

impl Default for CoverSurface {
    fn default() -> Self {
        Self {
            made: None,
            held: Ok(None),
            uses: AssetUses::default(),
        }
    }
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
        .made
        .as_ref()
        .and_then(|made| made.as_ref().ok().copied())
        .filter(|&handle| library.material(handle).is_some());
    let material = match kept {
        Some(handle) => Ok(handle),
        None => cover_material().and_then(|material| library.insert_material(material)),
    };
    let wanted: Vec<Asset> = material
        .as_ref()
        .ok()
        .copied()
        .map(Asset::Material)
        .into_iter()
        .collect();
    if surface.uses.assets() != wanted.as_slice() {
        surface.uses = AssetUses::new(&residency, wanted);
    }
    let held = residency.made_material(Some(&material), COVER_MATERIAL);
    let made = Some(material);
    if surface.made != made {
        surface.made = made;
    }
    if surface.held != held {
        surface.held = held;
    }
}
