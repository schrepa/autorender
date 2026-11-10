use bevy_ecs::prelude::*;
use engine_core::{AssetLibrary, Cull, Error, Material, MaterialHandle, SurfaceSlot};
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

#[derive(Resource, Default)]
pub(crate) struct CoverSurface {
    pub(crate) material: Option<MaterialHandle>,
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

pub(crate) fn hold_surface(mut library: ResMut<AssetLibrary>, mut surface: ResMut<CoverSurface>) {
    if surface
        .material
        .is_some_and(|handle| library.material(handle).is_some())
    {
        return;
    }
    let made = cover_material().and_then(|material| library.insert_material(material));
    surface.material = made.ok();
}
