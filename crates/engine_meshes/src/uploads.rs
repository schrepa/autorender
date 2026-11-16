use bytemuck::{Pod, Zeroable};
use engine_core::{AssetLibrary, DrawnTable, Material};
use engine_places::surface::SurfaceValues;
use engine_render::{pack_color, pack_surface};

pub(crate) type PoseData = engine_core::PoseRows;

#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct ColorData {
    pub color: [u32; 2],
    pub surface: u32,
    pub pad: u32,
}

fn tinted(color: [f32; 4], tint: [f32; 4]) -> [f32; 4] {
    let mut tinted = color;
    for (channel, factor) in tinted.iter_mut().zip(tint) {
        *channel *= if factor.is_finite() {
            engine_core::math::clamp(factor, 0.0, 1.0)
        } else {
            1.0
        };
    }
    tinted
}

pub(crate) fn restyled_slots(
    table: &DrawnTable,
    assets: &AssetLibrary,
    since: u64,
    slots: &mut Vec<u32>,
) {
    slots.extend((0..table.records().len() as u32).filter(|&slot| {
        table
            .drawn(slot)
            .and_then(|drawn| drawn.material)
            .is_some_and(|handle| {
                assets.revision_of(handle).is_none_or(|revision| revision > since)
            })
    }));
}

pub(crate) fn color_data(table: &DrawnTable, assets: &AssetLibrary, slot: usize) -> ColorData {
    let material = table
        .drawn(slot as u32)
        .and_then(|drawn| drawn.material)
        .and_then(|handle| assets.material(handle));
    let tint = table
        .records()
        .get(slot)
        .map_or([1.0; 4], |record| record.tint);
    ColorData {
        color: pack_color(tinted(material.map_or([1.0; 4], Material::color), tint)),
        surface: pack_surface(
            SurfaceValues::read(
                |name| material.and_then(|material| material.surface_value(name)),
                |name| material.and_then(|material| material.value(name)),
            )
            .units(),
            material.map_or(0.0, Material::emission),
        ),
        pad: 0,
    }
}
