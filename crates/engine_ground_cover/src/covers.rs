use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use engine_places::ground::{Cover, TerrainEntry, TerrainTiles};

use crate::layouts::{COVER_ENTRIES, CoverMaterial, CoverTerrain};
use crate::settings::{CoverLook, GroundCover};

const COVER_RISEN: f32 = 0.875;

const COVER_FILLED: f32 = 0.6;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CoverTerms {
    pub(crate) densest: f32,
    pub(crate) width: f32,
}

impl CoverTerms {
    pub(crate) fn span(self, pixels_per_meter: f32) -> f32 {
        self.width * pixels_per_meter
    }
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub(crate) struct CoverTable {
    pub(crate) terrains: Vec<CoverTerrain>,
    pub(crate) materials: Vec<CoverMaterial>,
    pub(crate) tallest: f32,
}

impl CoverTable {
    pub(crate) fn terms(&self, index: usize) -> CoverTerms {
        self.terrains
            .get(index)
            .map_or_else(CoverTerms::default, |terrain| CoverTerms {
                densest: terrain.densest,
                width: terrain.width,
            })
    }

    pub(crate) fn base(&self, index: usize) -> u32 {
        self.terrains.get(index).map_or(0, |terrain| terrain.base)
    }

    fn fill(&mut self, looks: &BTreeMap<String, CoverLook>, terrains: &[TerrainEntry]) {
        self.terrains.clear();
        self.materials.clear();
        for terrain in terrains {
            let room = COVER_ENTRIES as usize - self.materials.len();
            let surface = &terrain.materials;
            let kept = &surface[..surface.len().min(room)];
            let whole = kept.len() == surface.len();
            let terms = if whole {
                densest_cover(
                    kept.iter()
                        .filter_map(|material| looks.get(&*material.name)),
                )
            } else {
                CoverTerms::default()
            };
            self.terrains.push(CoverTerrain {
                base: if whole {
                    self.materials.len() as u32
                } else {
                    0
                },
                densest: terms.densest,
                width: terms.width,
                pad: 0,
            });
            self.materials.extend(kept.iter().map(|material| {
                looks
                    .get(&*material.name)
                    .map_or(CoverMaterial::default(), cover_material)
            }));
        }
        self.tallest = self
            .materials
            .iter()
            .flat_map(|entry| [entry.shape[1], entry.shape[2]])
            .fold(0.0, engine_core::math::max);
    }
}

pub(crate) fn write_covers(
    cover: Res<GroundCover>,
    tiles: Res<TerrainTiles>,
    mut table: ResMut<CoverTable>,
    mut shown: ResMut<Cover>,
    mut spare: Local<CoverTable>,
) {
    let inputs = cover.is_changed() || tiles.is_changed();
    if inputs {
        spare.fill(&cover.looks, &tiles.terrains);
        if *spare != *table {
            std::mem::swap(&mut *table, &mut *spare);
        }
    }
    if !(inputs || shown.is_changed()) {
        return;
    }
    shown.set_if_neq(Cover {
        enabled: cover.enabled,
        density: cover.density,
        widths: tiles
            .terrains
            .iter()
            .zip(&table.terrains)
            .map(|(terrain, row)| (terrain.terrain, row.width))
            .collect(),
    });
}

fn densest_cover<'a>(covers: impl IntoIterator<Item = &'a CoverLook>) -> CoverTerms {
    covers
        .into_iter()
        .fold(CoverTerms::default(), |terms, cover| {
            if cover.density > terms.densest {
                CoverTerms {
                    densest: cover.density,
                    width: cover.width,
                }
            } else {
                terms
            }
        })
}

fn cover_material(cover: &CoverLook) -> CoverMaterial {
    CoverMaterial {
        shape: [cover.density, cover.height[0], cover.height[1], cover.width],
        color: [cover.color[0], cover.color[1], cover.color[2], cover.sway],
        canopy: canopy(cover),
    }
}

fn canopy(cover: &CoverLook) -> [f32; 4] {
    let height = COVER_RISEN * 0.5 * (cover.height[0] + cover.height[1]);
    [
        cover.color[0],
        cover.color[1],
        cover.color[2],
        cover.density * height * cover.width * COVER_FILLED,
    ]
}
