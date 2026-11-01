use bevy_ecs::prelude::{Entity, Resource};
use engine_core::AssetLibrary;
use engine_places::ground::{PlacedLayer, TerrainEntry};

use crate::groups::Resolved;
use crate::layouts::PlacedFar;

#[derive(Resource, Default)]
pub(crate) struct FarCoverTable {
    pub(crate) bases: Vec<u32>,
    pub(crate) entries: Vec<PlacedFar>,
}

#[derive(Clone, Copy, Default)]
struct PlacedSums {
    covered: f32,
    color: [f32; 3],
    first: f32,
    last: f32,
}

impl PlacedSums {
    fn add(&mut self, other: &Self) {
        self.covered += other.covered;
        for (sum, part) in self.color.iter_mut().zip(other.color) {
            *sum += part;
        }
        self.first += other.first;
        self.last += other.last;
    }
}

impl FarCoverTable {
    pub(crate) fn fill(
        &mut self,
        terrains: &[TerrainEntry],
        resolved: &[(Entity, Vec<Option<Resolved<'_>>>)],
        library: &AssetLibrary,
    ) {
        self.bases.clear();
        self.entries.clear();
        let mut sums = Vec::new();
        for (terrain, (_, assets)) in terrains.iter().zip(resolved) {
            self.bases.push(self.entries.len() as u32);
            sums.clear();
            sums.extend(
                terrain
                    .layers
                    .iter()
                    .map(|layer| placed_sums(assets, library, layer)),
            );
            self.entries
                .extend((0..terrain.materials.len()).map(|material| {
                    let (placed, size) = placed_terms(&terrain.layers, &sums, material);
                    PlacedFar { placed, size }
                }));
        }
    }
}

fn octaves(meters: f32) -> f32 {
    engine_math::ln(engine_math::max(meters, 1.0e-6)) * std::f32::consts::LOG2_E
}

fn stands_on(layer: &PlacedLayer, material: usize) -> bool {
    layer
        .stands_on
        .is_none_or(|set| material < 256 && set[material / 64] & (1 << (material % 64)) != 0)
}

fn placed_sums(
    assets: &[Option<Resolved<'_>>],
    library: &AssetLibrary,
    layer: &PlacedLayer,
) -> PlacedSums {
    let [smallest, largest] = layer.scale;
    let squared = (smallest * smallest + smallest * largest + largest * largest) / 3.0;
    let mut sums = PlacedSums::default();
    let (mut chosen, mut first, mut last) = (0.0, 0.0, 0.0);
    for &(asset, share) in &layer.assets {
        let Some(Some(asset)) = assets.get(usize::from(asset)) else {
            continue;
        };
        let (area, diameter, color) = placed_asset(asset, library);
        let covered = layer.density * share * area * squared;
        sums.covered += covered;
        for (sum, channel) in sums.color.iter_mut().zip(color) {
            *sum += covered * channel;
        }
        chosen += share;
        first += share * octaves(diameter * largest);
        last += share * octaves(diameter * smallest);
    }
    if chosen > 0.0 {
        sums.first = sums.covered * first / chosen;
        sums.last = sums.covered * last / chosen;
    }
    sums
}

fn placed_asset(asset: &Resolved<'_>, library: &AssetLibrary) -> (f32, f32, [f32; 3]) {
    let bounds = asset.mesh.mesh.bounds();
    let extent = bounds.max() - bounds.min();
    let area = std::f32::consts::FRAC_PI_4 * extent.x * extent.z;
    let (low, high) = asset
        .parts
        .iter()
        .filter_map(|part| part.meshes.first())
        .chain(&asset.shadow)
        .map(|mesh| mesh.mesh.bounds())
        .fold((bounds.min(), bounds.max()), |(low, high), held| {
            (low.min(held.min()), high.max(held.max()))
        });
    (area, (high - low).length(), placed_color(asset, library))
}

fn placed_color(asset: &Resolved<'_>, library: &AssetLibrary) -> [f32; 3] {
    let material = asset.material.material;
    let factor = material.color();
    let mean = material
        .base_color_texture()
        .and_then(|map| library.texture(map.texture))
        .and_then(|texture| texture.summary())
        .map_or([1.0; 3], |summary| summary.color);
    [
        factor[0] * mean[0],
        factor[1] * mean[1],
        factor[2] * mean[2],
    ]
}

fn placed_terms(
    layers: &[PlacedLayer],
    sums: &[PlacedSums],
    material: usize,
) -> ([f32; 4], [f32; 4]) {
    let mut total = PlacedSums::default();
    for (layer, held) in layers.iter().zip(sums) {
        if stands_on(layer, material) {
            total.add(held);
        }
    }
    let covered = total.covered;
    if covered <= 0.0 {
        return ([0.0; 4], [0.0; 4]);
    }
    (
        [
            total.color[0] / covered,
            total.color[1] / covered,
            total.color[2] / covered,
            engine_math::min(covered, 1.0),
        ],
        [total.first / covered, total.last / covered, 0.0, 0.0],
    )
}
