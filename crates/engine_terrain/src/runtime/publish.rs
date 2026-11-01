use super::*;
use engine_places::ground::{PlacedAsset, PlacedLayer, SurfaceMaps, TerrainEntry, TerrainTiles};
use engine_places::water::Hydrology;

#[derive(Default, PartialEq)]
pub(crate) struct Published {
    epoch: u64,
    revision: u64,
    drawn: Vec<(Entity, bool)>,
    reach: Vec<f32>,
    assets: ((u64, u64), u64),
}

impl Published {
    fn of(terrains: &Terrains, assets: &AssetLibrary, into: &mut Self) {
        into.epoch = terrains.epoch;
        into.revision = terrains.revision();
        into.drawn.clear();
        into.reach.clear();
        for state in terrains.entries.iter().filter(|state| state.drawn) {
            into.drawn.push((state.entity, state.around.is_some()));
            into.reach
                .extend(state.ranges.iter().map(|range| range * LOAD_REACH));
        }
        into.assets = (assets.revision(), assets.material_revision());
    }
}

fn maps_of(look: &SurfaceMaterial, assets: &AssetLibrary) -> Option<SurfaceMaps> {
    let handle = look.material?;
    let material = assets.material(handle)?;
    Some(SurfaceMaps {
        material: handle,
        base_color: material.base_color_texture().map(|map| map.texture),
        normal: material.normal_texture().map(|map| map.texture),
        metallic_roughness: material.metallic_roughness_texture().map(|map| map.texture),
    })
}

fn placed_layers(rules: &Biomes) -> Vec<PlacedLayer> {
    rules
        .placements
        .iter()
        .map(|layer| {
            let stands_on = (!layer.materials.is_empty()).then(|| {
                let mut set = [0_u64; 4];
                for id in &layer.materials {
                    if let Some(material) = rules.materials.iter().position(|known| known == id)
                        && material < 256
                    {
                        set[material / 64] |= 1 << (material % 64);
                    }
                }
                set
            });
            let total: f32 = layer.assets.iter().map(|choice| choice.weight).sum();
            let assets = layer
                .assets
                .iter()
                .filter_map(|choice| {
                    let asset = rules
                        .assets
                        .iter()
                        .position(|known| *known == choice.asset)?;
                    Some((
                        asset as u16,
                        choice.weight / engine_math::max(total, f32::MIN_POSITIVE),
                    ))
                })
                .collect();
            PlacedLayer {
                density: layer.density / 10_000.0,
                scale: layer.scale,
                stands_on,
                assets,
            }
        })
        .collect()
}

fn entry_of(state: &TerrainState, assets: &AssetLibrary) -> TerrainEntry {
    let surface = &state.terrain.surface;
    TerrainEntry {
        terrain: state.entity,
        origin: state.terrain.origin,
        pyramid: state.terrain.pyramid,
        cutout: state.around.as_ref().map(|around| around.cutout),
        surroundings: state.around.is_some(),
        steep: surface.rules.steep(),
        may_hole: state.terrain.layers.may_hole(),
        reach: state.ranges.iter().map(|range| range * LOAD_REACH).collect(),
        macro_map: None,
        materials: Arc::from(surface.materials.as_slice()),
        maps: surface
            .materials
            .iter()
            .map(|look| maps_of(look, assets))
            .collect(),
        placement: surface
            .assets
            .iter()
            .map(|asset| PlacedAsset {
                mesh: asset.mesh,
                material: asset.material,
            })
            .collect(),
        layers: Arc::from(placed_layers(surface.rules.authored())),
        tiles: state.store.faces().cloned().collect(),
    }
}

pub(crate) fn publish_tiles(
    terrains: Res<Terrains>,
    assets: Res<AssetLibrary>,
    mut tiles: ResMut<TerrainTiles>,
    mut published: Local<(Published, Published)>,
) {
    let (last, seen) = &mut *published;
    Published::of(&terrains, &assets, seen);
    if *seen == *last && !tiles.is_changed() {
        return;
    }
    *tiles = TerrainTiles {
        terrains: terrains
            .entries
            .iter()
            .filter(|state| state.drawn)
            .map(|state| entry_of(state, &assets))
            .collect(),
    };
    std::mem::swap(last, seen);
}

pub(crate) fn publish_hydrology(
    terrains: Res<Terrains>,
    mut hydrology: ResMut<Hydrology>,
    mut published: Local<Vec<(Entity, engine_core::Stamp)>>,
) {
    let held = terrains.entries.iter().filter_map(|state| {
        let (stamp, _) = state.hydrology.bake.as_ref()?;
        state.around.is_none().then_some((state.entity, *stamp))
    });
    if held.clone().eq(published.iter().copied()) && !hydrology.is_changed() {
        return;
    }
    published.clear();
    published.extend(held);
    hydrology.bakes = terrains
        .entries
        .iter()
        .filter(|state| state.around.is_none())
        .filter_map(|state| Some((state.entity, Arc::clone(&state.hydrology.bake.as_ref()?.1))))
        .collect();
}
