use super::ground_face::TerrainGround;
use super::*;
use engine_places_simulation::ground::Ground;
use engine_places_world::ground::{
    PlacedAsset, PlacedLayer, SurfaceMaps, TerrainEntry, TerrainTiles,
};
use engine_core::Spawner;
use engine_places_world::water::Hydrology;

#[derive(Component)]
pub(crate) struct SurroundingsGround;

fn sample_offset(state: &TerrainState) -> Vec2 {
    state
        .surface
        .as_ref()
        .map_or(Vec2::ZERO, |(layers, _)| layers.offset)
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

fn relief_of(materials: &[SurfaceMaterial], assets: &AssetLibrary) -> u128 {
    let mut bytes = Vec::with_capacity(32 * materials.len());
    for look in materials {
        for value in [look.depth, look.tiling, look.height_contrast] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.push(match look.projection {
            SurfaceProjection::Top => 0,
            SurfaceProjection::Biplanar => 1,
        });
        let map = look.material.and_then(|handle| {
            let material = assets.material(handle)?;
            Some(material.metallic_roughness_texture()?.texture)
        });
        let content = map
            .and_then(|handle| assets.texture(handle))
            .map_or(0, |texture| texture.identity(content_hash));
        bytes.extend_from_slice(&content.to_le_bytes());
    }
    content_hash(&bytes)
}

fn solids_of<'a>(
    (terrains, state): (&'a Terrains, &'a TerrainState),
    assets: &'a AssetLibrary,
) -> impl Iterator<Item = Option<Solid>> + 'a {
    let owner = terrains.terrain(state.entity).unwrap_or(&state.terrain);
    owner
        .surface
        .assets
        .iter()
        .map(move |asset| asset.solid(assets))
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

fn entry_entity(state: &TerrainState) -> Entity {
    state
        .around
        .as_ref()
        .and_then(|around| around.entity)
        .unwrap_or(state.entity)
}

fn entry_of(
    (terrains, state): (&Terrains, &TerrainState),
    settings: &TerrainSettings,
    assets: &AssetLibrary,
) -> TerrainEntry {
    let surface = &state.terrain.surface;
    TerrainEntry {
        terrain: entry_entity(state),
        origin: state.terrain.origin,
        pyramid: state.terrain.pyramid,
        cutout: state.around.as_ref().map(|around| around.cutout),
        surroundings: state.around.is_some(),
        steep: surface.rules.steep(),
        may_hole: state.current.may_hole(),
        reach: state
            .ranges
            .iter()
            .map(|range| range * LOAD_REACH)
            .collect(),
        recipe: state.recipe.as_ref().map(|recipe| recipe.key.hash()),
        edits: state.terrain.edits.iter().map(|edit| edit.stamp).collect(),
        solids: solids_of((terrains, state), assets).collect(),
        relief: relief_of(&surface.materials, assets),
        pinned_levels: settings.pinned_levels,
        pixel_error: settings.pixel_error,
        level_errors: Arc::from(state.store.errors()),
        surface_program: state.programmed().is_some(),
        baked: (state.around.is_none() && terrains.bake_asked(state.entity))
            .then(|| terrains.layers(state.entity))
            .flatten(),
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
        tiles: Arc::clone(state.store.faces()),
        changes: Arc::clone(state.store.changes()),
        sample_offset: sample_offset(state),
        kept: Arc::clone(state.store.kept_faces()),
    }
}

fn assets_hold(
    (terrains, state): (&Terrains, &TerrainState),
    entry: &TerrainEntry,
    assets: &AssetLibrary,
) -> bool {
    let surface = &state.terrain.surface;
    solids_of((terrains, state), assets).eq(entry.solids.iter().cloned())
        && surface
            .materials
            .iter()
            .map(|look| maps_of(look, assets))
            .eq(entry.maps.iter().copied())
        && relief_of(&surface.materials, assets) == entry.relief
}

fn assets_moved(terrains: &Terrains, tiles: &TerrainTiles, assets: &AssetLibrary) -> bool {
    terrains
        .entries
        .iter()
        .filter(|state| state.drawn)
        .any(|state| {
            let entity = entry_entity(state);
            tiles
                .terrains
                .iter()
                .find(|entry| entry.terrain == entity)
                .is_some_and(|entry| !assets_hold((terrains, state), entry, assets))
        })
}

fn same_bakes(
    held: &[(Entity, Arc<crate::HydrologyBake>)],
    next: &[(Entity, Arc<crate::HydrologyBake>)],
) -> bool {
    held.len() == next.len()
        && held
            .iter()
            .zip(next)
            .all(|((a, held), (b, next))| a == b && Arc::ptr_eq(held, next))
}

pub(crate) fn publish_terrains(
    (settings, assets): (Res<TerrainSettings>, Res<AssetLibrary>),
    mut terrains: ResMut<Terrains>,
    (mut tiles, mut ground): (ResMut<TerrainTiles>, ResMut<Ground>),
    (mut hydrology, mut programs): (ResMut<Hydrology>, ResMut<SurfacePrograms>),
    mut spawner: Spawner<TerrainBrick>,
) {
    for entity in std::mem::take(&mut terrains.retired.entities) {
        spawner.despawn(entity);
    }
    let mut changed = std::mem::take(&mut terrains.changed);
    let mut reshaped = false;
    for state in &mut terrains.entries {
        if let Some(around) = state.around.as_mut()
            && around.entity.is_none()
        {
            around.entity = Some(spawner.spawn(SurroundingsGround));
            state.changed = true;
        }
        reshaped |= state.store.take_reshaped();
        changed |= std::mem::take(&mut state.changed);
        changed |= state.store.settle_tiles(settings.tile_feed as usize);
        changed |= state.store.settle_kept();
    }
    if changed {
        terrains.grounds += 1;
    }
    if reshaped {
        terrains.edits += 1;
    }
    let terrains = &*terrains;
    if changed {
        *ground = Ground(Arc::new(TerrainGround::of(terrains)));
        let bakes: Vec<(Entity, Arc<crate::HydrologyBake>)> = terrains
            .entries
            .iter()
            .filter(|state| state.around.is_none())
            .filter_map(|state| Some((state.entity, Arc::clone(state.hydrology.bake.as_ref()?))))
            .collect();
        if !same_bakes(&hydrology.bakes, &bakes) {
            hydrology.bakes = bakes;
        }
        let next = SurfacePrograms::of(terrains);
        if *programs != next {
            *programs = next;
        }
    }
    if changed
        || settings.is_changed()
        || (assets.is_changed() && assets_moved(terrains, &tiles, &assets))
    {
        *tiles = TerrainTiles {
            terrains: terrains
                .entries
                .iter()
                .filter(|state| state.drawn)
                .map(|state| entry_of((terrains, state), &settings, &assets))
                .collect(),
        };
    }
}
