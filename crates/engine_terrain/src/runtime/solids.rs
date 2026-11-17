use engine_core::{Cursor, Since};
use engine_open_world_kit::ground::TileChange;
use engine_places_simulation::physics::ColliderShape;
use engine_solids::{
    FieldCover, FieldGrid, FieldSamples, FieldTile, SolidField, SolidSource, SolidTree, WorldSolid,
};
use glam::Quat;

use super::*;

struct TileSamples(Arc<Tile>);

impl FieldSamples for TileSamples {
    fn height(&self, x: i32, z: i32) -> f32 {
        self.0.height(x, z)
    }

    fn hole(&self, x: u32, z: u32) -> bool {
        self.0.is_hole(x, z)
    }
}

fn field_of(state: &TerrainState, tile: &Arc<Tile>) -> Arc<FieldTile> {
    let key = tile.key();
    let size = TILE_QUADS as f32 * tile.spacing();
    let corner =
        state.terrain.origin + Vec3::new(f32::from(key.x) * size, 0.0, f32::from(key.z) * size);
    Arc::new(FieldTile::new(
        state.entity,
        corner,
        (TILE_QUADS, tile.spacing()),
        tile.bounds(),
        Box::new(TileSamples(Arc::clone(tile))),
    ))
}

#[derive(Default)]
struct Covered(HashMap<TileKey, (Arc<Tile>, Arc<FieldTile>)>);

impl Covered {
    fn cover(&mut self, state: &TerrainState) -> FieldCover {
        let pyramid = state.terrain.pyramid;
        let corner = Vec2::new(state.terrain.origin.x, state.terrain.origin.z);
        let mut grids: Vec<FieldGrid> = (0..pyramid.levels())
            .map(|level| FieldGrid {
                corner,
                size: pyramid.tile_size(level),
                cells: Vec::new(),
            })
            .collect();
        let mut held = std::mem::take(&mut self.0);
        for tile in state.store.tiles() {
            let key = tile.key();
            let field = match held.remove(&key) {
                Some((old, field)) if Arc::ptr_eq(&old, tile) => field,
                Some(_) | None => field_of(state, tile),
            };
            self.0.insert(key, (Arc::clone(tile), Arc::clone(&field)));
            if let Some(grid) = grids.get_mut(usize::from(key.level)) {
                grid.cells.push(([key.x, key.z], field));
            }
        }
        for grid in &mut grids {
            grid.cells.sort_unstable_by_key(|(cell, _)| *cell);
        }
        FieldCover {
            within: [corner, corner + pyramid.extent()],
            without: state.around.as_ref().map(|around| {
                let (low, high) = around.cutout;
                [corner + low, corner + high]
            }),
            grids,
        }
    }
}

fn collision_tiles(state: &TerrainState) -> impl Iterator<Item = &Arc<Tile>> {
    state
        .collision
        .iter()
        .filter(|_| state.terrain.collision)
        .filter_map(|key| state.store.get(key))
}

fn bridge_solids(state: &TerrainState, solids: &mut Vec<WorldSolid>) {
    if !state.terrain.collision {
        return;
    }
    let origin = state.terrain.origin;
    for (index, bridge) in state.bridges.iter().enumerate() {
        for pair in bridge.deck.windows(2) {
            let (a, b) = (origin + pair[0], origin + pair[1]);
            let along = b - a;
            let length = along.length();
            if length < 1.0e-3 {
                continue;
            }
            let center = 0.5 * (a + b) + Vec3::Y * (0.5 * (DECK_RISE - DECK_DEPTH));
            let model = Mat4::from_rotation_translation(
                Quat::from_rotation_arc(Vec3::X, along / length),
                center,
            );
            solids.push(WorldSolid {
                owner: state.entity,
                layer: BRIDGE_SOLIDS,
                asset: index.min(usize::from(u16::MAX)) as u16,
                shape: ColliderShape::Cuboid {
                    half_extents: Vec3::new(
                        0.5 * length,
                        0.5 * (DECK_RISE + DECK_DEPTH),
                        0.5 * bridge.width + DECK_MARGIN,
                    ),
                },
                model,
                memberships: 1,
                filter: u32::MAX,
            });
        }
    }
}

fn placing_extent(state: &TerrainState) -> Option<(Vec2, Vec2)> {
    let assets = &state.terrain.surface.assets;
    if assets.iter().all(|asset| asset.collider.is_none()) {
        return None;
    }
    collision_tiles(state)
        .map(|tile| {
            let key = tile.key();
            let size = TILE_QUADS as f32 * tile.spacing();
            let corner = Vec2::new(f32::from(key.x), f32::from(key.z)) * size;
            (corner, corner + size)
        })
        .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))
}

fn places_within(tile: &Tile, (low, high): (Vec2, Vec2)) -> bool {
    let key = tile.key();
    let reach = TILE_QUADS as f32 * tile.spacing();
    let corner = Vec2::new(f32::from(key.x), f32::from(key.z)) * reach;
    !tile.placed().is_empty() && !(corner + reach).cmplt(low).any() && !corner.cmpgt(high).any()
}

fn placed_solids(state: &TerrainState, solids: &mut Vec<WorldSolid>) -> HashSet<TileKey> {
    let mut placing = HashSet::default();
    let Some(extent) = placing_extent(state) else {
        return placing;
    };
    let origin = state.terrain.origin;
    let assets = &state.terrain.surface.assets;
    let mut tiles: HashMap<(u16, u16), &Arc<Tile>> = HashMap::default();
    let mut size = 0.0;
    for tile in collision_tiles(state) {
        let key = tile.key();
        size = TILE_QUADS as f32 * tile.spacing();
        tiles.insert((key.x, key.z), tile);
    }
    for tile in state.store.tiles() {
        if !places_within(tile, extent) {
            continue;
        }
        let before = solids.len();
        for placed in tile.placed() {
            if placed.near_level.is_some() {
                continue;
            }
            let Some(collider) = assets
                .get(usize::from(placed.asset))
                .and_then(|asset| asset.collider)
            else {
                continue;
            };
            let [x, _, z] = placed.position;
            let (column, row) = ((x / size).floor(), (z / size).floor());
            if !(0.0..65536.0).contains(&column) || !(0.0..65536.0).contains(&row) {
                continue;
            }
            let Some(ground) = tiles.get(&(column as u16, row as u16)) else {
                continue;
            };
            let local = Vec2::new(x - column * size, z - row * size);
            let Some(sample) = ground.sample(local).filter(|sample| !sample.hole) else {
                continue;
            };
            let model = placed.model(origin, sample.height - placed.sink)
                * Mat4::from_translation(Vec3::from_array(collider.center));
            let shape = match collider.shape {
                PlacedShape::Cuboid { half_extents } => ColliderShape::Cuboid {
                    half_extents: Vec3::from_array(half_extents),
                },
                PlacedShape::Sphere { radius } => ColliderShape::Sphere { radius },
            };
            solids.push(WorldSolid {
                owner: state.entity,
                layer: placed.layer,
                asset: placed.asset,
                shape,
                model,
                memberships: collider.memberships,
                filter: collider.filter,
            });
        }
        if solids.len() != before {
            placing.insert(tile.key());
        }
    }
    placing
}

#[derive(Default)]
pub(crate) struct Published {
    cursor: Cursor,
    around: Cursor,
    fields: Vec<(Arc<Tile>, Arc<FieldTile>)>,
    solids: Arc<SolidTree>,
    bridges: Arc<[Bridge]>,
    colliders: Vec<Option<PlacedCollider>>,
    placing: HashSet<TileKey>,
    surrounded: bool,
    covered: [Covered; 2],
    covers: Vec<FieldCover>,
}

impl Published {
    fn same_tiles(&self, state: &TerrainState) -> bool {
        let mut fields = self.fields.iter();
        collision_tiles(state)
            .all(|tile| fields.next().is_some_and(|(old, _)| Arc::ptr_eq(tile, old)))
            && fields.next().is_none()
    }

    fn touched(&self, state: &TerrainState, changes: &[TileChange]) -> bool {
        let extent = placing_extent(state);
        changes.iter().any(|change| {
            self.placing.contains(&change.key)
                || extent.is_some_and(|extent| {
                    state
                        .store
                        .get(&change.key)
                        .is_some_and(|tile| places_within(tile, extent))
                })
        })
    }

    fn update(
        &mut self,
        state: &TerrainState,
        around: Option<&TerrainState>,
    ) -> Option<Arc<SolidField>> {
        let since = state.store.changes().since(&mut self.cursor);
        let (touched, faced) = match since {
            Since::Same => (false, false),
            Since::Listed(changes) => (self.touched(state, changes), true),
            Since::All => (true, true),
        };
        let surrounded = match around {
            Some(around) => {
                !matches!(around.store.changes().since(&mut self.around), Since::Same)
                    || !self.surrounded
            }
            None => self.surrounded,
        };
        let regrounded = faced || surrounded;
        let assets = &state.terrain.surface.assets;
        let moved = touched
            || !self.same_tiles(state)
            || !Arc::ptr_eq(&self.bridges, &state.bridges)
            || !assets
                .iter()
                .map(|asset| asset.collider)
                .eq(self.colliders.iter().copied());
        if !moved && !regrounded {
            return None;
        }
        if moved {
            self.bridges = Arc::clone(&state.bridges);
            self.colliders.clear();
            self.colliders
                .extend(assets.iter().map(|asset| asset.collider));
            let mut kept = std::mem::take(&mut self.fields);
            if matches!(since, Since::All) {
                kept.clear();
            }
            self.fields = collision_tiles(state)
                .map(|tile| {
                    let field = kept
                        .iter()
                        .find(|(old, _)| Arc::ptr_eq(old, tile))
                        .map_or_else(|| field_of(state, tile), |(_, field)| Arc::clone(field));
                    (Arc::clone(tile), field)
                })
                .collect();
            let mut solids = Vec::new();
            bridge_solids(state, &mut solids);
            self.placing = placed_solids(state, &mut solids);
            self.solids = Arc::new(SolidTree::new(solids));
        }
        if regrounded {
            self.surrounded = around.is_some();
            let [inner, outer] = &mut self.covered;
            self.covers = std::iter::once(inner.cover(state))
                .chain(around.map(|around| outer.cover(around)))
                .collect();
        }
        Some(Arc::new(SolidField {
            fields: self
                .fields
                .iter()
                .map(|(_, field)| Arc::clone(field))
                .collect(),
            solids: Arc::clone(&self.solids),
            covers: self.covers.clone(),
        }))
    }
}

pub(crate) fn sync_solids(
    settings: Res<TerrainSettings>,
    terrains: Res<Terrains>,
    mut sources: Query<(Entity, &mut SolidSource), With<TerrainScene>>,
    mut published: Local<HashMap<Entity, Published>>,
) {
    for (entity, mut source) in &mut sources {
        let held = |surrounding: bool| {
            terrains
                .entries
                .iter()
                .find(|state| state.entity == entity && state.around.is_some() == surrounding)
        };
        let Some(state) = held(false).filter(|state| state.recipe.is_some()) else {
            if published.remove(&entity).is_some() {
                source.field = Arc::default();
            }
            continue;
        };
        let around = held(true).filter(|_| settings.surroundings);
        if let Some(field) = published.entry(entity).or_default().update(state, around) {
            source.field = field;
        }
    }
    published.retain(|entity, _| sources.contains(*entity));
}
