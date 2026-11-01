use std::any::Any;

use engine_places::solids::{
    FieldSamples, FieldTile, SolidField, SolidShape, SolidSource, WorldSolid,
};
use glam::Quat;

use super::*;

struct TerrainSolids {
    fields: Vec<FieldTile>,
    solids: Vec<WorldSolid>,
}

impl SolidField for TerrainSolids {
    fn fields(&self) -> &[FieldTile] {
        &self.fields
    }

    fn solids(&self) -> &[WorldSolid] {
        &self.solids
    }
}

struct TileSamples(Arc<Tile>);

impl FieldSamples for TileSamples {
    fn height(&self, x: i32, z: i32) -> f32 {
        self.0.height(x, z)
    }

    fn hole(&self, x: u32, z: u32) -> bool {
        self.0.is_hole(x, z)
    }

    fn same(&self, other: &dyn FieldSamples) -> bool {
        other
            .as_any()
            .downcast_ref::<TileSamples>()
            .is_some_and(|other| {
                self.0.heights() == other.0.heights() && self.0.hole_bits() == other.0.hole_bits()
            })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn collision_tiles(state: &TerrainState) -> impl Iterator<Item = &Arc<Tile>> {
    state
        .collision
        .iter()
        .filter(|_| state.terrain.collision)
        .filter_map(|key| state.store.get(key))
}

fn ground_fields(state: &TerrainState, fields: &mut Vec<FieldTile>) {
    for tile in collision_tiles(state) {
        let samples: Arc<dyn FieldSamples> = Arc::new(TileSamples(Arc::clone(tile)));
        fields.push(FieldTile {
            owner: state.entity,
            key: tile.key(),
            origin: state.terrain.origin,
            spacing: tile.spacing(),
            bounds: tile.bounds(),
            samples,
        });
    }
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
                shape: SolidShape::Cuboid {
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

fn placed_solids(terrains: &Terrains, state: &TerrainState, solids: &mut Vec<WorldSolid>) {
    let entity = state.entity;
    let origin = state.terrain.origin;
    let mut tiles: HashMap<(u16, u16), &Arc<Tile>> = HashMap::default();
    let mut size = 0.0;
    for tile in collision_tiles(state) {
        let key = tile.key();
        size = TILE_QUADS as f32 * tile.spacing();
        tiles.insert((key.x, key.z), tile);
    }
    if tiles.is_empty() {
        return;
    }
    let Some(terrain) = terrains.terrain(entity) else {
        return;
    };
    let assets = &terrain.surface.assets;
    if assets.iter().all(|asset| asset.collider.is_none()) {
        return;
    }
    let (low, high) = tiles.keys().fold(
        (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)),
        |(low, high), &(x, z)| {
            let corner = Vec2::new(f32::from(x), f32::from(z)) * size;
            (low.min(corner), high.max(corner + size))
        },
    );
    for tile in terrains.tiles(entity) {
        let key = tile.key();
        let reach = TILE_QUADS as f32 * tile.spacing();
        let corner = Vec2::new(f32::from(key.x), f32::from(key.z)) * reach;
        if tile.placed().is_empty() || (corner + reach).cmplt(low).any() || corner.cmpgt(high).any()
        {
            continue;
        }
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
                PlacedShape::Cuboid { half_extents } => SolidShape::Cuboid {
                    half_extents: Vec3::from_array(half_extents),
                },
                PlacedShape::Sphere { radius } => SolidShape::Sphere { radius },
            };
            solids.push(WorldSolid {
                owner: entity,
                layer: placed.layer,
                asset: placed.asset,
                shape,
                model,
                memberships: collider.memberships,
                filter: collider.filter,
            });
        }
    }
}

pub(crate) fn sync_solids(
    terrains: Res<Terrains>,
    mut sources: Query<(Entity, &mut SolidSource), With<TerrainScene>>,
    mut published: Local<Vec<(Entity, u64)>>,
) {
    for (entity, mut source) in &mut sources {
        let state = terrains
            .entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none());
        let revision = state.map_or(0, TerrainState::revision);
        let at = published.iter().position(|(seen, _)| *seen == entity);
        if at.is_some_and(|at| published[at].1 == revision) {
            continue;
        }
        let (mut fields, mut solids) = (Vec::new(), Vec::new());
        if let Some(state) = state {
            ground_fields(state, &mut fields);
            bridge_solids(state, &mut solids);
            placed_solids(&terrains, state, &mut solids);
        }
        let field: Arc<dyn SolidField> = Arc::new(TerrainSolids { fields, solids });
        source.field = field;
        match at {
            Some(at) => published[at].1 = revision,
            None => published.push((entity, revision)),
        }
    }
    published.retain(|(entity, _)| sources.contains(*entity));
}
