use std::any::Any;
use std::sync::Arc;

use bevy_ecs::prelude::{Entity, World};
use engine_core::collections::HashMap;
use engine_places::{FieldSamples, SolidField, SolidShape, SolidsFace, WorldSolid};
use glam::{Mat4, Quat, Vec2, Vec3};

use super::{BRIDGE_SOLIDS, DECK_DEPTH, DECK_MARGIN, DECK_RISE, PlacedShape, Terrains};
use crate::{TILE_QUADS, Tile};

pub(crate) struct TerrainFields;

pub(crate) struct TerrainDecks;

pub(crate) struct TerrainPlaced;

fn revision(world: &World) -> u64 {
    world
        .get_resource::<Terrains>()
        .map_or(0, Terrains::revision)
}

impl SolidsFace for TerrainFields {
    fn revision(&self, world: &World) -> u64 {
        revision(world)
    }

    fn fields(&self, world: &World) -> Vec<SolidField> {
        world
            .get_resource::<Terrains>()
            .map_or_else(Vec::new, |terrains| {
                collision_tiles(terrains)
                    .map(|(owner, origin, tile)| {
                        let samples: Arc<dyn FieldSamples> = tile.clone();
                        SolidField {
                            owner,
                            key: tile.key(),
                            origin,
                            spacing: tile.spacing(),
                            bounds: tile.bounds(),
                            samples,
                        }
                    })
                    .collect()
            })
    }
}

impl SolidsFace for TerrainDecks {
    fn revision(&self, world: &World) -> u64 {
        revision(world)
    }

    fn solids(&self, world: &World) -> Vec<WorldSolid> {
        let mut solids = Vec::new();
        if let Some(terrains) = world.get_resource::<Terrains>() {
            bridge_solids(terrains, &mut solids);
        }
        solids
    }
}

impl SolidsFace for TerrainPlaced {
    fn revision(&self, world: &World) -> u64 {
        revision(world)
    }

    fn solids(&self, world: &World) -> Vec<WorldSolid> {
        let mut solids = Vec::new();
        if let Some(terrains) = world.get_resource::<Terrains>() {
            placed_solids(terrains, &mut solids);
        }
        solids
    }
}

impl FieldSamples for Tile {
    fn height(&self, x: i32, z: i32) -> f32 {
        Tile::height(self, x, z)
    }

    fn hole(&self, x: u32, z: u32) -> bool {
        self.is_hole(x, z)
    }

    fn same(&self, other: &dyn FieldSamples) -> bool {
        other.as_any().downcast_ref::<Tile>().is_some_and(|other| {
            self.heights() == other.heights() && self.hole_bits() == other.hole_bits()
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn collision_tiles(terrains: &Terrains) -> impl Iterator<Item = (Entity, Vec3, &Arc<Tile>)> {
    terrains
        .entries
        .iter()
        .filter(|state| state.terrain.collision)
        .flat_map(|state| {
            state.collision.iter().filter_map(move |key| {
                let tile = state.store.get(key)?;
                Some((state.entity, state.terrain.origin, tile))
            })
        })
}

struct Field<'a> {
    origin: Vec3,
    size: f32,
    tiles: HashMap<(u16, u16), &'a Arc<Tile>>,
}

fn bridge_solids(terrains: &Terrains, solids: &mut Vec<WorldSolid>) {
    let bridged = terrains
        .entries
        .iter()
        .filter(|state| state.terrain.collision && !state.bridges.is_empty())
        .map(|state| (state.entity, state.terrain.origin, &*state.bridges));
    for (entity, origin, bridges) in bridged {
        for (index, bridge) in bridges.iter().enumerate() {
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
                    owner: entity,
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
}

fn placed_solids(terrains: &Terrains, solids: &mut Vec<WorldSolid>) {
    let mut fields: HashMap<Entity, Field<'_>> = HashMap::default();
    for (entity, origin, tile) in collision_tiles(terrains) {
        let key = tile.key();
        fields
            .entry(entity)
            .or_insert_with(|| Field {
                origin,
                size: TILE_QUADS as f32 * tile.spacing(),
                tiles: HashMap::default(),
            })
            .tiles
            .insert((key.x, key.z), tile);
    }
    for (
        entity,
        Field {
            origin,
            size,
            tiles,
        },
    ) in fields
    {
        let Some(terrain) = terrains.terrain(entity) else {
            continue;
        };
        let assets = &terrain.surface.assets;
        if assets.iter().all(|asset| asset.collider.is_none()) {
            continue;
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
            if tile.placed().is_empty()
                || (corner + reach).cmplt(low).any()
                || corner.cmpgt(high).any()
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
}
