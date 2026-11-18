use std::sync::Arc;

use glam::{Vec2, Vec3};
use serde::Deserialize;
use serde_json::Value;

use crate::nav::{self, Costs, Eyes, NavMesh, NavNumbers, PlaceShape, Tactics};
use crate::play::numbers::GameNumbers;
use crate::play::paths::{Built, Hold, MeshSource, SpawnIsland, Tuning, Watch, ZoneArea};
use crate::play::world::Arena;
use crate::rules::Side;
use crate::solids::Block;

use super::layout::{Layout, Map};

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct HazardFactors {
    pub damage: f32,
    pub strike: f32,
    pub slow: f32,
    pub near_sight: f32,
    pub reveal: f32,
    pub smoke: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct PathsNumbers {
    pub heat_seconds: f32,
    pub heat_radius_m: f32,
    pub heat_factor: f32,
    pub hazard_factors: HazardFactors,
}

pub(super) enum Ground {
    Floor {
        half_size: f32,
    },
    Cooked {
        positions: Vec<Vec3>,
        indices: Vec<u32>,
    },
}

pub(super) struct Prepared {
    pub ground: Ground,
    pub layout: Option<(Arc<str>, Layout)>,
}

fn table<T: serde::de::DeserializeOwned>(whole: &Value, key: &str) -> Result<T, String> {
    let part = whole
        .get(key)
        .cloned()
        .ok_or_else(|| format!("numbers.json has no `{key}` table"))?;
    serde_json::from_value(part).map_err(|error| format!("numbers.json {key}: {error}"))
}

pub(super) fn tables(numbers: &GameNumbers) -> Result<(NavNumbers, Option<PathsNumbers>), String> {
    let whole: Value =
        serde_json::from_str(&numbers.text).map_err(|error| format!("numbers.json: {error}"))?;
    let paths = match whole.get("paths") {
        Some(_) => Some(table(&whole, "paths")?),
        None => None,
    };
    Ok((table(&whole, "nav")?, paths))
}

pub(super) fn tuning(numbers: &GameNumbers, nav: &NavNumbers) -> Tuning {
    Tuning {
        nav: nav.clone(),
        radius: numbers.player.moving.radius_m,
        run_speed: numbers.player.run_speed_mps,
        eyes: Eyes {
            stand: numbers.player.eye_height_m,
            crouch: numbers.player.crouch_eye_height_m,
        },
    }
}

fn footprint(block: &Block) -> (Vec3, Vec3) {
    let axes = block.axes;
    let half = block.half_extents;
    let extent =
        axes.x_axis.abs() * half.x + axes.y_axis.abs() * half.y + axes.z_axis.abs() * half.z;
    (block.centre - extent, block.centre + extent)
}

fn floor_mesh(
    half: f32,
    arena: &Arena,
    numbers: &GameNumbers,
    weld: f32,
) -> Result<NavMesh, String> {
    let moving = &numbers.player.moving;
    let holes: Vec<(Vec2, Vec2)> = arena
        .solids()
        .blocks()
        .iter()
        .map(footprint)
        .filter(|(low, high)| low.y < moving.height_m && high.y > moving.step_height_m)
        .map(|(low, high)| (Vec2::new(low.x, low.z), Vec2::new(high.x, high.z)))
        .collect();
    let carved = nav::carve(Vec2::splat(-half), Vec2::splat(half), 0.0, &holes);
    NavMesh::build(&carved.positions, &carved.polygons, weld)
        .map_err(|error| format!("floor nav: {error:?}"))
}

fn zone_of(zones: &[ZoneArea], point: Vec3) -> Option<&ZoneArea> {
    zones
        .iter()
        .filter(|zone| zone.covers(point))
        .min_by(|a, b| a.area().total_cmp(&b.area()))
}

struct Sight<'a> {
    mesh: &'a NavMesh,
    tactics: &'a Tactics,
    tuning: &'a Tuning,
    arena: &'a Arena,
}

fn holds(sight: &Sight<'_>, map: &Map) -> Vec<Hold> {
    let Sight {
        mesh,
        tactics,
        tuning,
        arena,
    } = *sight;
    let mut found = Vec::new();
    for site in &map.sites {
        let Some(zone) = zone_of(&map.zones, site.centre) else {
            continue;
        };
        let reach = (zone.max - zone.min).with_y(0.0).length();
        let Some(near) = mesh.locate(site.centre, tuning.nav.locate_height_m, reach) else {
            continue;
        };
        for choke in map
            .chokes
            .iter()
            .filter(|c| c.zones.contains(&zone.place.name))
        {
            let threat = (choke.a + choke.b) * 0.5 + Vec3::Y * tuning.eyes.stand;
            let spots = tactics.positions_with_cover_from(
                mesh,
                threat,
                near,
                tuning.eyes,
                tuning.radius,
                reach,
                |eye, target| arena.sees(eye, target),
            );
            for spot in spots {
                let inside = zone_of(&map.zones, spot.spot.point)
                    .is_some_and(|held| held.place.name == zone.place.name);
                if inside {
                    found.push(Hold {
                        site: site.name.clone(),
                        choke: choke.name.clone(),
                        cover: tactics.places()[spot.place].place.name.clone(),
                        at: spot.spot.point,
                    });
                }
            }
        }
    }
    found
}

fn watches(map: &Map, tactics: &Tactics, tuning: &Tuning) -> Vec<Watch> {
    let lift = Vec3::Y * tuning.eyes.stand;
    let mut points: Vec<Vec3> = map
        .chokes
        .iter()
        .flat_map(|choke| [choke.a + lift, choke.b + lift])
        .collect();
    for facts in tactics.places() {
        if let PlaceShape::Cover { center, half, .. } = facts.place.shape {
            let base = center.y - half.y + tuning.eyes.stand;
            for (sx, sz) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                points.push(Vec3::new(
                    center.x + sx * half.x,
                    base,
                    center.z + sz * half.z,
                ));
            }
        }
    }
    points
        .into_iter()
        .map(|at| Watch {
            at,
            zone: zone_of(&map.zones, at).map(|zone| zone.place.name.clone()),
        })
        .collect()
}

fn spawn_islands(mesh: &NavMesh, arena: &Arena, nav: &NavNumbers) -> Vec<SpawnIsland> {
    [Side::Attack, Side::Defend]
        .into_iter()
        .flat_map(|side| {
            arena.spawns(side).iter().map(move |spawn| SpawnIsland {
                side,
                at: spawn.position,
                island: mesh
                    .locate(spawn.position, nav.locate_height_m, nav.locate_distance_m)
                    .map(|spot| mesh.island(spot.polygon)),
            })
        })
        .collect()
}

pub(super) fn build(
    prepared: &Prepared,
    numbers: &GameNumbers,
    nav: &NavNumbers,
    arena: &Arena,
) -> Result<Built, String> {
    let (source, mesh) = match &prepared.ground {
        Ground::Floor { half_size } => (
            MeshSource::Floor,
            floor_mesh(*half_size, arena, numbers, nav.weld_m)?,
        ),
        Ground::Cooked { positions, indices } => (
            MeshSource::Cooked,
            NavMesh::build_triangles(positions, indices, nav.weld_m)
                .map_err(|error| format!("cooked nav: {error:?}"))?,
        ),
    };
    let tuning = tuning(numbers, nav);
    let map = match &prepared.layout {
        Some((_, layout)) => layout.map(numbers.player.moving.height_m)?,
        None => Map::default(),
    };
    let tactics = Tactics::build(&mesh, map.places.clone(), tuning.radius, nav);
    let sight = Sight {
        mesh: &mesh,
        tactics: &tactics,
        tuning: &tuning,
        arena,
    };
    let holds = holds(&sight, &map);
    let watches = watches(&map, &tactics, &tuning);
    let spawn_islands = spawn_islands(&mesh, arena, nav);
    Ok(Built {
        source,
        plain: Costs::new(&mesh),
        places: map.zones.iter().map(|zone| zone.place.clone()).collect(),
        zones: map.zones,
        holds,
        watches,
        spawn_islands,
        layout: prepared.layout.as_ref().map(|(text, _)| text.clone()),
        mesh,
        tactics,
    })
}
