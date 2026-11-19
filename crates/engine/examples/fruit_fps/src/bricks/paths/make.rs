use std::sync::Arc;

use glam::{Vec2, Vec3};
use serde::Deserialize;
use serde_json::Value;

use crate::nav::{self, Costs, Eyes, NavMesh, NavNumbers, Tactics};
use crate::play::numbers::GameNumbers;
use crate::play::paths::{Built, MeshSource, Tuning};
use crate::play::world::Arena;
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
    let map = match &prepared.layout {
        Some((_, layout)) => layout.map(numbers.player.moving.height_m),
        None => Map::default(),
    };
    Ok(Built {
        source,
        plain: Costs::new(&mesh),
        tactics: Tactics::build(map.covers, nav),
        zones: map.zones,
        layout: prepared.layout.as_ref().map(|(text, _)| text.clone()),
        mesh,
    })
}
