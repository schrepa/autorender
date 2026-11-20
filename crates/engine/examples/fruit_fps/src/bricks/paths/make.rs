use engine_core::Asset;
use engine_nav::{Costs, NavMesh};
use engine_solids::Block;
use glam::{Vec2, Vec3};

use crate::play::cover::Tactics;
use crate::play::files::Triangles;
use crate::play::layout::Layout;
use crate::play::numbers::GameNumbers;
use crate::play::paths::{Built, MeshSource, Tuning};
use crate::play::world::Arena;

use super::floor;
use super::layout::{self, Map};

pub(super) enum Ground<'a> {
    Floor { half_size: f32 },
    Cooked(&'a Triangles),
}

pub(super) type Placed<'a> = Option<(Asset<Layout>, &'a Layout)>;

pub(super) fn tuning(numbers: &GameNumbers) -> Tuning {
    Tuning {
        nav: numbers.nav.clone(),
        radius: numbers.player.moving.radius_m,
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
    let carved = floor::carve(Vec2::splat(-half), Vec2::splat(half), 0.0, &holes);
    NavMesh::build(&carved.positions, &carved.polygons, weld)
        .map_err(|error| format!("floor nav: {error}"))
}

pub(super) fn build(
    (ground, placed): (Ground<'_>, Placed<'_>),
    numbers: &GameNumbers,
    arena: &Arena,
) -> Result<Built, String> {
    let nav = &numbers.nav;
    let (source, mesh) = match ground {
        Ground::Floor { half_size } => (
            MeshSource::Floor,
            floor_mesh(half_size, arena, numbers, nav.weld_m)?,
        ),
        Ground::Cooked(triangles) => (
            MeshSource::Cooked,
            NavMesh::build_triangles(&triangles.positions, &triangles.indices, nav.weld_m)
                .map_err(|error| format!("cooked nav: {error}"))?,
        ),
    };
    let map = match placed {
        Some((_, read)) => layout::map(&read.map, numbers.player.moving.height_m, &numbers.player),
        None => Map::default(),
    };
    Ok(Built {
        source,
        plain: Costs::new(&mesh),
        tactics: Tactics::build(map.covers, nav),
        zones: map.zones,
        layout: placed.map(|(layout, _)| layout),
        mesh,
    })
}
