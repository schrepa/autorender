//! Waterfalls' sheets: where terrains' rivers leave their beds
//! (`engine_core::terrain_data::Fall`), the water pass draws the water's
//! path as a sheet the river's width, from a little up the river over its
//! lip, through the air on each jet and down the chutes between, on into
//! the water it lands in at last: widening as it flies, its edges curling
//! back toward the rock as thick as it stands, lying on the chutes' water.
//! Each vertex holds where across the sheet it lies, the seconds since the
//! water passing it left the lip, since it last took off or landed, and how
//! fast it runs there, so the shader (`meshes/water.wgsl`) draws what the
//! water carries where it has carried it: streaks falling and stretching as
//! the water speeds up, white water breaking up as it flies and boiling
//! where it lands.

use engine_core::glam::{Vec2, Vec3};
use engine_core::math::{clamp, max};
use engine_core::terrain_data::Fall;

use crate::gpu_layouts::water::FallVertex;

/// Meters between the sheet's rows along its path, and its columns across.
const ROW: f32 = 0.75;
const COLUMNS: u32 = 8;
/// Meters the sheet reaches up the river over its lip, and on down into
/// the water it lands in.
const UPSTREAM: f32 = 1.5;
const PLUNGE: f32 = 1.5;
/// How much the sheet widens as it flies: a share of its width a second.
const SPREAD: f32 = 0.03;
/// The thinnest and thickest the sheet stands (meters): the water each
/// meter of its width carries over its speed.
const THICKNESS: [f32; 2] = [0.05, 1.2];
/// Meters the sheet stands off the river's water where it lies on it.
const LIFT: f32 = 0.05;
/// The slowest the sheet's water runs (meters a second).
const SLOWEST: f32 = 0.3;
/// What the seconds since landing read before the water has landed.
const UNLANDED: f32 = 1.0e3;

/// Every fall's sheet of `falls` (each terrain's origin and falls), into
/// `vertices` and the triangles' `indices`.
pub(crate) fn sheets<'a>(
    falls: impl IntoIterator<Item = (Vec3, &'a [Fall])>,
    vertices: &mut Vec<FallVertex>,
    indices: &mut Vec<u32>,
) {
    vertices.clear();
    indices.clear();
    let mut number = 0_u32;
    for (origin, set) in falls {
        for fall in set {
            sheet(fall, origin, number, vertices, indices);
            number += 1;
        }
    }
}

/// A point of a fall's path: where it stands (terrain-local), the seconds
/// after leaving the lip its water passes it, and whether it flies from
/// there to the next point.
#[derive(Clone, Copy)]
struct Passing {
    at: Vec3,
    time: f32,
    aloft: bool,
}

/// Fall `fall`'s path, reaching UPSTREAM up the river over its lip at the
/// speed it leaves at and PLUNGE on into the water it lands in as it lands,
/// with that speed (meters a second); `None` for a path too short to draw.
fn passings(fall: &Fall) -> Option<(Vec<Passing>, f32)> {
    let count = fall.path.len();
    if count < 2 || fall.times.len() != count || fall.aloft.len() != count {
        return None;
    }
    let mut path: Vec<Passing> = (0..count)
        .map(|index| Passing {
            at: Vec3::from_array(fall.path[index]),
            time: fall.times[index],
            aloft: fall.aloft[index],
        })
        .collect();
    let (first, second) = (path[0], path[1]);
    let leaving = second.at - first.at;
    let lip_speed = max(
        Vec2::new(leaving.x, leaving.z).length() / max(second.time - first.time, 1.0e-3),
        SLOWEST,
    );
    let upstream = Vec3::new(leaving.x, 0.0, leaving.z).normalize_or(Vec3::X);
    path.insert(
        0,
        Passing {
            at: first.at - upstream * UPSTREAM,
            time: -UPSTREAM / lip_speed,
            aloft: false,
        },
    );
    let (before, last) = (path[path.len() - 2], path[path.len() - 1]);
    let landing = last.at - before.at;
    let landing_speed = max(
        landing.length() / max(last.time - before.time, 1.0e-3),
        SLOWEST,
    );
    path.push(Passing {
        at: last.at + landing.normalize_or(Vec3::NEG_Y) * PLUNGE,
        time: last.time + PLUNGE / landing_speed,
        aloft: false,
    });
    Some((path, lip_speed))
}

/// A row of a sheet: its middle (terrain-local), the seconds after leaving
/// the lip its water passes it, the way it runs and how fast, the seconds
/// since it took off where it flies, less those since it landed where it
/// runs down a chute, and whether it flies.
#[derive(Clone, Copy)]
struct Row {
    center: Vec3,
    time: f32,
    tangent: Vec3,
    speed: f32,
    flying: f32,
    aloft: bool,
}

/// A sheet's rows along `path`: every ROW meters, and at each of its points.
fn rows(path: &[Passing]) -> Vec<Row> {
    let mut rows = Vec::new();
    let (mut took_off, mut landed) = (0.0, -UNLANDED);
    for index in 0..path.len() - 1 {
        let (a, b) = (path[index], path[index + 1]);
        if a.aloft && (index == 0 || !path[index - 1].aloft) {
            took_off = a.time;
        }
        if !a.aloft && index > 0 && path[index - 1].aloft {
            landed = a.time;
        }
        let run = b.at - a.at;
        let length = run.length();
        let steps = max((length / ROW).ceil(), 1.0) as u32;
        let speed = max(length / max(b.time - a.time, 1.0e-3), SLOWEST);
        let tangent = run.normalize_or(Vec3::NEG_Y);
        for step in 0..steps {
            let share = step as f32 / steps as f32;
            let time = a.time + (b.time - a.time) * share;
            rows.push(Row {
                center: a.at + run * share,
                time,
                tangent,
                speed,
                flying: if a.aloft {
                    time - took_off
                } else {
                    landed - time
                },
                aloft: a.aloft,
            });
        }
    }
    if let (Some(&end), Some(&last)) = (path.last(), rows.last()) {
        rows.push(Row {
            center: end.at,
            time: end.time,
            flying: landed - end.time,
            aloft: false,
            ..last
        });
    }
    rows
}

/// Fall `fall`'s sheet, its terrain at `origin` and its number `number`.
fn sheet(
    fall: &Fall,
    origin: Vec3,
    number: u32,
    vertices: &mut Vec<FallVertex>,
    indices: &mut Vec<u32>,
) {
    let Some((path, lip_speed)) = passings(fall) else {
        return;
    };
    let half = 0.5 * fall.width;
    let carried = fall.discharge / max(fall.width, 0.1);
    let heading = Vec3::new(
        fall.bottom[0] - fall.top[0],
        0.0,
        fall.bottom[2] - fall.top[2],
    )
    .normalize_or(Vec3::X);
    let rows = rows(&path);
    for (index, row) in rows.iter().enumerate() {
        // Rows' tangents turn smoothly between their segments.
        let previous = rows[index.saturating_sub(1)].tangent;
        let tangent = (row.tangent + previous).normalize_or(row.tangent);
        let level = Vec3::new(tangent.x, 0.0, tangent.z);
        let across = if level.length() > 0.05 {
            Vec3::new(-level.z, 0.0, level.x).normalize()
        } else {
            Vec3::new(-heading.z, 0.0, heading.x)
        };
        // Up where it lies on the water, out from the rock as it turns down.
        let normal = across.cross(tangent).normalize_or(Vec3::Y);
        let width = half * (1.0 + SPREAD * max(row.flying, 0.0));
        let thickness = clamp(carried / row.speed, THICKNESS[0], THICKNESS[1]);
        let lift = if row.aloft { 0.0 } else { LIFT };
        let base = vertices.len() as u32;
        for column in 0..=COLUMNS {
            let side = column as f32 / COLUMNS as f32 * 2.0 - 1.0;
            let curl = if row.aloft {
                0.5 * thickness * side * side
            } else {
                0.0
            };
            let position = origin + row.center + across * (side * width) + normal * (lift - curl);
            vertices.push(FallVertex {
                position: position.to_array(),
                normal: normal.to_array(),
                tangent: [tangent.x, tangent.y, tangent.z, row.speed],
                jet: [side, row.time, lip_speed, half],
                fall: [
                    row.flying,
                    thickness,
                    number as f32,
                    if row.aloft { 1.0 } else { 0.0 },
                ],
            });
        }
        if index > 0 {
            let previous = base - (COLUMNS + 1);
            for column in 0..COLUMNS {
                let (a, b) = (previous + column, base + column);
                indices.extend_from_slice(&[a, a + 1, b + 1, a, b + 1, b]);
            }
        }
    }
}

/// The vertex buffer layout of [`FallVertex`]es.
pub(crate) fn fall_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = crate::pipelines::vertex_attributes!(FallVertex {
        0 => position as Float32x3,
        1 => normal as Float32x3,
        2 => tangent as Float32x4,
        3 => jet as Float32x4,
        4 => fall as Float32x4,
    });
    wgpu::VertexBufferLayout {
        array_stride: size_of::<FallVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    }
}
