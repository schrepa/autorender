//! Waterfalls' jets: where terrains' rivers leave their beds
//! (`engine_core::terrain_data::Fall`), the water pass draws each jet's
//! water through the air as a sheet the river's width, from where it leaves
//! its lip to where it meets the water or rock below: widening as it flies,
//! its edges curling back toward the rock as thick as it stands. Water lying
//! on the ground, up the river from a lip, down the rock between a cascade's
//! jets and in the pool a jet lands in, the cooked water draws with its own
//! flow and foam. Each vertex holds how far across the sheet it stands from
//! the jet's middle, half the width of its water there and how far that
//! water's edges have spread, the sheet's thickness, how white its water has
//! turned and how far the air has broken it up, which the shader
//! (`meshes/water_falls.wgsl`) shows and nothing else, smoothly across the
//! sheet. All of it follows the falling water's physics: a jet leaves its
//! lip as thick as the water each meter of it carries over its speed, its
//! edges spread a share of how far it has fallen, and its core lasts a fall
//! that grows with what it carries (Horeni, after USBR 2015); clinging
//! to rock between a cascade's jets, its water turns white once its
//! turbulence reaches its surface (Chanson's inception point) as far as the
//! rock is steep enough to keep air in it, turns white where it lands and
//! lets its air go down gentler rock, its foam lingering in patches, so each
//! jet takes off as white as its water has turned.

use std::ops::Range;

use engine_core::glam::{Vec2, Vec3};
use engine_core::math::{clamp, max, min};
use engine_core::terrain_data::Fall;

use crate::gpu_layouts::water::FallVertex;

/// Meters between a jet's rows along its path, and its columns across.
const ROW: f32 = 0.75;
const COLUMNS: u32 = 8;
/// The meters each edge of a jet spreads a meter it falls (2 to 5%).
const EDGE_SPREAD: f32 = 0.035;
/// The meters a jet's coherent core lasts as it falls, carrying `q` cubic
/// meters a second each meter of its width: BREAKUP q^0.32 (Horeni); the
/// meters of rock water clinging to it runs before it turns white, the
/// first of INCEPTION plus its second q^1/2, at most its third; and the
/// seconds over which a jet's water turns white in the air.
const BREAKUP: f32 = 6.0;
const INCEPTION: [f32; 3] = [1.0, 2.0, 6.0];
const AIRING: f32 = 0.4;
/// Clinging to rock, water turns white as its air fraction, about
/// AIR_SHARE of the sine of the rock's angle (Chanson), rises through
/// AIRED, the angle read across SKIM_REACH meters of its path either way:
/// skimming over a stepped bed it rides its steps' edges, taking in air as
/// the rock falls overall rather than step by step; landing, it turns
/// white, and it lets its air go down gentler rock over AIR_RELEASE
/// seconds, down to LINGERING white: its foam's patches.
const AIR_SHARE: f32 = 0.9;
const AIRED: [f32; 2] = [0.15, 0.55];
const SKIM_REACH: f32 = 6.0;
const AIR_RELEASE: f32 = 1.2;
const LINGERING: f32 = 0.3;
/// The thinnest and thickest a jet stands (meters): the water each meter
/// of its width carries over its speed.
const THICKNESS: [f32; 2] = [0.05, 1.2];
/// The share wider than its river a jet's sheet stands besides its edges'
/// spread, so its water's edges thin out on the sheet rather than at its
/// border.
const OVERHANG: f32 = 1.15;
/// How much faster a jet's middle leaves its lip than its edges (a share
/// of its speed): its curtain bows out in the middle as it falls.
const BOW: f32 = 0.12;
/// The slowest the falling water runs (meters a second).
const SLOWEST: f32 = 0.3;

/// Every fall's jets of `falls` (each terrain's origin and falls), into
/// `vertices` and the triangles' `indices`; returns the indices of each
/// terrain's jets, in its order.
pub(crate) fn sheets<'a>(
    falls: impl IntoIterator<Item = (Vec3, &'a [Fall])>,
    vertices: &mut Vec<FallVertex>,
    indices: &mut Vec<u32>,
) -> Vec<Range<u32>> {
    vertices.clear();
    indices.clear();
    falls
        .into_iter()
        .map(|(origin, set)| {
            let first = indices.len() as u32;
            for fall in set {
                fall_jets(fall, origin, vertices, indices);
            }
            first..indices.len() as u32
        })
        .collect()
}

/// A point of a fall's path: where it stands (terrain-local), the seconds
/// after leaving the lip its water passes it, whether it flies through the
/// air from there to the next point, half the meters across its water
/// there and how much of the sky it sees.
#[derive(Clone, Copy)]
struct Passing {
    at: Vec3,
    time: f32,
    aloft: bool,
    half: f32,
    sky: f32,
}

/// Fall `fall`'s path, and the speed its water leaves its lip at (meters a
/// second); `None` for a path too short to draw.
fn passings(fall: &Fall) -> Option<(Vec<Passing>, f32)> {
    let count = fall.path.len();
    if count < 2
        || fall.times.len() != count
        || fall.aloft.len() != count
        || fall.widths.len() != count
        || fall.sky.len() != count
    {
        return None;
    }
    let path: Vec<Passing> = (0..count)
        .map(|index| Passing {
            at: Vec3::from_array(fall.path[index]),
            time: fall.times[index],
            aloft: fall.aloft[index],
            half: 0.5 * fall.widths[index],
            sky: fall.sky[index],
        })
        .collect();
    let (first, second) = (path[0], path[1]);
    let leaving = second.at - first.at;
    let lip_speed = max(
        Vec2::new(leaving.x, leaving.z).length() / max(second.time - first.time, 1.0e-3),
        SLOWEST,
    );
    Some((path, lip_speed))
}

/// A row of a jet: its middle (terrain-local), the way it runs and how
/// fast, the seconds since it took off, how white its water has turned and
/// how far the air has broken it up (0 to 1 each), the meters each of its
/// edges has spread, half the meters across its water and how much of the
/// sky it sees.
#[derive(Clone, Copy)]
struct Row {
    center: Vec3,
    tangent: Vec3,
    speed: f32,
    flying: f32,
    white: f32,
    broken: f32,
    spread: f32,
    half: f32,
    sky: f32,
}

/// Where the water passing `share` of the way from `a` to `b` is, the way it
/// runs there and how fast: straight along the way between them.
fn on_segment(a: &Passing, b: &Passing, share: f32) -> (Vec3, Vec3, f32) {
    let run = b.at - a.at;
    let seconds = max(b.time - a.time, 1.0e-3);
    let speed = max(run.length() / seconds, SLOWEST);
    (a.at + run * share, run.normalize_or(Vec3::NEG_Y), speed)
}

/// The sine of the angle `path` falls at each of its points, read across
/// SKIM_REACH meters of it either way.
fn skimmed(path: &[Passing]) -> Vec<f32> {
    let mut along = vec![0.0_f32; path.len()];
    for index in 1..path.len() {
        along[index] = along[index - 1] + path[index].at.distance(path[index - 1].at);
    }
    let (mut first, mut last) = (0, 0);
    (0..path.len())
        .map(|index| {
            while along[index] - along[first] > SKIM_REACH {
                first += 1;
            }
            while last + 1 < path.len() && along[last + 1] - along[index] <= SKIM_REACH {
                last += 1;
            }
            let run = along[last] - along[first];
            if run > 0.0 {
                (path[first].at.y - path[last].at.y) / run
            } else {
                0.0
            }
        })
        .collect()
}

/// 0 at or below `low`, rising smoothly to 1 at `high`.
fn smooth(low: f32, high: f32, value: f32) -> f32 {
    let t = clamp((value - low) / (high - low), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Each jet's rows along `path`, its water carrying `carried` cubic meters
/// a second each meter of its width: every ROW meters of its flight from
/// where it takes off, and where it lands. Between its jets the water
/// clings to the rock, turning white once its turbulence reaches its
/// surface, as far as the rock is steep enough to keep air in it, and
/// letting its air go down gentler rock, its foam lingering in patches;
/// each jet takes off as white as its water has turned.
fn rows(path: &[Passing], carried: f32) -> Vec<Vec<Row>> {
    let falling = skimmed(path);
    let breakup = BREAKUP * libm::powf(max(carried, 1.0e-3), 0.32);
    let inception = min(INCEPTION[0] + INCEPTION[1] * carried.sqrt(), INCEPTION[2]);
    let mut jets: Vec<Vec<Row>> = Vec::new();
    // Where and when its water last took off and how white it was then;
    // how white it is, the meters it has clung to rock and when it last
    // passed a row.
    let (mut height, mut took_off, mut airborne_white) = (0.0, 0.0, 0.0);
    let (mut white, mut clung, mut before) = (0.0, 0.0, path[0].time);
    for index in 0..path.len() - 1 {
        let (a, b) = (path[index], path[index + 1]);
        let flew = index > 0 && path[index - 1].aloft;
        if a.aloft && !flew {
            took_off = a.time;
            height = a.at.y;
            airborne_white = white;
            jets.push(Vec::new());
        }
        if !a.aloft && flew {
            white = 1.0;
        }
        let length = (b.at - a.at).length();
        let steps = max((length / ROW).ceil(), 1.0) as u32;
        // A jet's last stretch ends in the row where it lands.
        let lands = a.aloft && (!b.aloft || index + 2 == path.len());
        for step in 0..steps + u32::from(lands) {
            let share = step as f32 / steps as f32;
            let time = a.time + (b.time - a.time) * share;
            if a.aloft {
                let (center, tangent, speed) = on_segment(&a, &b, share);
                let fallen = max(height - center.y, 0.0);
                let flying = time - took_off;
                white = max(airborne_white, 1.0 - libm::expf(-flying / AIRING));
                if let Some(jet) = jets.last_mut() {
                    jet.push(Row {
                        center,
                        tangent,
                        speed,
                        flying,
                        white,
                        broken: smooth(0.3 * breakup, breakup, fallen),
                        spread: EDGE_SPREAD * fallen,
                        half: a.half + (b.half - a.half) * share,
                        sky: a.sky + (b.sky - a.sky) * share,
                    });
                }
            } else {
                clung += length / steps as f32;
                let sine = falling[index] + (falling[index + 1] - falling[index]) * share;
                let churning =
                    smooth(0.0, inception, clung) * smooth(AIRED[0], AIRED[1], AIR_SHARE * sine);
                let released = LINGERING
                    + (white - LINGERING) * libm::expf(-max(time - before, 0.0) / AIR_RELEASE);
                white = max(churning, if white > LINGERING { released } else { white });
            }
            before = time;
        }
    }
    jets
}

/// Fall `fall`'s jets, its terrain at `origin`: each a strip of its rows,
/// COLUMNS across.
fn fall_jets(fall: &Fall, origin: Vec3, vertices: &mut Vec<FallVertex>, indices: &mut Vec<u32>) {
    let Some((path, lip_speed)) = passings(fall) else {
        return;
    };
    let carried = fall.discharge / max(fall.width, 0.1);
    let heading = Vec3::new(
        fall.bottom[0] - fall.top[0],
        0.0,
        fall.bottom[2] - fall.top[2],
    )
    .normalize_or(Vec3::X);
    for jet in rows(&path, carried) {
        for (index, row) in jet.iter().enumerate() {
            // Rows' tangents turn smoothly between their segments.
            let previous = jet[index.saturating_sub(1)].tangent;
            let tangent = (row.tangent + previous).normalize_or(row.tangent);
            let level = Vec3::new(tangent.x, 0.0, tangent.z);
            let across = if level.length() > 0.05 {
                Vec3::new(-level.z, 0.0, level.x).normalize()
            } else {
                Vec3::new(-heading.z, 0.0, heading.x)
            };
            // Out from the rock it falls from.
            let normal = across.cross(tangent).normalize_or(Vec3::Y);
            let width = row.half * OVERHANG + row.spread;
            let thickness = clamp(carried / row.speed, THICKNESS[0], THICKNESS[1]);
            let base = vertices.len() as u32;
            for column in 0..=COLUMNS {
                let side = column as f32 / COLUMNS as f32 * 2.0 - 1.0;
                // Its edges curl back toward the rock as thick as it
                // stands, and its middle bows out ahead of them.
                let curl = 0.5 * thickness * side * side;
                let bow = BOW * (1.0 - side * side) * lip_speed * row.flying;
                let position = origin + row.center + across * (side * width) - normal * curl
                    + level.normalize_or(Vec3::ZERO) * bow;
                vertices.push(FallVertex {
                    position: position.to_array(),
                    normal: normal.to_array(),
                    sheet: [side * width, row.half, row.spread, thickness],
                    state: [row.white, row.broken],
                    sky: row.sky,
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
}

/// The vertex buffer layout of [`FallVertex`]es.
pub(crate) fn fall_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = crate::pipelines::vertex_attributes!(FallVertex {
        0 => position as Float32x3,
        1 => normal as Float32x3,
        2 => sheet as Float32x4,
        3 => state as Float32x2,
        4 => sky as Float32,
    });
    wgpu::VertexBufferLayout {
        array_stride: size_of::<FallVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    }
}
