use std::ops::Range;

use engine_core::math::{clamp, exp, max, min, pow};
use engine_heightfield::Fall;
use engine_render::wgpu;
use glam::{Vec2, Vec3};

use crate::gpu::FallVertex;
use crate::settings::WaterFallSheets;

const BREAKUP: f32 = 6.0;
const BREAKUP_POWER: f32 = 0.32;

pub(crate) fn sheets<'a>(
    falls: impl IntoIterator<Item = (Vec3, &'a [Fall], u32)>,
    look: &WaterFallSheets,
    vertices: &mut Vec<FallVertex>,
    indices: &mut Vec<u32>,
) -> Vec<Range<u32>> {
    vertices.clear();
    indices.clear();
    falls
        .into_iter()
        .map(|(origin, set, terrain)| {
            let first = indices.len() as u32;
            let made = vertices.len();
            for fall in set {
                fall_sheet(fall, origin, look, vertices, indices);
            }
            for vertex in &mut vertices[made..] {
                vertex.terrain = terrain;
            }
            first..indices.len() as u32
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Passing {
    at: Vec3,
    time: f32,
    aloft: bool,
    half: f32,
    sky: f32,
}

fn passings(fall: &Fall) -> Option<Vec<Passing>> {
    let count = fall.path.len();
    if count < 2
        || fall.times.len() != count
        || fall.aloft.len() != count
        || fall.widths.len() != count
        || fall.sky.len() != count
    {
        return None;
    }
    Some(
        (0..count)
            .map(|index| Passing {
                at: Vec3::from_array(fall.path[index]),
                time: fall.times[index],
                aloft: fall.aloft[index],
                half: 0.5 * fall.widths[index],
                sky: fall.sky[index],
            })
            .collect(),
    )
}

#[derive(Clone, Copy)]
struct Row {
    center: Vec3,
    tangent: Vec3,
    speed: f32,
    fallen: f32,
    free: f32,
    bowed: f32,
    white: f32,
    broken: f32,
    spread: f32,
    half: f32,
    sky: f32,
    along: f32,
}

fn speeds(path: &[Passing], slowest: f32) -> Vec<f32> {
    let run = |index: usize| {
        let (a, b) = (path[index], path[index + 1]);
        max(a.at.distance(b.at) / max(b.time - a.time, 1.0e-3), slowest)
    };
    let last = path.len() - 2;
    (0..path.len())
        .map(|index| 0.5 * (run(index.saturating_sub(1).min(last)) + run(index.min(last))))
        .collect()
}

fn skimmed(path: &[Passing], reach: f32) -> Vec<f32> {
    let mut along = vec![0.0_f32; path.len()];
    for index in 1..path.len() {
        along[index] = along[index - 1] + path[index].at.distance(path[index - 1].at);
    }
    let (mut first, mut last) = (0, 0);
    (0..path.len())
        .map(|index| {
            while along[index] - along[first] > reach {
                first += 1;
            }
            while last + 1 < path.len() && along[last + 1] - along[index] <= reach {
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

pub(crate) fn smooth(low: f32, high: f32, value: f32) -> f32 {
    let t = clamp((value - low) / (high - low), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn rows(path: &[Passing], carried: f32, look: &WaterFallSheets) -> Vec<Row> {
    let falling = skimmed(path, look.skim_reach);
    let speeds = speeds(path, look.slowest);
    let breakup = BREAKUP * pow(max(carried, 1.0e-3), BREAKUP_POWER);
    let [starts, grows, most] = look.inception;
    let inception = min(starts + grows * carried.sqrt(), most);
    let mut rows = Vec::new();
    let (mut height, mut took_off, mut leaving, mut airborne_white) = (0.0, 0.0, 0.0, 0.0);
    let (mut white, mut clung, mut before, mut along) = (0.0, 0.0, path[0].time, 0.0);
    let last = path.len() - 2;
    for index in 0..=last {
        let (a, b) = (path[index], path[index + 1]);
        let flew = index > 0 && path[index - 1].aloft;
        let run = b.at - a.at;
        if a.aloft && !flew {
            took_off = a.time;
            height = a.at.y;
            airborne_white = white;
            leaving = Vec2::new(run.x, run.z).length() / max(b.time - a.time, 1.0e-3);
        }
        if !a.aloft && flew {
            white = 1.0;
        }
        let length = run.length();
        let steps = max((length / look.row).ceil(), 1.0) as u32;
        for step in 0..steps + u32::from(index == last) {
            let share = step as f32 / steps as f32;
            let time = a.time + (b.time - a.time) * share;
            let center = a.at + run * share;
            let (mut flying, mut dropped) = (0.0, 0.0);
            if a.aloft {
                flying = time - took_off;
                dropped = max(height - center.y, 0.0);
                white = max(airborne_white, 1.0 - exp(-flying / look.airing));
            } else {
                clung += length / steps as f32;
                let sine = falling[index] + (falling[index + 1] - falling[index]) * share;
                let [onset, full] = look.aired;
                let churning =
                    smooth(0.0, inception, clung) * smooth(onset, full, look.air_share * sine);
                let lingering = look.lingering;
                let released = lingering
                    + (white - lingering) * exp(-max(time - before, 0.0) / look.air_release);
                white = max(churning, if white > lingering { released } else { white });
            }
            before = time;
            rows.push(Row {
                center,
                tangent: run.normalize_or(Vec3::NEG_Y),
                speed: speeds[index] + (speeds[index + 1] - speeds[index]) * share,
                fallen: time - path[0].time,
                free: if a.aloft { 1.0 } else { 0.0 },
                bowed: look.bow * leaving * flying,
                white,
                broken: smooth(look.breaking_from * breakup, breakup, dropped),
                spread: look.edge_spread * dropped,
                half: a.half + (b.half - a.half) * share,
                sky: a.sky + (b.sky - a.sky) * share,
                along: along + length * share,
            });
        }
        along += length;
    }
    rows
}

fn leaps(rows: &[Row]) -> Vec<Option<[f32; 2]>> {
    let mut leaps = vec![None; rows.len()];
    let mut index = 0;
    while index < rows.len() {
        if rows[index].free <= 0.0 {
            index += 1;
            continue;
        }
        let first = index;
        while index < rows.len() && rows[index].free > 0.0 {
            index += 1;
        }
        let landing = index.min(rows.len() - 1);
        let leap = [rows[first].along, rows[landing].along];
        leaps[first..=landing].fill(Some(leap));
    }
    leaps
}

fn fall_sheet(
    fall: &Fall,
    origin: Vec3,
    look: &WaterFallSheets,
    vertices: &mut Vec<FallVertex>,
    indices: &mut Vec<u32>,
) {
    let columns = look.columns;
    let Some(path) = passings(fall) else {
        return;
    };
    let heading = Vec3::new(
        fall.bottom[0] - fall.top[0],
        0.0,
        fall.bottom[2] - fall.top[2],
    )
    .normalize_or(Vec3::X);
    let rows = rows(&path, fall.discharge / max(fall.width, 0.1), look);
    let leaps = leaps(&rows);
    let mut before: Option<(u32, [f32; 2])> = None;
    for (index, row) in rows.iter().enumerate() {
        let Some(leap) = leaps[index] else {
            before = None;
            continue;
        };
        let previous = rows[index.saturating_sub(1)].tangent;
        let tangent = (row.tangent + previous).normalize_or(row.tangent);
        let level = Vec3::new(tangent.x, 0.0, tangent.z);
        let across = if level.length() > 0.05 {
            Vec3::new(-level.z, 0.0, level.x).normalize()
        } else {
            Vec3::new(-heading.z, 0.0, heading.x)
        };
        let normal = across.cross(tangent).normalize_or(Vec3::Y);
        let width = row.half * look.overhang + row.spread;
        let carried = fall.discharge / max(2.0 * row.half, 0.1);
        let thickness = clamp(carried / row.speed, look.thickness[0], look.thickness[1]);
        let reach = max(look.springing * thickness, 1.0e-3);
        let rising = smooth(0.0, reach, row.along - leap[0]);
        let sinking = smooth(0.0, reach, leap[1] - row.along);
        let base = vertices.len() as u32;
        for column in 0..=columns {
            let side = column as f32 / columns as f32 * 2.0 - 1.0;
            let curl = 0.5 * thickness * side * side;
            let bow = (1.0 - side * side) * row.bowed;
            let position = origin + row.center + across * (side * width) - normal * curl
                + level.normalize_or(Vec3::ZERO) * bow;
            vertices.push(FallVertex {
                position: position.to_array(),
                normal: normal.to_array(),
                sheet: [side * width, row.half, row.spread, thickness],
                state: [row.white, row.broken],
                sky: row.sky,
                terrain: 0,
                flight: [
                    row.fallen,
                    row.speed,
                    carried * rising * sinking,
                    row.free,
                ],
            });
        }
        if let Some((previous, _)) = before.filter(|&(_, joined)| joined == leap) {
            for column in 0..columns {
                let (a, b) = (previous + column, base + column);
                indices.extend_from_slice(&[a, a + 1, b + 1, a, b + 1, b]);
            }
        }
        before = Some((base, leap));
    }
}

const FALL_ATTRIBUTES: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
    0 => Float32x3,
    1 => Float32x3,
    2 => Float32x4,
    3 => Float32x2,
    4 => Float32,
    5 => Uint32,
    6 => Float32x4,
];

pub(crate) const FALL_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: size_of::<FallVertex>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &FALL_ATTRIBUTES,
};
