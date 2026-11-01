use std::ops::Range;

use glam::{Vec2, Vec3};
use engine_core::math::{clamp, exp, max, min, pow};
use engine_heightfield::Fall;
use engine_render::wgpu;

use crate::gpu::FallVertex;

const ROW: f32 = 0.75;
const COLUMNS: u32 = 8;
const EDGE_SPREAD: f32 = 0.035;
const BREAKUP: f32 = 6.0;
const INCEPTION: [f32; 3] = [1.0, 2.0, 6.0];
const AIRING: f32 = 0.4;
const AIR_SHARE: f32 = 0.9;
const AIRED: [f32; 2] = [0.15, 0.55];
const SKIM_REACH: f32 = 6.0;
const AIR_RELEASE: f32 = 1.2;
const LINGERING: f32 = 0.3;
const THICKNESS: [f32; 2] = [0.05, 1.2];
const OVERHANG: f32 = 1.15;
const BOW: f32 = 0.12;
const SLOWEST: f32 = 0.3;

pub(crate) fn sheets<'a>(
    falls: impl IntoIterator<Item = (Vec3, &'a [Fall], u32)>,
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
                fall_jets(fall, origin, vertices, indices);
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

fn on_segment(a: &Passing, b: &Passing, share: f32) -> (Vec3, Vec3, f32) {
    let run = b.at - a.at;
    let seconds = max(b.time - a.time, 1.0e-3);
    let speed = max(run.length() / seconds, SLOWEST);
    (a.at + run * share, run.normalize_or(Vec3::NEG_Y), speed)
}

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

fn smooth(low: f32, high: f32, value: f32) -> f32 {
    let t = clamp((value - low) / (high - low), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn rows(path: &[Passing], carried: f32) -> Vec<Vec<Row>> {
    let falling = skimmed(path);
    let breakup = BREAKUP * pow(max(carried, 1.0e-3), 0.32);
    let inception = min(INCEPTION[0] + INCEPTION[1] * carried.sqrt(), INCEPTION[2]);
    let mut jets: Vec<Vec<Row>> = Vec::new();
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
        let lands = a.aloft && (!b.aloft || index + 2 == path.len());
        for step in 0..steps + u32::from(lands) {
            let share = step as f32 / steps as f32;
            let time = a.time + (b.time - a.time) * share;
            if a.aloft {
                let (center, tangent, speed) = on_segment(&a, &b, share);
                let fallen = max(height - center.y, 0.0);
                let flying = time - took_off;
                white = max(airborne_white, 1.0 - exp(-flying / AIRING));
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
                let released =
                    LINGERING + (white - LINGERING) * exp(-max(time - before, 0.0) / AIR_RELEASE);
                white = max(churning, if white > LINGERING { released } else { white });
            }
            before = time;
        }
    }
    jets
}

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
            let previous = jet[index.saturating_sub(1)].tangent;
            let tangent = (row.tangent + previous).normalize_or(row.tangent);
            let level = Vec3::new(tangent.x, 0.0, tangent.z);
            let across = if level.length() > 0.05 {
                Vec3::new(-level.z, 0.0, level.x).normalize()
            } else {
                Vec3::new(-heading.z, 0.0, heading.x)
            };
            let normal = across.cross(tangent).normalize_or(Vec3::Y);
            let width = row.half * OVERHANG + row.spread;
            let thickness = clamp(carried / row.speed, THICKNESS[0], THICKNESS[1]);
            let base = vertices.len() as u32;
            for column in 0..=COLUMNS {
                let side = column as f32 / COLUMNS as f32 * 2.0 - 1.0;
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
                    terrain: 0,
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

pub(crate) fn fall_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x4,
        3 => Float32x2,
        4 => Float32,
        5 => Uint32,
    ];
    wgpu::VertexBufferLayout {
        array_stride: size_of::<FallVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    }
}
