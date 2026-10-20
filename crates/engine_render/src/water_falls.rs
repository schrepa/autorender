//! Waterfalls' sheets: where terrains' rivers leave their beds
//! (`engine_core::terrain_data::Fall`), the water pass draws the water's
//! path as a sheet the river's width, from a little up the river over its
//! lip, through the air on each jet and down the chutes between, on into
//! the water it lands in at last: widening as it flies, its edges curling
//! back toward the rock as thick as it stands, lying on the chutes' water.
//! Each vertex holds where across the sheet it lies, the seconds since the
//! water passing it left the lip, since it last took off or landed, and how
//! fast it runs there, so the shader (`meshes/water_falls.wgsl`) draws what the
//! water carries where it has carried it: streaks falling and stretching as
//! the water speeds up, white water breaking up as it flies and boiling
//! where it lands. How white the water has turned and how far the air has
//! broken it up follow the falling water's physics: its sheet leaves a lip
//! as thick as the water each meter of it carries over its speed, its edges
//! spread a share of how far it has fallen, and its core lasts a fall that
//! grows with what it carries (Horeni, after USBR 2015); clinging to rock,
//! it turns white once its turbulence reaches its surface (Chanson's
//! inception point) as far as the rock is steep enough to keep air in it,
//! and lets its air go down gentler rock, its foam lingering in patches.
//! Off a step's face high enough it leaps clear as a nappe, leaving the lip
//! level and falling as it goes, flying as a jet does.
//! Where a jet lands, the water boils: a disc of white
//! water on the pool, flowing out from where it lands and thinning to
//! nothing at its rim.

use engine_core::glam::{Vec2, Vec3};
use engine_core::math::{clamp, max, min};
use engine_core::terrain_data::{Fall, NAPPE_FACE};

use crate::gpu_layouts::water::FallVertex;

/// Meters between the sheet's rows along its path, and its columns across.
const ROW: f32 = 0.75;
const COLUMNS: u32 = 8;
/// Meters the sheet reaches up the river over its lip, and on down into
/// the water it lands in.
const UPSTREAM: f32 = 1.5;
const PLUNGE: f32 = 1.5;
/// The meters each edge of a sheet spreads a meter it falls (2 to 5%).
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
/// the rock falls overall rather than step by step; landing, it boils
/// white, and it lets its air go down gentler rock over AIR_RELEASE
/// seconds, down to LINGERING white: its foam's patches.
const AIR_SHARE: f32 = 0.9;
const AIRED: [f32; 2] = [0.15, 0.55];
const SKIM_REACH: f32 = 6.0;
const AIR_RELEASE: f32 = 1.2;
const LINGERING: f32 = 0.3;
/// The thinnest and thickest the sheet stands (meters): the water each
/// meter of its width carries over its speed.
const THICKNESS: [f32; 2] = [0.05, 1.2];
/// Meters the sheet stands off the river's water where it lies on it, and
/// the share of its thickness more it stands where it has turned white
/// (aerated water bulks out over the clear water's surface, nearly three
/// times as deep on 45-degree rock), at most LIFTED meters.
const LIFT: f32 = 0.05;
const BULKING: f32 = 0.8;
const LIFTED: f32 = 0.4;
/// Seconds over which a jet leaving a chute settles from the chute's lift
/// to its own path, and the share wider than its river a sheet stands, so
/// its edges fade over the banks rather than short of the water.
const SETTLING: f32 = 0.4;
const OVERHANG: f32 = 1.15;
/// How much faster a jet's middle leaves its lip than its edges (a share
/// of its speed): its curtain bows out in the middle as it falls.
const BOW: f32 = 0.12;
/// The slowest the sheet's water runs (meters a second), and the gravity a
/// nappe falls under (meters a second squared).
const SLOWEST: f32 = 0.3;
const GRAVITY: f32 = 9.81;
/// What the seconds since landing read before the water has landed.
const UNLANDED: f32 = 1.0e3;
/// A boil's radius, a share of the fall's width between the least and the
/// most meters; its rings and segments; the share of the landing speed its
/// water flows out at, between the slowest and fastest meters a second; how
/// far over the pool it stands (meters); and the share of its radius from
/// which it thins out to nothing at its rim.
const BOIL_SIZE: [f32; 3] = [0.6, 2.0, 15.0];
const BOIL_RINGS: u32 = 6;
const BOIL_SEGMENTS: u32 = 16;
const BOIL_OUTFLOW: [f32; 3] = [0.35, 1.0, 4.0];
const BOIL_LIFT: f32 = 0.08;
const BOIL_RIM: f32 = 0.55;

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
            boils(fall, origin, number, vertices, indices);
            number += 1;
        }
    }
}

/// A point of a fall's path: where it stands (terrain-local), the seconds
/// after leaving the lip its water passes it, whether it flies from there
/// to the next point, through the air or clear of a step's face (a nappe),
/// half the meters across its water there and how much of the sky it sees.
#[derive(Clone, Copy)]
struct Passing {
    at: Vec3,
    time: f32,
    aloft: bool,
    nappe: bool,
    half: f32,
    sky: f32,
}

impl Passing {
    /// Whether its water flies from here to the next point.
    fn flies(&self) -> bool {
        self.aloft || self.nappe
    }
}

/// Fall `fall`'s path, reaching UPSTREAM up the river over its lip at the
/// speed it leaves at and PLUNGE on into the water it lands in as it lands,
/// with that speed (meters a second); `None` for a path too short to draw.
fn passings(fall: &Fall) -> Option<(Vec<Passing>, f32)> {
    let count = fall.path.len();
    if count < 2
        || fall.times.len() != count
        || fall.aloft.len() != count
        || fall.nappes.len() != count
        || fall.widths.len() != count
        || fall.sky.len() != count
    {
        return None;
    }
    let mut path: Vec<Passing> = (0..count)
        .map(|index| Passing {
            at: Vec3::from_array(fall.path[index]),
            time: fall.times[index],
            aloft: fall.aloft[index],
            nappe: fall.nappes[index],
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
    let upstream = Vec3::new(leaving.x, 0.0, leaving.z).normalize_or(Vec3::X);
    path.insert(
        0,
        Passing {
            at: first.at - upstream * UPSTREAM,
            time: -UPSTREAM / lip_speed,
            aloft: false,
            nappe: false,
            half: first.half,
            sky: first.sky,
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
        nappe: false,
        half: last.half,
        sky: last.sky,
    });
    Some((
        landed_nappes(&path, fall.discharge / max(fall.width, 0.1)),
        lip_speed,
    ))
}

/// `path` with each nappe landing where it carries its water, carrying
/// `carried` cubic meters a second each meter of its width: leaving its lip
/// level at least as fast as its waves run, as far out as it falls in the
/// time it takes to fall its step's height, no nearer than the face's foot
/// ([`NAPPE_FACE`] of the stretch) nor past the stretch; on from there down
/// the rock below.
fn landed_nappes(path: &[Passing], carried: f32) -> Vec<Passing> {
    let critical = libm::cbrtf(carried * carried / GRAVITY);
    let mut out = Vec::with_capacity(path.len() + 4);
    for (index, &a) in path.iter().enumerate() {
        out.push(a);
        let Some(&b) = path.get(index + 1).filter(|_| a.nappe) else {
            continue;
        };
        let run = Vec2::new(b.at.x - a.at.x, b.at.z - a.at.z).length();
        let (drop, seconds) = (a.at.y - b.at.y, b.time - a.time);
        if run <= 0.0 || drop <= 0.0 || seconds <= 0.0 {
            continue;
        }
        let speed = max(run / seconds, (GRAVITY * critical).sqrt());
        let share = clamp(speed * (2.0 * drop / GRAVITY).sqrt() / run, NAPPE_FACE, 1.0);
        if share < 0.99 {
            let at = a.at + (b.at - a.at) * share;
            out.push(Passing {
                at: Vec3::new(at.x, b.at.y, at.z),
                time: a.time + seconds * share,
                aloft: false,
                nappe: false,
                half: a.half + (b.half - a.half) * share,
                sky: a.sky + (b.sky - a.sky) * share,
            });
        }
    }
    out
}

/// A row of a sheet: its middle (terrain-local), the seconds after leaving
/// the lip its water passes it, the way it runs and how fast, the seconds
/// since it took off where it flies, less those since it landed where it
/// runs down a chute, whether it flies, how white its water has turned and
/// how far the air has broken it up (0 to 1 each), and the meters each of
/// its edges has spread.
#[derive(Clone, Copy)]
struct Row {
    center: Vec3,
    time: f32,
    tangent: Vec3,
    speed: f32,
    flying: f32,
    aloft: bool,
    white: f32,
    broken: f32,
    spread: f32,
    lift: f32,
    half: f32,
    sky: f32,
}

/// Where the water passing `share` of the way from `a` to `b` is, the way it
/// runs there and how fast: straight along the way between them, or clear
/// of a step's face on the curve of water leaving the lip level and falling
/// as it goes.
fn on_segment(a: &Passing, b: &Passing, share: f32) -> (Vec3, Vec3, f32) {
    let run = b.at - a.at;
    let seconds = max(b.time - a.time, 1.0e-3);
    if !a.nappe {
        let speed = max(run.length() / seconds, SLOWEST);
        return (a.at + run * share, run.normalize_or(Vec3::NEG_Y), speed);
    }
    let drop = a.at.y - b.at.y;
    let level = Vec3::new(run.x, 0.0, run.z);
    let center = a.at + level * share - Vec3::Y * (drop * share * share);
    let velocity = (level - Vec3::Y * (2.0 * drop * share)) / seconds;
    (
        center,
        velocity.normalize_or(Vec3::NEG_Y),
        max(velocity.length(), SLOWEST),
    )
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

/// A sheet's rows along `path`, carrying `carried` cubic meters a second
/// each meter of its width: every ROW meters, and at each of its points.
fn rows(path: &[Passing], carried: f32) -> Vec<Row> {
    let falling = skimmed(path);
    let breakup = BREAKUP * libm::powf(max(carried, 1.0e-3), 0.32);
    let inception = min(INCEPTION[0] + INCEPTION[1] * carried.sqrt(), INCEPTION[2]);
    let mut rows = Vec::new();
    let (mut took_off, mut landed) = (0.0, -UNLANDED);
    // Where it last took off, how white it was then, and the meters it
    // has clung to rock.
    let (mut height, mut airborne_white, mut clung) = (0.0, 0.0, 0.0);
    let (mut white, mut lift, mut lifted) = (0.0, LIFT, LIFT);
    let mut before = path[0].time;
    for index in 0..path.len() - 1 {
        let (a, b) = (path[index], path[index + 1]);
        let flew = index > 0 && path[index - 1].flies();
        if a.flies() && !flew {
            took_off = a.time;
            height = a.at.y;
            airborne_white = white;
            lifted = lift;
        }
        if !a.flies() && flew {
            landed = a.time;
            white = 1.0;
        }
        let length = (b.at - a.at).length();
        let steps = max((length / ROW).ceil(), 1.0) as u32;
        for step in 0..steps {
            let share = step as f32 / steps as f32;
            let time = a.time + (b.time - a.time) * share;
            let (center, tangent, speed) = on_segment(&a, &b, share);
            let (broken, spread);
            if a.flies() {
                let fallen = max(height - center.y, 0.0);
                let flying = time - took_off;
                white = max(airborne_white, 1.0 - libm::expf(-flying / AIRING));
                broken = smooth(0.3 * breakup, breakup, fallen);
                spread = EDGE_SPREAD * fallen;
                lift = lifted * (1.0 - smooth(0.0, SETTLING, flying));
            } else {
                // Clinging to rock it turns white once its turbulence
                // reaches its surface, as far as the rock is steep enough
                // to keep air in it; white water lets its air go down
                // gentler rock, its foam lingering in patches.
                if time >= 0.0 {
                    clung += length / steps as f32;
                }
                let surfaced = if time >= 0.0 {
                    smooth(0.0, inception, clung)
                } else {
                    0.0
                };
                let sine = falling[index] + (falling[index + 1] - falling[index]) * share;
                let churning = surfaced * smooth(AIRED[0], AIRED[1], AIR_SHARE * sine);
                let released = LINGERING
                    + (white - LINGERING) * libm::expf(-max(time - before, 0.0) / AIR_RELEASE);
                white = max(churning, if white > LINGERING { released } else { white });
                broken = 0.0;
                spread = 0.0;
                let thickness = clamp(carried / speed, THICKNESS[0], THICKNESS[1]);
                lift = LIFT + min(BULKING * white * thickness, LIFTED);
            }
            before = time;
            rows.push(Row {
                center,
                time,
                tangent,
                speed,
                flying: if a.flies() {
                    time - took_off
                } else {
                    landed - time
                },
                aloft: a.flies(),
                white,
                broken,
                spread,
                lift,
                half: a.half + (b.half - a.half) * share,
                sky: a.sky + (b.sky - a.sky) * share,
            });
        }
    }
    if let (Some(&end), Some(&last)) = (path.last(), rows.last()) {
        rows.push(Row {
            center: end.at,
            time: end.time,
            half: end.half,
            sky: end.sky,
            flying: landed - end.time,
            aloft: false,
            broken: 0.0,
            spread: 0.0,
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
    let carried = fall.discharge / max(fall.width, 0.1);
    let heading = Vec3::new(
        fall.bottom[0] - fall.top[0],
        0.0,
        fall.bottom[2] - fall.top[2],
    )
    .normalize_or(Vec3::X);
    let rows = rows(&path, carried);
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
        let half = row.half * OVERHANG;
        let width = half + row.spread;
        let thickness = clamp(carried / row.speed, THICKNESS[0], THICKNESS[1]);
        let lift = row.lift;
        let base = vertices.len() as u32;
        for column in 0..=COLUMNS {
            let side = column as f32 / COLUMNS as f32 * 2.0 - 1.0;
            let curl = if row.aloft {
                0.5 * thickness * side * side
            } else {
                0.0
            };
            let bow = if row.aloft {
                BOW * (1.0 - side * side) * lip_speed * row.flying
            } else {
                0.0
            };
            let position = origin
                + row.center
                + across * (side * width)
                + normal * (lift - curl)
                + level.normalize_or(Vec3::ZERO) * bow;
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
                state: [row.white, row.broken, 1.0, 0.0],
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

/// The boils where fall `fall`'s jets land (its terrain at `origin`, its
/// number `number`): a disc on the pool each, its water flowing out from
/// where the jet lands, as white as the boil it churns, thinning out at its
/// rim; its seam toward the rock the jet fell from, behind the jet.
fn boils(
    fall: &Fall,
    origin: Vec3,
    number: u32,
    vertices: &mut Vec<FallVertex>,
    indices: &mut Vec<u32>,
) {
    let radius = clamp(BOIL_SIZE[0] * fall.width, BOIL_SIZE[1], BOIL_SIZE[2]);
    for index in 1..fall.path.len() {
        if !(fall.aloft[index - 1] && !fall.aloft[index]) {
            continue;
        }
        let (before, at) = (
            Vec3::from_array(fall.path[index - 1]),
            Vec3::from_array(fall.path[index]),
        );
        let seconds = max(fall.times[index] - fall.times[index - 1], 1.0e-3);
        let landing = before.distance(at) / seconds;
        let outflow = clamp(BOIL_OUTFLOW[0] * landing, BOIL_OUTFLOW[1], BOIL_OUTFLOW[2]);
        // The pool's water: the river's where it runs on from the landing.
        let level = fall
            .path
            .get(index + 1)
            .filter(|_| !fall.aloft[index])
            .map_or(at.y, |point| point[1]);
        let forward = Vec3::new(at.x - before.x, 0.0, at.z - before.z).normalize_or(Vec3::X);
        let right = Vec3::new(-forward.z, 0.0, forward.x);
        let center = origin + Vec3::new(at.x, level + BOIL_LIFT, at.z);
        let base = vertices.len() as u32;
        for ring in 0..=BOIL_RINGS {
            let out = ring as f32 / BOIL_RINGS as f32;
            let from = radius * out;
            let flown = from / outflow;
            let fade = 1.0 - smooth(BOIL_RIM, 1.0, out);
            for segment in 0..=BOIL_SEGMENTS {
                let side = segment as f32 / BOIL_SEGMENTS as f32 * 2.0 - 1.0;
                let (sin, cos) = engine_core::math::sin_cos(side * std::f32::consts::PI);
                let way = forward * cos + right * sin;
                vertices.push(FallVertex {
                    position: (center + way * from).to_array(),
                    normal: [0.0, 1.0, 0.0],
                    tangent: [way.x, way.y, way.z, outflow],
                    jet: [side, flown, outflow, radius],
                    fall: [-flown, THICKNESS[0] * 4.0, number as f32, 0.0],
                    state: [1.0, 0.0, fade, 1.0],
                    sky: fall.sky[index],
                });
            }
        }
        let across = BOIL_SEGMENTS + 1;
        for ring in 1..=BOIL_RINGS {
            for segment in 0..BOIL_SEGMENTS {
                let a = base + (ring - 1) * across + segment;
                let b = base + ring * across + segment;
                indices.extend_from_slice(&[a, a + 1, b + 1, a, b + 1, b]);
            }
        }
    }
}

/// The vertex buffer layout of [`FallVertex`]es.
pub(crate) fn fall_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 7] = crate::pipelines::vertex_attributes!(FallVertex {
        0 => position as Float32x3,
        1 => normal as Float32x3,
        2 => tangent as Float32x4,
        3 => jet as Float32x4,
        4 => fall as Float32x4,
        5 => state as Float32x4,
        6 => sky as Float32,
    });
    wgpu::VertexBufferLayout {
        array_stride: size_of::<FallVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    }
}
