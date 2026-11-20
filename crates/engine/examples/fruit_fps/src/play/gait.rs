use std::f32::consts::{FRAC_PI_2, PI};

use engine_core::math::{atan2, clamp, max, min, sin_cos};
use engine_math::wrap_angle;
use glam::Vec3;
use serde::Deserialize;

use crate::play::skinned::{Blended, ClipNo, Stride};

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Band {
    pub min: f32,
    pub max: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Heading {
    Forward,
    Left,
    Right,
    Back,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gait {
    pub clip: String,
    pub planted_mps: f32,
    pub direction: Heading,
    pub from_mps: f32,
    pub to_mps: f32,
    pub crouch: bool,
    pub dropped: bool,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gaits {
    pub gaits: Vec<Gait>,
    pub speed_fade_mps: f32,
    pub fade_seconds: f32,
    pub hip_turn_limit_radians: f32,
    pub rate: Band,
    pub stride: Band,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gaited {
    pub clip: ClipNo,
    pub blend: Vec<Blended>,
    pub rate: f32,
    pub stride: Stride,
    pub hip_turn: f32,
    pub speed: f32,
    pub planted: f32,
    pub planned_slide: f32,
    pub dropped: bool,
}

fn heading_of(quarter: i32) -> Heading {
    match quarter.rem_euclid(4) {
        0 => Heading::Forward,
        1 => Heading::Left,
        3 => Heading::Right,
        _ => Heading::Back,
    }
}

fn travel(velocity: Vec3, forward: Vec3) -> (f32, f32) {
    let flat = Vec3::new(velocity.x, 0.0, velocity.z);
    let left = Vec3::new(forward.z, 0.0, -forward.x);
    let speed = flat.length();
    if speed <= 0.0 {
        return (0.0, 0.0);
    }
    (speed, atan2(flat.dot(left), flat.dot(forward)))
}

fn speed_share(gait: &Gait, speed: f32, fade: f32) -> f32 {
    let gap = if speed < gait.from_mps {
        gait.from_mps - speed
    } else if speed > gait.to_mps {
        speed - gait.to_mps
    } else {
        0.0
    };
    if gap <= 0.0 {
        1.0
    } else if fade > 0.0 {
        max(1.0 - gap / fade, 0.0)
    } else {
        0.0
    }
}

type Has<'a> = &'a [Option<ClipNo>];

fn serves(table: &Gaits, (wanted, crouch): (Heading, bool), has: Has<'_>) -> bool {
    (table.gaits.iter().zip(has))
        .any(|(gait, clip)| gait.direction == wanted && gait.crouch == crouch && clip.is_some())
}

type Rows<'a> = Vec<(&'a Gait, ClipNo, f32)>;

fn gaits_for<'a>(
    table: &'a Gaits,
    (heading, crouch): (Heading, bool),
    (speed, weight): (f32, f32),
    (has, rows): (Has<'_>, &mut Rows<'a>),
) {
    let serving = |wanted: Heading| {
        (table.gaits.iter().zip(has)).filter_map(move |(gait, clip)| {
            let clip = clip.filter(|_| gait.direction == wanted && gait.crouch == crouch)?;
            Some((gait, clip))
        })
    };
    let found = if serving(heading).next().is_some() {
        heading
    } else {
        Heading::Forward
    };
    let share = |gait: &Gait| speed_share(gait, speed, table.speed_fade_mps);
    let total: f32 = serving(found).map(|(gait, _)| share(gait)).sum();
    if total > 0.0 {
        rows.extend(
            serving(found).map(|(gait, clip)| (gait, clip, weight * (share(gait) / total))),
        );
        return;
    }
    let nearest = serving(found).min_by(|(a, _), (b, _)| {
        let gap = |gait: &Gait| min((gait.from_mps - speed).abs(), (gait.to_mps - speed).abs());
        gap(a).total_cmp(&gap(b))
    });
    rows.extend(nearest.map(|(gait, clip)| (gait, clip, weight)));
}

fn weighed<'a>(
    table: &'a Gaits,
    crouch: bool,
    (speed, residual): (f32, f32),
    has: Has<'_>,
) -> Rows<'a> {
    let quarter = residual / FRAC_PI_2;
    let low = quarter.floor();
    let share = quarter - low;
    let low = low as i32;
    let mut rows = Vec::new();
    for (heading, weight) in [(heading_of(low), 1.0 - share), (heading_of(low + 1), share)] {
        if weight <= 0.0 {
            continue;
        }
        gaits_for(table, (heading, crouch), (speed, weight), (has, &mut rows));
    }
    rows
}

fn nearest_served(table: &Gaits, crouch: bool, angle: f32, has: Has<'_>) -> f32 {
    let quarters = [
        (0.0, Heading::Forward),
        (FRAC_PI_2, Heading::Left),
        (-FRAC_PI_2, Heading::Right),
        (PI, Heading::Back),
        (-PI, Heading::Back),
    ];
    quarters
        .into_iter()
        .filter(|(_, heading)| serves(table, (*heading, crouch), has))
        .map(|(at, _)| at)
        .min_by(|a, b| (angle - a).abs().total_cmp(&(angle - b).abs()))
        .unwrap_or(0.0)
}

pub fn gait(
    table: &Gaits,
    crouch: bool,
    (velocity, forward): (Vec3, Vec3),
    has: Has<'_>,
) -> Option<Gaited> {
    let (speed, angle) = travel(velocity, forward);
    let nearest = nearest_served(table, crouch, angle, has);
    let limit = max(table.hip_turn_limit_radians, 0.0);
    let hip_turn = clamp(angle - nearest, -limit, limit);
    let residual = wrap_angle(angle - hip_turn);
    let rows = weighed(table, crouch, (speed, residual), has);
    let planted: f32 = rows
        .iter()
        .map(|(gait, _, weight)| gait.planted_mps * weight)
        .sum();
    let dropped_share: f32 = rows
        .iter()
        .filter(|(gait, ..)| gait.dropped)
        .map(|(.., weight)| weight)
        .sum();
    let mut clips: Vec<(ClipNo, f32)> = Vec::with_capacity(rows.len());
    for &(_, clip, weight) in &rows {
        match clips.iter_mut().find(|(held, _)| *held == clip) {
            Some((_, total)) => *total += weight,
            None => clips.push((clip, weight)),
        }
    }
    clips.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut clips = clips.into_iter();
    let (clip, _) = clips.next()?;
    let blend = clips
        .filter(|(_, weight)| *weight > 0.0)
        .map(|(clip, weight)| Blended { clip, weight })
        .collect();
    let wanted = if planted > 0.0 { speed / planted } else { 1.0 };
    let rate = clamp(wanted, table.rate.min, table.rate.max);
    let scale = clamp(
        wanted / max(rate, 1.0e-3),
        table.stride.min,
        table.stride.max,
    );
    let (sin, cos) = sin_cos(angle);
    Some(Gaited {
        clip,
        blend,
        rate,
        stride: Stride {
            scale,
            axis: [sin, cos],
        },
        hip_turn,
        speed,
        planted,
        planned_slide: (speed - planted * rate * scale).abs(),
        dropped: dropped_share >= 0.5,
    })
}
