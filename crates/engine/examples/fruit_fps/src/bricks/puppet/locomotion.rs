use std::f32::consts::{FRAC_PI_2, PI};

use engine_core::math::{atan2, clamp, max, min, sin_cos};
use glam::Vec3;

use crate::play::actors::{Body, Wish};
use crate::play::figures::{Blended, Stride};
use crate::play::puppet::{Gait, Heading, Locomotion};

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Moved {
    pub(super) clip: String,
    pub(super) blend: Vec<Blended>,
    pub(super) rate: f32,
    pub(super) stride: Stride,
    pub(super) hip_turn: f32,
    pub(super) speed: f32,
    pub(super) planted: f32,
    pub(super) planned_slide: f32,
    pub(super) dropped: bool,
}

fn heading_of(quarter: i32) -> Heading {
    match quarter.rem_euclid(4) {
        0 => Heading::Forward,
        1 => Heading::Left,
        3 => Heading::Right,
        _ => Heading::Back,
    }
}

fn travel(body: &Body, wish: &Wish) -> (f32, f32) {
    let flat = Vec3::new(body.velocity.x, 0.0, body.velocity.z);
    let (sin, cos) = sin_cos(wish.yaw_radians);
    let forward = Vec3::new(-sin, 0.0, -cos);
    let left = Vec3::new(-cos, 0.0, sin);
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

type Has<'a> = &'a dyn Fn(&str) -> bool;

fn serves(table: &Locomotion, (wanted, crouch): (Heading, bool), has: Has<'_>) -> bool {
    table
        .gaits
        .iter()
        .any(|gait| gait.direction == wanted && gait.crouch == crouch && has(&gait.clip))
}

fn gaits_for<'a>(
    table: &'a Locomotion,
    (heading, crouch): (Heading, bool),
    speed: f32,
    has: Has<'_>,
) -> Vec<(&'a Gait, f32)> {
    let serving = |wanted: Heading| {
        table.gaits.iter().filter(move |gait| {
            gait.direction == wanted && gait.crouch == crouch && has(&gait.clip)
        })
    };
    let found: Vec<&Gait> = if serving(heading).next().is_some() {
        serving(heading).collect()
    } else {
        serving(Heading::Forward).collect()
    };
    let shares: Vec<f32> = found
        .iter()
        .map(|gait| speed_share(gait, speed, table.speed_fade_mps))
        .collect();
    let total: f32 = shares.iter().sum();
    if total > 0.0 {
        return found
            .into_iter()
            .zip(shares)
            .map(|(gait, share)| (gait, share / total))
            .collect();
    }
    let nearest = found.into_iter().min_by(|a, b| {
        let gap = |gait: &Gait| min((gait.from_mps - speed).abs(), (gait.to_mps - speed).abs());
        gap(a).total_cmp(&gap(b))
    });
    nearest.map(|gait| vec![(gait, 1.0)]).unwrap_or_default()
}

fn weighed<'a>(
    table: &'a Locomotion,
    crouch: bool,
    (speed, residual): (f32, f32),
    has: Has<'_>,
) -> Vec<(&'a Gait, f32)> {
    let quarter = residual / FRAC_PI_2;
    let low = quarter.floor();
    let share = quarter - low;
    let low = low as i32;
    let mut rows = Vec::new();
    for (heading, weight) in [(heading_of(low), 1.0 - share), (heading_of(low + 1), share)] {
        if weight <= 0.0 {
            continue;
        }
        for (gait, part) in gaits_for(table, (heading, crouch), speed, has) {
            rows.push((gait, weight * part));
        }
    }
    rows
}

fn nearest_served(table: &Locomotion, crouch: bool, angle: f32, has: Has<'_>) -> f32 {
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

pub(super) fn moved(
    table: &Locomotion,
    crouch: bool,
    (body, wish): (&Body, &Wish),
    has: Has<'_>,
) -> Option<Moved> {
    let (speed, angle) = travel(body, wish);
    let nearest = nearest_served(table, crouch, angle, has);
    let limit = max(table.hip_turn_limit_radians, 0.0);
    let hip_turn = clamp(angle - nearest, -limit, limit);
    let residual = (angle - hip_turn + PI).rem_euclid(2.0 * PI) - PI;
    let rows = weighed(table, crouch, (speed, residual), has);
    let planted: f32 = rows
        .iter()
        .map(|(gait, weight)| gait.planted_mps * weight)
        .sum();
    let dropped_share: f32 = rows
        .iter()
        .filter(|(gait, _)| gait.dropped)
        .map(|(_, weight)| weight)
        .sum();
    let mut clips: Vec<(String, f32)> = Vec::new();
    for (gait, weight) in &rows {
        match clips.iter_mut().find(|(clip, _)| *clip == gait.clip) {
            Some((_, total)) => *total += weight,
            None => clips.push((gait.clip.clone(), *weight)),
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
    Some(Moved {
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
