use bevy_ecs::prelude::*;
use engine_core::math::{GRAVITY, clamp, max};
use engine_core::{Spawner, Transform};
use engine_heightfield::Fall;
use engine_places::particles::{ParticleBlend, ParticleEmitter, ParticleLook, ParticleShape};
use glam::Vec3;

use crate::WaterBrick;
use crate::shallow_water::DENSITY;
use crate::state::WaterState;

const SPLASH_WATTS: f32 = 2.0e3;
const SPLASH_RATE: [f32; 2] = [20.0, 600.0];
const MIST_WATTS: f32 = 1.5e5;
const MIST_RATE: [f32; 2] = [1.0, 12.0];
const BURST_WATTS: f32 = 6.0e4;
const BURST_RATE: [f32; 2] = [2.0, 24.0];

fn jets(fall: &Fall) -> Vec<(usize, usize, f32)> {
    let mut found = Vec::new();
    let mut leaving = None;
    for index in 0..fall.path.len() {
        let aloft = fall.aloft.get(index).copied().unwrap_or(false);
        match (leaving, aloft) {
            (None, true) => leaving = Some(index),
            (Some(lip), false) => {
                let (a, b) = (
                    Vec3::from_array(fall.path[index - 1]),
                    Vec3::from_array(fall.path[index]),
                );
                let seconds = fall.times.get(index).zip(fall.times.get(index - 1));
                let speed = seconds.map_or(fall.speed, |(after, before)| {
                    a.distance(b) / max(after - before, 1.0e-3)
                });
                found.push((lip, index, speed));
                leaving = None;
            }
            _ => {}
        }
    }
    found
}

fn jet(flight: Vec3, speed: f32, seconds: f32, width: f32, seed: u32) -> ParticleEmitter {
    ParticleEmitter {
        rate: 30.0 + 10.0 * width,
        look: ParticleLook {
            lifetime: seconds,
            speed,
            direction: flight.normalize_or(Vec3::X),
            spread: 0.2,
            acceleration: Vec3::new(0.0, -GRAVITY, 0.0),
            shape: ParticleShape::Ball,
            extent: Vec3::splat(0.45 * width),
            size: [0.05, 0.12],
            color: [[0.9, 0.93, 0.97, 0.6], [0.94, 0.96, 1.0, 0.15]],
            brightness: 1.0,
            blend: ParticleBlend::Alpha,
            stretch: 0.04,
            seed,
            ..ParticleLook::default()
        },
        ..ParticleEmitter::default()
    }
}

fn emitters(fall: &Fall, origin: Vec3, seed: u32) -> Vec<(Transform, ParticleEmitter)> {
    let width = fall.width;
    let at = |point: Vec3| Transform {
        translation: origin + point,
        ..Transform::default()
    };
    let point = |index: usize| Vec3::from_array(fall.path[index]);
    let sky = |index: usize| fall.sky.get(index).copied().unwrap_or(1.0);
    let mut landings = Vec::new();
    let mut spawned = Vec::new();
    for (number, &(lip, landing, landing_speed)) in jets(fall).iter().enumerate() {
        let (top, bottom) = (point(lip), point(landing));
        let flight = Vec3::new(bottom.x - top.x, 0.0, bottom.z - top.z);
        let (reach, drop) = (flight.length(), max(top.y - bottom.y, 0.1));
        let speed = max(reach * (GRAVITY / (2.0 * drop)).sqrt(), 0.5);
        let seconds = (2.0 * drop / GRAVITY).sqrt();
        let key = seed.wrapping_mul(97).wrapping_add(number as u32);
        spawned.push((at(top), jet(flight, speed, seconds, width, key)));
        landings.push((bottom, drop, landing_speed, sky(landing)));
    }
    let last = fall.path.len() - 1;
    let clinging = jets(fall)
        .last()
        .is_none_or(|&(_, landing, _)| landing + 1 < last);
    if last >= 1 && clinging && !fall.aloft.get(last - 1).copied().unwrap_or(false) {
        let from = landings.last().map_or(point(0).y, |&(landed, ..)| landed.y);
        landings.push((
            point(last),
            max(from - point(last).y, 0.1),
            fall.speed,
            sky(last),
        ));
    }
    let puffs = clamp(width / 10.0, 0.6, 2.5);
    for (number, &(landed, drop, landing_speed, open)) in landings.iter().enumerate() {
        let power = DENSITY * GRAVITY * fall.discharge * drop;
        let lit =
            |[red, green, blue, alpha]: [f32; 4]| [red * open, green * open, blue * open, alpha];
        let key = seed.wrapping_mul(131).wrapping_add(number as u32);
        spawned.push((
            at(landed),
            ParticleEmitter {
                rate: clamp(power / SPLASH_WATTS, SPLASH_RATE[0], SPLASH_RATE[1]),
                look: ParticleLook {
                    lifetime: 1.4,
                    speed: clamp(landing_speed * 0.4, 2.0, 9.0),
                    direction: Vec3::Y,
                    spread: 0.9,
                    acceleration: Vec3::new(0.0, -GRAVITY, 0.0),
                    shape: ParticleShape::Ball,
                    extent: Vec3::splat(0.35 * width),
                    size: [0.06, 0.16],
                    color: [lit([0.92, 0.95, 0.98, 0.65]), lit([0.95, 0.97, 1.0, 0.0])],
                    brightness: 1.0,
                    blend: ParticleBlend::Alpha,
                    stretch: 0.05,
                    seed: key.wrapping_mul(31) ^ 0x5bd1_e995,
                    ..ParticleLook::default()
                },
                ..ParticleEmitter::default()
            },
        ));
        spawned.push((
            at(landed + Vec3::Y * 0.5),
            ParticleEmitter {
                rate: clamp(power / BURST_WATTS, BURST_RATE[0], BURST_RATE[1]),
                look: ParticleLook {
                    lifetime: 1.6,
                    speed: clamp(landing_speed * 0.3, 2.0, 6.0),
                    direction: Vec3::Y,
                    spread: 1.45,
                    acceleration: Vec3::new(0.0, 0.6, 0.0),
                    shape: ParticleShape::Ball,
                    extent: Vec3::splat(0.3 * width),
                    size: [1.5 * puffs, 5.0 * puffs],
                    color: [lit([0.94, 0.96, 1.0, 0.25]), lit([1.0, 1.0, 1.0, 0.0])],
                    brightness: 1.0,
                    blend: ParticleBlend::Mist,
                    stretch: 0.0,
                    seed: key.wrapping_mul(7) ^ 0x2c1b_3c6d,
                    ..ParticleLook::default()
                },
                ..ParticleEmitter::default()
            },
        ));
        spawned.push((
            at(landed + Vec3::Y * 1.5),
            ParticleEmitter {
                rate: clamp(power / MIST_WATTS, MIST_RATE[0], MIST_RATE[1]),
                look: ParticleLook {
                    lifetime: 6.0,
                    speed: clamp(landing_speed * 0.12, 0.5, 2.5),
                    direction: Vec3::Y,
                    spread: 1.3,
                    acceleration: Vec3::new(0.0, 0.3, 0.0),
                    shape: ParticleShape::Ball,
                    extent: Vec3::splat(0.5 * width),
                    size: [4.0 * puffs, 11.0 * puffs],
                    color: [lit([0.94, 0.96, 1.0, 0.3]), lit([1.0, 1.0, 1.0, 0.0])],
                    brightness: 1.0,
                    blend: ParticleBlend::Mist,
                    stretch: 0.0,
                    seed: key ^ 0x9e37_79b9,
                    ..ParticleLook::default()
                },
                ..ParticleEmitter::default()
            },
        ));
    }
    spawned
}

pub(crate) fn sync(mut state: ResMut<WaterState>, mut spawner: Spawner<WaterBrick>) {
    let state = state.bypass_change_detection();
    for entity in state.gone.drain(..) {
        spawner.despawn(entity);
    }
    for terrain in &mut state.terrains {
        if !std::mem::take(&mut terrain.falls_fresh) {
            continue;
        }
        for entity in terrain.fall_entities.drain(..) {
            spawner.despawn(entity);
        }
        for (index, fall) in terrain.falls.iter().enumerate() {
            for (transform, emitter) in emitters(fall, terrain.origin, index as u32) {
                terrain
                    .fall_entities
                    .push(spawner.spawn((transform, emitter)));
            }
        }
    }
}
