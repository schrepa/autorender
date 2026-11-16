use bevy_ecs::prelude::*;
use engine_core::{Spawner, Transform};
use engine_heightfield::Fall;
use engine_math::{GRAVITY, atan2, clamp, max, min};
use engine_places_rendering::particles::{
    ParticleBlend, ParticleEmitter, ParticleLook, ParticleShape,
};
use glam::{Quat, Vec2, Vec3};

use crate::jobs::shallow_water::DENSITY;
use crate::settings::{WaterFallSheets, WaterFallSpray, WaterMist};
use crate::{WaterBrick, WaterPoint};

pub(crate) type FallLook = (WaterFallSheets, WaterFallSpray, WaterMist);

#[derive(Clone, Copy)]
struct Jet {
    lip: usize,
    landing: usize,
}

fn whole(fall: &Fall) -> bool {
    let count = fall.path.len();
    count >= 2
        && fall.times.len() == count
        && fall.aloft.len() == count
        && fall.widths.len() == count
}

fn jets(fall: &Fall) -> Vec<Jet> {
    let mut found = Vec::new();
    let mut leaving = None;
    for (index, &aloft) in fall.aloft.iter().enumerate() {
        match (leaving, aloft) {
            (None, true) => leaving = Some(index),
            (Some(lip), false) => {
                found.push(Jet {
                    lip,
                    landing: index,
                });
                leaving = None;
            }
            (None, false) | (Some(_), true) => {}
        }
    }
    found
}

fn level(fall: &Fall, index: usize) -> Vec2 {
    Vec2::new(fall.path[index][0], fall.path[index][2])
}

fn leaving_speed(fall: &Fall, jet: Jet, slowest: f32) -> f32 {
    let run = level(fall, jet.lip + 1).distance(level(fall, jet.lip));
    let seconds = max(fall.times[jet.lip + 1] - fall.times[jet.lip], 1.0e-3);
    max(run / seconds, slowest)
}

fn carried(fall: &Fall, jet: Jet) -> f32 {
    fall.discharge / max(fall.widths[jet.lip], 0.1)
}

fn drowned_jet(
    fall: &Fall,
    jet: Jet,
    slowest: f32,
    water: &dyn Fn(Vec2) -> Option<WaterPoint>,
) -> bool {
    let (lip, foot) = (level(fall, jet.lip), level(fall, jet.landing));
    let Some(below) = water(foot) else {
        return false;
    };
    let bed = |at: Vec2| water(at).map(|over| over.surface - over.depth);
    let crest = match (bed(lip), bed(lip + lip - foot)) {
        (Some(brink), Some(above)) => max(brink, above),
        (Some(only), None) | (None, Some(only)) => only,
        (None, None) => {
            fall.path[jet.lip][1] - carried(fall, jet) / leaving_speed(fall, jet, slowest)
        }
    };
    below.surface >= crest
}

pub(crate) fn judged_at(falls: &[Fall]) -> Vec<Vec2> {
    let mut points = Vec::new();
    for fall in falls.iter().filter(|fall| whole(fall)) {
        for jet in jets(fall) {
            let (lip, foot) = (level(fall, jet.lip), level(fall, jet.landing));
            points.extend([foot, lip, lip + lip - foot]);
        }
    }
    points
}

pub(crate) fn drowned(
    falls: &[Fall],
    slowest: f32,
    water: &dyn Fn(Vec2) -> Option<WaterPoint>,
) -> Vec<bool> {
    falls
        .iter()
        .filter(|fall| whole(fall))
        .flat_map(|fall| {
            jets(fall)
                .into_iter()
                .map(move |jet| drowned_jet(fall, jet, slowest, water))
        })
        .collect()
}

fn stretch(fall: &Fall, first: usize, last: usize) -> Fall {
    if first == 0 && last + 1 == fall.path.len() {
        return fall.clone();
    }
    let mut aloft = fall.aloft[first..=last].to_vec();
    if let Some(end) = aloft.last_mut() {
        *end = false;
    }
    let (ended, before) = (
        Vec3::from_array(fall.path[last]),
        Vec3::from_array(fall.path[last - 1]),
    );
    let seconds = max(fall.times[last] - fall.times[last - 1], 1.0e-3);
    Fall {
        top: fall.path[first],
        bottom: fall.path[last],
        path: fall.path[first..=last].to_vec(),
        times: fall.times[first..=last].to_vec(),
        aloft,
        widths: fall.widths[first..=last].to_vec(),
        sky: fall
            .sky
            .get(first..=last)
            .map_or_else(Vec::new, <[f32]>::to_vec),
        width: fall.widths[first],
        area: fall.area,
        speed: if last + 1 == fall.path.len() {
            fall.speed
        } else {
            ended.distance(before) / seconds
        },
        discharge: fall.discharge,
    }
}

pub(crate) fn falling(falls: &[Fall], drowned: &[bool]) -> Vec<Fall> {
    let mut verdicts = drowned.iter().copied();
    let mut found = Vec::new();
    for fall in falls.iter().filter(|fall| whole(fall)) {
        let (mut first, mut leaps) = (0, false);
        for jet in jets(fall) {
            if !verdicts.next().unwrap_or(false) {
                leaps = true;
                continue;
            }
            if leaps {
                found.push(stretch(fall, first, jet.lip));
            }
            (first, leaps) = (jet.landing, false);
        }
        if leaps {
            found.push(stretch(fall, first, fall.path.len() - 1));
        }
    }
    found
}

fn seed_of(fall: &Fall, number: usize) -> u32 {
    let [x, y, z] = fall.top.map(f32::to_bits);
    (x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b) ^ z.wrapping_mul(0xc2b2_ae35))
        .wrapping_add((number as u32).wrapping_mul(0x27d4_eb2f))
}

fn tinted([red, green, blue]: [f32; 3], open: f32, alpha: f32) -> [f32; 4] {
    [red * open, green * open, blue * open, alpha]
}

struct Foot {
    width: f32,
    struck: f32,
    landing: f32,
    power: f32,
    open: f32,
    seed: u32,
}

fn spray(foot: &Foot, look: &WaterFallSpray) -> ParticleEmitter {
    let thrown = clamp(look.throw * foot.landing, look.speed[0], look.speed[1]);
    let direction = Vec3::new(0.0, 1.0, look.forward).normalize();
    ParticleEmitter {
        rate: clamp(
            foot.power * foot.width / look.watts,
            look.rate[0],
            look.rate[1],
        ),
        look: ParticleLook {
            lifetime: 2.0 * thrown * direction.y / GRAVITY,
            speed: thrown,
            direction,
            spread: look.fan,
            acceleration: Vec3::new(0.0, -GRAVITY, 0.0),
            shape: ParticleShape::Box,
            extent: Vec3::new(look.foot_share * foot.width, 0.0, foot.struck),
            size: look.size,
            color: [
                tinted(look.color, foot.open, look.alpha),
                tinted(look.color, foot.open, 0.0),
            ],
            brightness: 1.0,
            blend: ParticleBlend::Alpha,
            stretch: look.stretch,
            seed: foot.seed,
            ..ParticleLook::default()
        },
        ..ParticleEmitter::default()
    }
}

fn veil(foot: &Foot, strength: f32, puff: f32, look: &WaterMist) -> ParticleEmitter {
    let alive = min(look.overlap * (1.0 + foot.width / puff), look.most);
    ParticleEmitter {
        rate: alive / look.seconds,
        look: ParticleLook {
            lifetime: look.seconds,
            lifetime_spread: look.uneven,
            speed: look.travel * puff / look.seconds,
            speed_spread: look.uneven,
            direction: Vec3::Y,
            spread: look.fan,
            acceleration: Vec3::ZERO,
            shape: ParticleShape::Box,
            extent: Vec3::new(0.5 * foot.width, look.spread * puff, look.spread * puff),
            size: [look.growth[0] * puff, look.growth[1] * puff],
            size_spread: look.uneven,
            color: [
                tinted(look.color, foot.open, look.alpha * strength),
                tinted(look.color, foot.open, 0.0),
            ],
            fade_in: look.fade_in,
            brightness: 1.0,
            blend: ParticleBlend::Mist,
            stretch: 0.0,
            seed: foot.seed ^ 0x9e37_79b9,
            ..ParticleLook::default()
        },
        ..ParticleEmitter::default()
    }
}

fn emitters(
    fall: &Fall,
    origin: Vec3,
    (sheets, droplets, mist): &FallLook,
) -> Vec<(Transform, ParticleEmitter)> {
    let point = |index: usize| Vec3::from_array(fall.path[index]);
    let mut spawned = Vec::new();
    for (number, jet) in jets(fall).into_iter().enumerate() {
        let (top, bottom) = (point(jet.lip), point(jet.landing));
        let heading = Vec3::new(bottom.x - top.x, 0.0, bottom.z - top.z).normalize_or(Vec3::X);
        let turned = Quat::from_rotation_y(atan2(heading.x, heading.z));
        let placed = |at: Vec3| Transform {
            translation: origin + at,
            rotation: turned,
            ..Transform::default()
        };
        let carried = carried(fall, jet);
        let leaving = leaving_speed(fall, jet, sheets.slowest);
        let drop = max(top.y - bottom.y, 0.0);
        let head = leaving * leaving / (2.0 * GRAVITY) + drop;
        let landing = (2.0 * GRAVITY * head).sqrt();
        let foot = Foot {
            width: max(fall.widths[jet.lip], 0.1),
            struck: carried / landing + sheets.edge_spread * drop,
            landing,
            power: DENSITY * GRAVITY * carried * head,
            open: fall.sky.get(jet.landing).copied().unwrap_or(1.0),
            seed: seed_of(fall, number),
        };
        spawned.push((placed(bottom), spray(&foot, droplets)));
        let strength = min((foot.power / mist.full_watts).sqrt(), 1.0);
        if strength >= mist.least {
            let puff = clamp(mist.puff_per_head * head, mist.puff[0], mist.puff[1]);
            spawned.push((
                placed(bottom + Vec3::Y * (mist.spread * puff)),
                veil(&foot, strength, puff, mist),
            ));
        }
    }
    spawned
}

pub(crate) fn spawn_fall(
    fall: &Fall,
    origin: Vec3,
    look: &FallLook,
    spawner: &mut Spawner<WaterBrick>,
) -> Vec<Entity> {
    emitters(fall, origin, look)
        .into_iter()
        .map(|(transform, emitter)| spawner.spawn((transform, emitter)))
        .collect()
}
