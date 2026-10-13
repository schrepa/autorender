//! What moves in the city, placed once its blocks stand.

use std::f32::consts::TAU;

use engine::prelude::*;

use super::shapes::{Shape, Tex};
use super::{
    City, NEON, RAIL_HEIGHT, WATER_TOP, WHITE, crane, drone, ferris, helicopter, look, vehicle,
};
use crate::kit::StressConfig;
use crate::kit::motion::{Motion, Slide, Spin, Travel};
use crate::kit::plan::{PERIOD, compose, part_at};
use crate::kit::random::{Rng, hash};

/// Every moving renderable: assemblies at the sites the blocks offered, then
/// trains, boats, aircraft, rooftop parts, buses and cars for the rest.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub(super) fn traffic(city: &mut City, config: &StressConfig) {
    let total = city.moving_left;
    let mut rng = Rng::new(config.seed ^ 0x7a11, 0, 0);
    let share = |percent: u32| total / 100 * percent + (total % 100) * percent / 100;
    // Cranes: a jib and its load.
    let cranes = std::mem::take(&mut city.cranes);
    for site in cranes.iter().take((share(3) / 2) as usize) {
        crane(city, &mut rng, site);
    }
    let wheels = std::mem::take(&mut city.wheels);
    for site in wheels.iter().take((share(3) / 9) as usize) {
        ferris(city, &mut rng, site);
    }
    if let Some(ring) = city.rail {
        let half = ring as f32 * PERIOD;
        let perimeter = 8.0 * half;
        let trains = math::min((share(8) / 5) as f32, perimeter / 60.0) as u32;
        let speed = rng.range(12.0, 16.0);
        let world = city.world();
        for train in 0..trains {
            let phase = perimeter * train as f32 / trains as f32;
            let color = rng.pick(&[[0.9, 0.92, 0.95], [0.95, 0.3, 0.2], [0.2, 0.5, 0.95]]);
            for wagon in 0..5 {
                let motion = Motion::Travel(Travel {
                    center: Vec3::new(0.0, RAIL_HEIGHT + 1.4, 0.0),
                    half: Vec2::splat(half),
                    speed,
                    phase: phase - wagon as f32 * 5.3,
                });
                let rest = part_at(Vec3::ZERO, Vec3::new(5.0, 1.8, 1.8));
                city.mover(
                    &world,
                    rest,
                    look(Shape::Wagon, Tex::Vehicle, color),
                    Some(motion),
                    Some(motion),
                );
            }
        }
    }
    if let Some((row, span)) = city.river {
        let world = city.world();
        let boats = share(4).min((span * 0.3) as u32);
        for _ in 0..boats {
            let half = Vec2::new(rng.range(12.0, math::max(span * 0.6, 14.0)), 3.5);
            let center = Vec3::new(
                rng.range(-span, span) * 0.5,
                WATER_TOP + 0.45,
                row as f32 * PERIOD + PERIOD * 0.5,
            );
            let motion = Motion::Travel(Travel {
                center,
                half,
                speed: rng.range(3.0, 7.0) * rng.sign(),
                phase: rng.range(0.0, 8.0 * (half.x + half.y)),
            });
            let color = rng.pick(&[WHITE, [0.95, 0.4, 0.3], [0.3, 0.6, 0.95], [0.95, 0.85, 0.3]]);
            city.mover(
                &world,
                part_at(Vec3::ZERO, Vec3::new(4.5, 1.2, 1.8)),
                look(Shape::Boat, Tex::Vehicle, color),
                Some(motion),
                Some(motion),
            );
        }
    }
    for _ in 0..share(3) / 2 {
        helicopter(city, &mut rng);
    }
    for _ in 0..share(14) / 2 {
        drone(city, &mut rng);
    }
    let rooftops = std::mem::take(&mut city.rooftops);
    if !rooftops.is_empty() {
        for index in 0..share(4) {
            let roof = rooftops[(hash(index ^ config.seed) as usize) % rooftops.len()];
            let at = roof.base + Vec3::Y * (roof.height + 1.3);
            let spin = Spin {
                base: Quat::IDENTITY,
                axis_z: false,
                speed: rng.range(1.5, 4.0) * rng.sign(),
                phase: rng.range(0.0, TAU),
            };
            let flat = Spin {
                base: roof.anchor.frame.rotation,
                ..spin
            };
            let size = math::min(roof.half * 1.6, 5.0);
            city.mover(
                &roof.anchor,
                part_at(at, Vec3::new(size, 2.6, size)),
                look(Shape::Beacon, Tex::Neon, rng.pick(&NEON)),
                Some(Motion::Spin(spin)),
                Some(Motion::Spin(flat)),
            );
        }
        for index in 0..share(4) {
            let roof = rooftops[(hash(index ^ config.seed ^ 0xe1e) as usize) % rooftops.len()];
            let side = Vec3::new(roof.half + 0.9, 0.0, 0.0);
            let anchor = roof.base + side + Vec3::Y * (roof.height * 0.5);
            let slide = Slide {
                anchor,
                axis: Vec3::Y,
                amplitude: roof.height * 0.35,
                speed: rng.range(0.3, 0.8),
                phase: rng.range(0.0, TAU),
            };
            let flat = Slide {
                anchor: compose(&roof.anchor.frame, &part_at(anchor, Vec3::ONE)).translation,
                axis: roof.anchor.frame.rotation * Vec3::Y,
                ..slide
            };
            city.mover(
                &roof.anchor,
                part_at(anchor, Vec3::new(1.4, 2.2, 1.4)),
                look(Shape::Block, Tex::Glass, rng.pick(&NEON)),
                Some(Motion::Slide(slide)),
                Some(Motion::Slide(flat)),
            );
        }
    }
    let buses = share(5);
    let cars = city.moving_left.saturating_sub(buses);
    for index in 0..buses + cars {
        vehicle(city, &mut rng, index < buses);
    }
}
