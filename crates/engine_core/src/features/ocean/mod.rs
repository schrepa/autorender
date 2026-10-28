use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::environment::InvalidSetting;
use crate::math::{clamp, max, min, sin_cos};
use crate::terrain_data::hydraulics::GRAVITY;
use crate::{FixedTime, Parent, Transform};
use engine_settings::{ExtraRules, settings};

pub mod wind_waves;

pub const MAX_OCEAN_WAVES: usize = 16;

settings! {
    extra_rules;
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneOcean")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Ocean {
        pub enabled: bool = false => any,
        pub level: f32 = 0.0 => range(-10_000.0, 10_000.0),
        pub wind: [f32; 2] = [6.0, 2.0] => range(-30.0, 30.0),
        pub choppiness: f32 = 0.7 => range(0.0, 1.0),
        pub waves: u32 = 12 => range(1, MAX_OCEAN_WAVES),
        pub seed: u32 = 1 => any,
        pub color: [f32; 3] = [0.004, 0.032, 0.055] => range(0.0, 1.0),
        pub clarity: f32 = 8.0 => range(0.1, 100.0),
        pub reach: f32 = 20_000.0 => range(100.0, 100_000.0),
    }
}

impl ExtraRules for Ocean {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        let [x, z] = self.wind;
        let speed = (x * x + z * z).sqrt();
        if (0.5..=30.0).contains(&speed) {
            Ok(())
        } else {
            Err(InvalidSetting::new(
                "wind",
                "must have a speed in [0.5, 30]",
            ))
        }
    }

    fn sanitize_extra(self) -> Self {
        let [x, z] = self.wind;
        let speed = (x * x + z * z).sqrt();
        let wind = if speed > 1.0e-3 {
            let scale = clamp(speed, 0.5, 30.0) / speed;
            [x * scale, z * scale]
        } else {
            Self::default().wind
        };
        Self { wind, ..self }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OceanWave {
    pub vector: [f32; 2],
    pub amplitude: f32,
    pub reach: f32,
    pub frequency: f32,
    pub phase: f32,
}

impl OceanWave {
    pub fn length(&self) -> f32 {
        std::f32::consts::TAU / self.number()
    }

    fn number(&self) -> f32 {
        let [x, z] = self.vector;
        max((x * x + z * z).sqrt(), 1.0e-6)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OceanWaves {
    pub level: f32,
    pub count: usize,
    pub waves: [OceanWave; MAX_OCEAN_WAVES],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OceanPoint {
    pub displacement: Vec3,
    pub normal: Vec3,
    pub stretch: [f32; 3],
}

impl OceanPoint {
    pub fn compression(&self) -> f32 {
        let [xx, xz, zz] = self.stretch;
        xx * zz - xz * xz
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct OceanSurface {
    pub height: f32,
    pub normal: [f32; 3],
    pub compression: f32,
    pub still: [f64; 2],
}

impl Ocean {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    pub fn spectrum(&self) -> OceanWaves {
        let settings = self.sanitized();
        let [x, z] = settings.wind;
        let speed = (x * x + z * z).sqrt();
        let heading = [x / speed, z / speed];
        let sea = wind_waves::WindSea::new(speed, f32::INFINITY, f32::INFINITY);
        let peak = (GRAVITY * sea.peak).sqrt();
        let count = settings.waves as usize;
        let (low, high) = (0.75 * peak, 3.5 * peak);
        let ratio = if count > 1 {
            (high / low).powf(1.0 / (count - 1) as f32)
        } else {
            1.0
        };
        let hash = |index: u32, salt: u32| {
            let mut h = settings
                .seed
                .wrapping_mul(0x9e37_79b9)
                .wrapping_add(index.wrapping_mul(0x85eb_ca6b))
                ^ salt.wrapping_mul(0xc2b2_ae35);
            h ^= h >> 16;
            h = h.wrapping_mul(0x7feb_352d);
            h ^= h >> 15;
            h = h.wrapping_mul(0x846c_a68b);
            h ^= h >> 16;
            (h & 0x00ff_ffff) as f32 / 16_777_215.0
        };
        let mut waves = [OceanWave::default(); MAX_OCEAN_WAVES];
        let mut slopes = 0.0;
        for (index, wave) in waves.iter_mut().enumerate().take(count) {
            let (frequency, width) = if count > 1 {
                let frequency = low * ratio.powi(index as i32);
                (frequency, frequency * (ratio.sqrt() - 1.0 / ratio.sqrt()))
            } else {
                (peak, 0.5 * peak)
            };
            let number = frequency * frequency / GRAVITY;
            let variance = sea.slopes(number, 2.0 * width / frequency) / (number * number);
            let amplitude = (2.0 * variance).sqrt();
            let side = if index % 2 == 0 { 1.0 } else { -1.0 };
            let turn = engine_tables::waves::spread_turn(
                f64::from(frequency / peak),
                side * f64::from(hash(index as u32, 1)),
            ) as f32;
            let (sin, cos) = sin_cos(turn);
            let direction = [
                heading[0] * cos - heading[1] * sin,
                heading[0] * sin + heading[1] * cos,
            ];
            *wave = OceanWave {
                vector: [direction[0] * number, direction[1] * number],
                amplitude,
                reach: 0.0,
                frequency,
                phase: hash(index as u32, 2) * std::f32::consts::TAU,
            };
            slopes += number * amplitude;
        }
        let circling = if slopes > 1.0e-6 {
            min(settings.choppiness / slopes, 3.0)
        } else {
            0.0
        };
        for wave in &mut waves[..count] {
            wave.reach = circling * wave.amplitude;
        }
        OceanWaves {
            level: settings.level,
            count,
            waves,
        }
    }
}

settings! {
    #[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneFloating")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Floating {
        pub draft: f32 = 0.0 => range(-100.0, 100.0),
        pub tilt: f32 = 1.0 => range(0.0, 1.0),
    }
}

#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OceanSystems {
    Float,
}

pub(crate) fn float(
    ocean: Option<Res<Ocean>>,
    time: Res<FixedTime>,
    mut floating: Query<(&mut Transform, &Floating), Without<Parent>>,
) {
    let Some(ocean) = ocean.filter(|ocean| ocean.enabled) else {
        return;
    };
    if floating.is_empty() {
        return;
    }
    let waves = ocean.spectrum();
    for (mut transform, floating) in &mut floating {
        let at = transform.translation;
        let surface = waves.surface(f64::from(at.x), f64::from(at.z), time.elapsed_seconds);
        let normal = Vec3::from_array(surface.normal);
        let up = Vec3::Y
            .lerp(normal, clamp(floating.tilt, 0.0, 1.0))
            .normalize_or(Vec3::Y);
        let forward = transform.rotation * Vec3::Z;
        let heading = Quat::from_rotation_y(forward.x.atan2(forward.z));
        transform.translation.y = surface.height - floating.draft;
        transform.rotation = Quat::from_rotation_arc(Vec3::Y, up) * heading;
    }
}

fn wrapped(angle: f64) -> f32 {
    let turns = (angle / std::f64::consts::TAU).round();
    (angle - turns * std::f64::consts::TAU) as f32
}

impl OceanWaves {
    pub fn waves(&self) -> &[OceanWave] {
        &self.waves[..self.count]
    }

    pub fn amplitude(&self) -> f32 {
        self.waves().iter().map(|wave| wave.amplitude).sum()
    }

    pub fn phases(&self, x: f64, z: f64, seconds: f64) -> [f32; MAX_OCEAN_WAVES] {
        let mut phases = [0.0; MAX_OCEAN_WAVES];
        for (phase, wave) in phases.iter_mut().zip(self.waves()) {
            *phase = wrapped(
                f64::from(wave.vector[0]) * x + f64::from(wave.vector[1]) * z
                    - f64::from(wave.frequency) * seconds
                    + f64::from(wave.phase),
            );
        }
        phases
    }

    pub fn moved(&self, x: f64, z: f64, seconds: f64) -> OceanPoint {
        let phases = self.phases(x, z, seconds);
        let mut displacement = Vec3::ZERO;
        let (mut rise_x, mut rise_z) = (0.0, 0.0);
        let (mut xx, mut xz, mut zz) = (1.0, 0.0, 1.0);
        for (wave, &phase) in self.waves().iter().zip(&phases) {
            let (sin, cos) = sin_cos(phase);
            let [kx, kz] = wave.vector;
            let number = wave.number();
            let (dx, dz) = (kx / number, kz / number);
            displacement += Vec3::new(
                wave.reach * dx * cos,
                wave.amplitude * sin,
                wave.reach * dz * cos,
            );
            rise_x += wave.amplitude * kx * cos;
            rise_z += wave.amplitude * kz * cos;
            xx -= wave.reach * dx * kx * sin;
            xz -= wave.reach * dx * kz * sin;
            zz -= wave.reach * dz * kz * sin;
        }
        let along_x = Vec3::new(xx, rise_x, xz);
        let along_z = Vec3::new(xz, rise_z, zz);
        OceanPoint {
            displacement,
            normal: along_z.cross(along_x).normalize_or(Vec3::Y),
            stretch: [xx, xz, zz],
        }
    }

    pub fn surface(&self, x: f64, z: f64, seconds: f64) -> OceanSurface {
        let mut still = [x, z];
        let mut point = self.moved(x, z, seconds);
        for _ in 0..8 {
            let miss = [
                still[0] + f64::from(point.displacement.x) - x,
                still[1] + f64::from(point.displacement.z) - z,
            ];
            if miss[0].abs() + miss[1].abs() < 1.0e-5 {
                break;
            }
            let [xx, xz, zz] = point.stretch.map(f64::from);
            let determinant = xx * zz - xz * xz;
            let step = if determinant > 0.05 {
                [
                    (zz * miss[0] - xz * miss[1]) / determinant,
                    (xx * miss[1] - xz * miss[0]) / determinant,
                ]
            } else {
                miss
            };
            still = [still[0] - step[0], still[1] - step[1]];
            point = self.moved(still[0], still[1], seconds);
        }
        OceanSurface {
            height: self.level + point.displacement.y,
            normal: point.normal.to_array(),
            compression: point.compression(),
            still,
        }
    }
}

const VELOCITY_STEP: f64 = 0.05;
const OPEN_SEA: f32 = 1000.0;

pub(crate) fn sea_water(world: &World, point: Vec3) -> Option<crate::WaterSample> {
    let ocean = world
        .get_resource::<Ocean>()
        .filter(|ocean| ocean.enabled)?;
    let seconds = world
        .get_resource::<FixedTime>()
        .map_or(0.0, |time| time.elapsed_seconds);
    let waves = ocean.spectrum();
    let surface = waves.surface(f64::from(point.x), f64::from(point.z), seconds);
    let [x, z] = surface.still;
    let ahead = waves.moved(x, z, seconds + VELOCITY_STEP).displacement;
    let behind = waves.moved(x, z, seconds - VELOCITY_STEP).displacement;
    let velocity = (ahead - behind) / (2.0 * VELOCITY_STEP as f32);
    let ground = crate::Ground::sample_in(world, point.x, point.z).map(|ground| ground.height);
    Some(crate::WaterSample {
        kind: crate::WaterKind::Sea,
        surface: surface.height,
        depth: ground.map_or(OPEN_SEA, |ground| max(surface.height - ground, 0.0)),
        velocity: velocity.to_array(),
        foam: clamp((0.8 - surface.compression) * 8.0, 0.0, 1.0),
    })
}

#[derive(Default)]
pub struct OceanPlugin;

impl bevy_app::Plugin for OceanPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        crate::install(app, crate::Feature::Ocean);
        app.world_mut()
            .resource_mut::<crate::WaterSources>()
            .register("sea", sea_water);
        app.add_systems(crate::FixedSimulation, float.in_set(OceanSystems::Float))
            .configure_sets(
                crate::FixedSimulation,
                OceanSystems::Float.in_set(crate::FixedSimulationSystems::Ride),
            );
    }
}
