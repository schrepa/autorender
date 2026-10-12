//! The sea: an endless water surface at a still level that
//! waves raise, the same on the CPU and the GPU. The waves are Gerstner
//! waves, deep-water waves whose surface points circle so crests sharpen,
//! drawn from the spectrum of a sea the wind has fully raised and seeded.
//! Queries find the surface's height and normal over any point at any
//! time, as frames draw it: frames draw it from a grid around the camera
//! whose cells widen with distance, the waves shorter than a cell left to
//! the shading. Light passing through the sea fades toward its scattered
//! color over the distance to the ground behind it, foam gathers on
//! crests the waves compress and along shores, and sunlight glows through
//! crests seen against it.
//!
//! Frames draw the waves at the simulation time they present
//! ([`crate::PresentedTime`]), so entities [`Floating`] on the sea, which ride
//! it each fixed tick, rise and fall with the waves drawn around them.

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::environment::InvalidSetting;
use crate::math::{clamp, max, min, sin_cos};
use crate::{FixedTime, Parent, Transform};
use engine_settings::{ExtraRules, settings};

/// Most waves a sea holds ([`Ocean::waves`]).
pub const MAX_OCEAN_WAVES: usize = 16;

/// Gravity's pull on the waves (meters per second squared).
const GRAVITY: f32 = 9.81;

settings! {
    extra_rules;
    /// The sea. Requires version 9.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneOcean")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Ocean {
        /// Whether the sea draws.
        pub enabled: bool = false => any,
        /// The height of the still sea, world meters.
        pub level: f32 = 0.0 => range(-10_000.0, 10_000.0),
        /// The wind that raised it, meters per second along x and z: its speed,
        /// from 0.5 to 30, sets the waves' height and length and its direction
        /// heads them.
        pub wind: [f32; 2] = [6.0, 2.0] => range(-30.0, 30.0),
        /// How sharply crests stand: 0 round swells, 1 nearly breaking.
        pub choppiness: f32 = 0.7 => range(0.0, 1.0),
        /// The waves the sea holds.
        pub waves: u32 = 12 => range(1, MAX_OCEAN_WAVES),
        /// The seed of their headings and phases.
        pub seed: u32 = 1 => any,
        /// The color deep water scatters back, linear.
        pub color: [f32; 3] = [0.004, 0.032, 0.055] => range(0.0, 1.0),
        /// The meters of water through which half the light passes.
        pub clarity: f32 = 8.0 => range(0.1, 100.0),
        /// How much foam crests and shores gather.
        pub foam: f32 = 0.5 => range(0.0, 1.0),
        /// The meters from the camera the sea draws.
        pub reach: f32 = 20_000.0 => range(100.0, 100_000.0),
    }
}

impl ExtraRules for Ocean {
    /// The wind's speed lies from 0.5 to 30 meters per second.
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

    /// The wind rescaled to a speed from 0.5 to 30, the default without a
    /// heading.
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

/// One Gerstner wave: its wave vector (radians per meter along x and z,
/// pointing where it runs), its height above the still sea (meters), how
/// far its surface points move along it (meters), its angular frequency
/// (radians per second) and its phase at time zero at the origin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct OceanWave {
    pub vector: [f32; 2],
    pub amplitude: f32,
    pub reach: f32,
    pub frequency: f32,
    pub phase: f32,
}

impl OceanWave {
    /// Meters from crest to crest.
    pub fn length(&self) -> f32 {
        std::f32::consts::TAU / self.number()
    }

    /// Radians per meter.
    fn number(&self) -> f32 {
        let [x, z] = self.vector;
        max((x * x + z * z).sqrt(), 1.0e-6)
    }
}

/// A sea's waves, longest first, over its still level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OceanWaves {
    pub level: f32,
    pub count: usize,
    pub waves: [OceanWave; MAX_OCEAN_WAVES],
}

/// Where the waves move a point of the still sea: its displacement
/// (meters), the surface's normal there, and how the waves stretch the
/// surface across x and z (the derivatives of the moved point's x and z
/// along x, along z and across: `[dx/dx, dx/dz, dz/dz]`), whose
/// determinant falls from 1 toward 0 where crests sharpen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OceanPoint {
    pub displacement: Vec3,
    pub normal: Vec3,
    pub stretch: [f32; 3],
}

impl OceanPoint {
    /// How much the waves compress the surface here: 1 on still water,
    /// toward 0 on sharpening crests, where foam gathers.
    pub fn compression(&self) -> f32 {
        let [xx, xz, zz] = self.stretch;
        xx * zz - xz * xz
    }
}

/// The sea's surface over a world point at a time: its height (world
/// meters), normal, compression ([`OceanPoint::compression`]) and the
/// point of the still sea that moved there (world x and z).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct OceanSurface {
    pub height: f32,
    pub normal: [f32; 3],
    pub compression: f32,
    pub still: [f64; 2],
}

impl Ocean {
    /// What a world without these settings reads as: no sea.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// The sea's waves: [`Ocean::waves`] of them from the Pierson-Moskowitz
    /// spectrum of a sea the wind has fully raised, from three quarters of
    /// its peak frequency to three and a half times it, each as high as
    /// its share of the spectrum. They head within a spread around the
    /// wind that widens for shorter waves, alternately to either side,
    /// with seeded phases, and every wave's points circle by the same share
    /// of its height, sized so the surface's crests sharpen by
    /// [`Ocean::choppiness`] without folding over.
    pub fn spectrum(&self) -> OceanWaves {
        let settings = self.sanitized();
        let [x, z] = settings.wind;
        let speed = (x * x + z * z).sqrt();
        let heading = [x / speed, z / speed];
        let peak = 0.877 * GRAVITY / speed;
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
            // The fully raised sea's spectral density at this frequency.
            let density = 0.0081 * GRAVITY * GRAVITY / frequency.powi(5)
                * (-1.25 * (peak / frequency).powi(4)).exp();
            let amplitude = (2.0 * density * width).sqrt();
            let number = frequency * frequency / GRAVITY;
            let share = if count > 1 {
                index as f32 / (count - 1) as f32
            } else {
                0.0
            };
            let side = if index % 2 == 0 { 1.0 } else { -1.0 };
            let turn = side * (0.35 + 0.55 * share) * (0.3 + 0.7 * hash(index as u32, 1));
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
        // Points circle by the same share of each wave's height: the
        // surface's stretch falls to 1 - choppiness on the sharpest crests.
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
    /// Rides the sea: each fixed tick the entity rises and falls with the sea's
    /// surface under it and leans about its own heading. Unparented entities
    /// only; nothing moves without an enabled sea. Requires version 9.
    #[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneFloating")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Floating {
        /// Meters of it below the surface.
        pub draft: f32 = 0.0 => range(-100.0, 100.0),
        /// How far it leans: 0 upright, 1 with the surface's normal.
        pub tilt: f32 = 1.0 => range(0.0, 1.0),
    }
}

/// The sea's systems.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OceanSystems {
    /// [`Floating`] entities ride the sea at the tick's time, in
    /// [`crate::FixedSimulation`]; gameplay moving them across it runs
    /// before.
    Float,
}

/// Floating entities ride the sea at the fixed tick's time.
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

/// `angle` radians wrapped into [-pi, pi].
fn wrapped(angle: f64) -> f32 {
    let turns = (angle / std::f64::consts::TAU).round();
    (angle - turns * std::f64::consts::TAU) as f32
}

impl OceanWaves {
    /// The waves held.
    pub fn waves(&self) -> &[OceanWave] {
        &self.waves[..self.count]
    }

    /// The most the surface rises above or falls below the still sea.
    pub fn amplitude(&self) -> f32 {
        self.waves().iter().map(|wave| wave.amplitude).sum()
    }

    /// Each wave's phase at world point (`x`, `z`), `seconds` in, wrapped
    /// into [-pi, pi]: frames take them at the eye, and their vertices
    /// add the wave vector's product with their offset from it.
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

    /// Where the waves move the still sea's point (`x`, `z`) (world
    /// meters), `seconds` in.
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
        // The moved surface's tangents along x and z; their cross product
        // faces up.
        let along_x = Vec3::new(xx, rise_x, xz);
        let along_z = Vec3::new(xz, rise_z, zz);
        OceanPoint {
            displacement,
            normal: along_z.cross(along_x).normalize_or(Vec3::Y),
            stretch: [xx, xz, zz],
        }
    }

    /// The surface over world point (`x`, `z`), `seconds` in: the still
    /// sea's point the waves move there (found by Newton's method on the
    /// waves' stretch), and the surface at it.
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

/// The sea and the entities riding it: [`crate::Feature::Ocean`].
#[derive(Default)]
pub struct OceanPlugin;

impl bevy_app::Plugin for OceanPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        crate::install(app, crate::Feature::Ocean);
        app.add_systems(crate::FixedSimulation, float.in_set(OceanSystems::Float))
            .configure_sets(
                crate::FixedSimulation,
                OceanSystems::Float.in_set(crate::FixedSimulationSystems::Ride),
            );
    }
}
