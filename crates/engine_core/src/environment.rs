//! The look as data (decision 022): a world's atmosphere and the settings of
//! every rendering effect. Each settings type is a resource, a section of a
//! scene's `environment` in the same JSON shape (the atmosphere's section is
//! a preset with overrides), and checks itself with `validate`; frames carry
//! them all sanitized as [`crate::RenderSnapshot::environment`].

use std::borrow::Cow;

use bevy_ecs::prelude::{Resource, World};

use crate::{
    AmbientOcclusion, Atmosphere, Bloom, Decals, LevelOfDetail, LocalLights, Particles,
    ReflectionProbes, Reflections, Refraction, Shadows, SunShafts, VirtualGeometry,
};

/// A settings field outside its valid values: its name and the rule it
/// breaks, such as `must be in [0, 1]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidSetting {
    pub field: &'static str,
    pub rule: Cow<'static, str>,
}

impl InvalidSetting {
    pub(crate) fn new(field: &'static str, rule: impl Into<Cow<'static, str>>) -> Self {
        Self {
            field,
            rule: rule.into(),
        }
    }
}

impl std::fmt::Display for InvalidSetting {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} {}", self.field, self.rule)
    }
}

impl std::error::Error for InvalidSetting {}

/// Every look setting of a world: its atmosphere and each effect's settings.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Environment {
    pub atmosphere: Atmosphere,
    pub shadows: Shadows,
    pub ambient_occlusion: AmbientOcclusion,
    pub bloom: Bloom,
    pub local_lights: LocalLights,
    pub sun_shafts: SunShafts,
    pub reflections: Reflections,
    pub particles: Particles,
    pub refraction: Refraction,
    pub decals: Decals,
    pub reflection_probes: ReflectionProbes,
    pub virtual_geometry: VirtualGeometry,
    pub level_of_detail: LevelOfDetail,
}

impl Environment {
    /// `world`'s settings as its resources hold them. A missing atmosphere
    /// reads as the default one, and a missing effect resource as that
    /// effect turned off.
    pub fn current(world: &World) -> Self {
        fn read<T: Resource + Copy>(world: &World, off: T) -> T {
            world.get_resource::<T>().copied().unwrap_or(off)
        }
        Self {
            atmosphere: read(world, Atmosphere::default()),
            shadows: read(
                world,
                Shadows {
                    enabled: false,
                    ..Shadows::default()
                },
            ),
            ambient_occlusion: read(
                world,
                AmbientOcclusion {
                    intensity: 0.0,
                    ..AmbientOcclusion::default()
                },
            ),
            bloom: read(
                world,
                Bloom {
                    intensity: 0.0,
                    ..Bloom::default()
                },
            ),
            local_lights: read(
                world,
                LocalLights {
                    enabled: false,
                    ..LocalLights::default()
                },
            ),
            sun_shafts: read(
                world,
                SunShafts {
                    intensity: 0.0,
                    ..SunShafts::default()
                },
            ),
            reflections: read(
                world,
                Reflections {
                    intensity: 0.0,
                    ..Reflections::default()
                },
            ),
            particles: read(
                world,
                Particles {
                    enabled: false,
                    ..Particles::default()
                },
            ),
            refraction: read(
                world,
                Refraction {
                    strength: 0.0,
                    ..Refraction::default()
                },
            ),
            decals: read(
                world,
                Decals {
                    enabled: false,
                    ..Decals::default()
                },
            ),
            reflection_probes: read(
                world,
                ReflectionProbes {
                    enabled: false,
                    ..ReflectionProbes::default()
                },
            ),
            virtual_geometry: read(
                world,
                VirtualGeometry {
                    enabled: false,
                    ..VirtualGeometry::default()
                },
            ),
            level_of_detail: read(
                world,
                LevelOfDetail {
                    enabled: false,
                    ..LevelOfDetail::default()
                },
            ),
        }
    }

    /// Every section sanitized: usable for rendering whatever it held.
    pub fn sanitized(self) -> Self {
        Self {
            atmosphere: self.atmosphere.sanitized(),
            shadows: self.shadows.sanitized(),
            ambient_occlusion: self.ambient_occlusion.sanitized(),
            bloom: self.bloom.sanitized(),
            local_lights: self.local_lights.sanitized(),
            sun_shafts: self.sun_shafts.sanitized(),
            reflections: self.reflections.sanitized(),
            particles: self.particles.sanitized(),
            refraction: self.refraction.sanitized(),
            decals: self.decals.sanitized(),
            reflection_probes: self.reflection_probes.sanitized(),
            virtual_geometry: self.virtual_geometry.sanitized(),
            level_of_detail: self.level_of_detail.sanitized(),
        }
    }
}

/// The rule of a value that must lie in `range`, as `must be in [a, b]`.
pub(crate) fn within(range: &std::ops::RangeInclusive<f32>) -> String {
    format!("must be in [{}, {}]", range.start(), range.end())
}

/// Checks that `value` lies in `range` (NaN never does).
pub(crate) fn check_range(
    field: &'static str,
    value: f32,
    range: std::ops::RangeInclusive<f32>,
) -> Result<(), InvalidSetting> {
    if range.contains(&value) {
        Ok(())
    } else {
        Err(InvalidSetting::new(field, within(&range)))
    }
}

/// Checks that an optional budget is at least 1.
pub(crate) fn check_budget(field: &'static str, budget: Option<u32>) -> Result<(), InvalidSetting> {
    if budget == Some(0) {
        Err(InvalidSetting::new(field, "must be at least 1"))
    } else {
        Ok(())
    }
}
