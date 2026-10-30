use engine_core::{
    Error, LightRegion, LocalLight, MAX_SPOT_DEGREES, MAX_SPREAD_DEGREES, MIN_SPOT_DEGREES,
    ParticleBlend, ParticleEmitter, Spot, bevy_ecs::world::EntityWorldMut, glam::Vec3,
};
use engine_settings::{ExtraRules, InvalidSetting, Rule, Setting, settings};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::components::{
    ComponentContext, ComponentKind, Placement, Prepared, PreparedComponent, Reach, admitted,
    no_files, object_example, parsed, ruled, valid_world,
};
use crate::instance::Lookups;

settings! {
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct SceneLight {
        pub color: [f32; 3] = LocalLight::default().color.to_array()
            => same(LocalLight::rule_of("color")),
        pub intensity: f32 = LocalLight::default().intensity
            => same(LocalLight::rule_of("intensity")),
        pub range: f32 = LocalLight::default().range => same(LocalLight::rule_of("range")),
        pub offset: [f32; 3] = LocalLight::default().offset.to_array()
            => same(LocalLight::rule_of("offset")),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub spot: Option<SceneSpot> = None => any,
        #[serde(skip_serializing_if = "is_zero")]
        pub radius: f32 = LocalLight::default().radius => same(LocalLight::rule_of("radius")),
        #[serde(skip_serializing_if = "is_zero")]
        pub length: f32 = LocalLight::default().length => same(LocalLight::rule_of("length")),
        #[serde(skip_serializing_if = "is_both")]
        pub region: SceneLightRegion = SceneLightRegion::Both => any,
    }
}

fn is_zero(value: &f32) -> bool {
    *value == 0.0
}

fn is_both(region: &SceneLightRegion) -> bool {
    *region == SceneLightRegion::Both
}

impl Setting for SceneLightRegion {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneLightRegion {
    #[default]
    Both,
    Interior,
    Exterior,
}

impl From<SceneLightRegion> for LightRegion {
    fn from(region: SceneLightRegion) -> Self {
        match region {
            SceneLightRegion::Both => Self::Both,
            SceneLightRegion::Interior => Self::Interior,
            SceneLightRegion::Exterior => Self::Exterior,
        }
    }
}

impl From<LightRegion> for SceneLightRegion {
    fn from(region: LightRegion) -> Self {
        match region {
            LightRegion::Both => Self::Both,
            LightRegion::Interior => Self::Interior,
            LightRegion::Exterior => Self::Exterior,
        }
    }
}

settings! {
    extra_rules;
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct SceneSpot {
        pub direction: [f32; 3] = Spot::default().direction.to_array()
            => same(Spot::rule_of("direction")),
        pub inner_angle_degrees: f32 = Spot::default().inner_angle.to_degrees()
            => range(0.0, MAX_SPOT_DEGREES),
        pub outer_angle_degrees: f32 = Spot::default().outer_angle.to_degrees()
            => range(MIN_SPOT_DEGREES, MAX_SPOT_DEGREES),
    }
}

impl ExtraRules for SceneSpot {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        self.spot().check_extra()
    }
}

impl SceneSpot {
    fn spot(&self) -> Spot {
        Spot {
            direction: Vec3::from_array(self.direction),
            inner_angle: self.inner_angle_degrees.to_radians(),
            outer_angle: self.outer_angle_degrees.to_radians(),
        }
    }
}

impl SceneLight {
    pub(crate) fn runtime(&self) -> LocalLight {
        let spot = self.spot.map(|spot| spot.spot());
        LocalLight {
            color: Vec3::from_array(self.color),
            intensity: self.intensity,
            range: self.range,
            offset: Vec3::from_array(self.offset),
            spot,
            radius: self.radius,
            length: self.length,
            region: self.region.into(),
        }
    }
}

impl From<&LocalLight> for SceneLight {
    fn from(light: &LocalLight) -> Self {
        Self {
            color: light.color.to_array(),
            intensity: light.intensity,
            range: light.range,
            offset: light.offset.to_array(),
            spot: light.spot.map(|spot| SceneSpot {
                direction: spot.direction.to_array(),
                inner_angle_degrees: spot.inner_angle.to_degrees(),
                outer_angle_degrees: spot.outer_angle.to_degrees(),
            }),
            radius: light.radius,
            length: light.length,
            region: light.region.into(),
        }
    }
}

impl Setting for SceneParticleBlend {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneParticleBlend {
    #[default]
    Alpha,
    Additive,
    Mist,
}

fn default_emitter() -> SceneEmitter {
    SceneEmitter::from(&ParticleEmitter::default())
}

settings! {
    extra_rules;
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct SceneEmitter {
        pub rate: f32 = default_emitter().rate
            => same(ParticleEmitter::rule_of("rate")),
        pub lifetime: f32 = default_emitter().lifetime
            => same(ParticleEmitter::rule_of("lifetime")),
        pub speed: f32 = default_emitter().speed
            => same(ParticleEmitter::rule_of("speed")),
        pub direction: [f32; 3] = default_emitter().direction
            => same(ParticleEmitter::rule_of("direction")),
        pub spread_degrees: f32 = default_emitter().spread_degrees
            => range(0.0, MAX_SPREAD_DEGREES),
        pub acceleration: [f32; 3] = default_emitter().acceleration
            => same(ParticleEmitter::rule_of("acceleration")),
        pub size: [f32; 2] = default_emitter().size
            => same(ParticleEmitter::rule_of("size")),
        pub color: [[f32; 4]; 2] = default_emitter().color
            => same(ParticleEmitter::rule_of("color")),
        pub emission: f32 = default_emitter().emission
            => same(ParticleEmitter::rule_of("emission")),
        pub blend: SceneParticleBlend = default_emitter().blend => any,
        pub stretch: f32 = default_emitter().stretch
            => same(ParticleEmitter::rule_of("stretch")),
        pub offset: [f32; 3] = default_emitter().offset
            => same(ParticleEmitter::rule_of("offset")),
        pub radius: f32 = default_emitter().radius
            => same(ParticleEmitter::rule_of("radius")),
        pub seed: u32 = default_emitter().seed => any,
    }
}

impl ExtraRules for SceneEmitter {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        self.runtime().check_extra()
    }
}

impl From<&ParticleEmitter> for SceneEmitter {
    fn from(emitter: &ParticleEmitter) -> Self {
        Self {
            rate: emitter.rate,
            lifetime: emitter.lifetime,
            speed: emitter.speed,
            direction: emitter.direction.to_array(),
            spread_degrees: emitter.spread.to_degrees(),
            acceleration: emitter.acceleration.to_array(),
            size: emitter.size,
            color: emitter.color,
            emission: emitter.emission,
            blend: match emitter.blend {
                ParticleBlend::Alpha => SceneParticleBlend::Alpha,
                ParticleBlend::Additive => SceneParticleBlend::Additive,
                ParticleBlend::Mist => SceneParticleBlend::Mist,
            },
            stretch: emitter.stretch,
            offset: emitter.offset.to_array(),
            radius: emitter.radius,
            seed: emitter.seed,
        }
    }
}

impl SceneEmitter {
    pub(crate) fn runtime(&self) -> ParticleEmitter {
        let direction = Vec3::from_array(self.direction);
        ParticleEmitter {
            rate: self.rate,
            lifetime: self.lifetime,
            speed: self.speed,
            direction,
            spread: self.spread_degrees.to_radians(),
            acceleration: Vec3::from_array(self.acceleration),
            size: self.size,
            color: self.color,
            emission: self.emission,
            blend: match self.blend {
                SceneParticleBlend::Alpha => ParticleBlend::Alpha,
                SceneParticleBlend::Additive => ParticleBlend::Additive,
                SceneParticleBlend::Mist => ParticleBlend::Mist,
            },
            stretch: self.stretch,
            offset: Vec3::from_array(self.offset),
            radius: self.radius,
            seed: self.seed,
        }
    }
}

pub(crate) static LIGHT: ComponentKind = ComponentKind {
    name: "light",
    shows: "LocalLight",
    placement: Placement::Anywhere,
    schema: crate::component_schema!(SceneLight),
    example: object_example,
    validate: validate_light,
    validate_world: valid_world,
    prepare: prepare_light,
    files: no_files,
    admit: admitted,
};

fn validate_light(value: &Value, path: &str) -> Result<(), Error> {
    ruled::<SceneLight>(value, path, LIGHT.name, Rule::Any).map(drop)
}

fn prepare_light(value: &Value, context: &ComponentContext<'_>) -> Prepared {
    Ok(Box::new(parsed::<SceneLight>(value, context.path)?.runtime()))
}

impl PreparedComponent for LocalLight {
    fn spawn(&self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*self);
    }

    fn reach(&self, reach: &Reach<'_>, extend: &mut dyn FnMut(f32)) {
        extend(reach.from + self.range);
    }
}

pub(crate) static PARTICLES: ComponentKind = ComponentKind {
    name: "particles",
    shows: "ParticleEmitter",
    placement: Placement::Anywhere,
    schema: crate::component_schema!(SceneEmitter),
    example: object_example,
    validate: validate_particles,
    validate_world: valid_world,
    prepare: prepare_particles,
    files: no_files,
    admit: admitted,
};

fn validate_particles(value: &Value, path: &str) -> Result<(), Error> {
    ruled::<SceneEmitter>(value, path, PARTICLES.name, Rule::Any).map(drop)
}

fn prepare_particles(value: &Value, context: &ComponentContext<'_>) -> Prepared {
    Ok(Box::new(parsed::<SceneEmitter>(value, context.path)?.runtime()))
}

impl PreparedComponent for ParticleEmitter {
    fn spawn(&self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*self);
    }

    fn reach(&self, reach: &Reach<'_>, extend: &mut dyn FnMut(f32)) {
        extend(
            reach.from
                + self.offset.length()
                + self.radius
                + self.speed * self.lifetime
                + 0.5 * self.acceleration.length() * self.lifetime * self.lifetime,
        );
    }
}
