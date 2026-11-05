use engine_core::{Error, ErrorCode, ExtraRules, InvalidSetting, Reach, Setting, fields};
use engine_places::particles::{MAX_SPREAD_DEGREES, ParticleBlend, ParticleEmitter, ParticleShape};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::emitters::{emitter_extra, emitter_rule};

impl Setting for SceneParticleBlend {}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SceneParticleBlend {
    #[default]
    Alpha,
    Additive,
    Mist,
}

impl Setting for SceneParticleShape {}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SceneParticleShape {
    Box,
    #[default]
    Ball,
}

fn default_emitter() -> SceneEmitter {
    SceneEmitter::from(&ParticleEmitter::default())
}

fields! {
    extra_rules;
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct SceneEmitter {
        pub rate: f32 = default_emitter().rate
            => same(emitter_rule("rate")),
        pub offset: [f32; 3] = default_emitter().offset
            => same(emitter_rule("offset")),
        pub emitting: bool = default_emitter().emitting => any,
        pub lifetime: f32 = default_emitter().lifetime
            => same(emitter_rule("lifetime")),
        pub lifetime_spread: f32 = default_emitter().lifetime_spread
            => same(emitter_rule("lifetime_spread")),
        pub speed: f32 = default_emitter().speed
            => same(emitter_rule("speed")),
        pub speed_spread: f32 = default_emitter().speed_spread
            => same(emitter_rule("speed_spread")),
        pub direction: [f32; 3] = default_emitter().direction
            => same(emitter_rule("direction")),
        pub spread_degrees: f32 = default_emitter().spread_degrees
            => range(0.0, MAX_SPREAD_DEGREES),
        pub acceleration: [f32; 3] = default_emitter().acceleration
            => same(emitter_rule("acceleration")),
        pub shape: SceneParticleShape = default_emitter().shape => any,
        pub extent: [f32; 3] = default_emitter().extent
            => same(emitter_rule("extent")),
        pub size: [f32; 2] = default_emitter().size
            => same(emitter_rule("size")),
        pub size_spread: f32 = default_emitter().size_spread
            => same(emitter_rule("size_spread")),
        pub color: [[f32; 4]; 2] = default_emitter().color
            => same(emitter_rule("color")),
        pub fade_in: f32 = default_emitter().fade_in
            => same(emitter_rule("fade_in")),
        pub brightness: f32 = default_emitter().brightness
            => same(emitter_rule("brightness")),
        pub blend: SceneParticleBlend = default_emitter().blend => any,
        pub stretch: f32 = default_emitter().stretch
            => same(emitter_rule("stretch")),
        pub seed: u32 = default_emitter().seed => any,
        pub formula: u32 = default_emitter().formula => any,
        pub params: [f32; 8] = default_emitter().params
            => same(emitter_rule("params")),
    }
}

impl ExtraRules for SceneEmitter {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        emitter_extra(Vec3::from_array(self.direction), self.rate, self.lifetime)
    }
}

impl From<&ParticleEmitter> for SceneEmitter {
    fn from(emitter: &ParticleEmitter) -> Self {
        let look = &emitter.look;
        Self {
            rate: emitter.rate,
            offset: emitter.offset.to_array(),
            emitting: emitter.emitting,
            lifetime: look.lifetime,
            lifetime_spread: look.lifetime_spread,
            speed: look.speed,
            speed_spread: look.speed_spread,
            direction: look.direction.to_array(),
            spread_degrees: look.authored_degrees.unwrap_or(look.spread.to_degrees()),
            acceleration: look.acceleration.to_array(),
            shape: match look.shape {
                ParticleShape::Box => SceneParticleShape::Box,
                ParticleShape::Ball => SceneParticleShape::Ball,
            },
            extent: look.extent.to_array(),
            size: look.size,
            size_spread: look.size_spread,
            color: look.color,
            fade_in: look.fade_in,
            brightness: look.brightness,
            blend: match look.blend {
                ParticleBlend::Alpha => SceneParticleBlend::Alpha,
                ParticleBlend::Additive => SceneParticleBlend::Additive,
                ParticleBlend::Mist => SceneParticleBlend::Mist,
            },
            stretch: look.stretch,
            seed: look.seed,
            formula: look.formula,
            params: look.params,
        }
    }
}

pub(crate) fn check_emitter(emitter: &ParticleEmitter, path: &str) -> Result<(), Error> {
    SceneEmitter::from(emitter)
        .validate()
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))
}

pub(crate) fn emitter_reach(
    emitter: &ParticleEmitter,
    reach: &Reach<'_>,
    extend: &mut dyn FnMut(f32),
) {
    let look = &emitter.look;
    let shape = match look.shape {
        ParticleShape::Box => look.extent.length(),
        ParticleShape::Ball => look.extent.max_element(),
    };
    extend(
        reach.from
            + emitter.offset.length()
            + shape
            + look.speed * look.lifetime
            + 0.5 * look.acceleration.length() * look.lifetime * look.lifetime,
    );
}

pub(crate) fn emitter_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(SceneEmitter)).unwrap_or(Value::Null)
}

pub(crate) fn emitter_example() -> Value {
    json!({})
}
