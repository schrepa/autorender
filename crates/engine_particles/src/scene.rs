use engine_core::{Error, ErrorCode, ExtraRules, InvalidSetting, Reach, Setting, fields};
use engine_places::particles::{MAX_SPREAD_DEGREES, ParticleBlend, ParticleEmitter};
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
        pub lifetime: f32 = default_emitter().lifetime
            => same(emitter_rule("lifetime")),
        pub speed: f32 = default_emitter().speed
            => same(emitter_rule("speed")),
        pub direction: [f32; 3] = default_emitter().direction
            => same(emitter_rule("direction")),
        pub spread_degrees: f32 = default_emitter().spread_degrees
            => range(0.0, MAX_SPREAD_DEGREES),
        pub acceleration: [f32; 3] = default_emitter().acceleration
            => same(emitter_rule("acceleration")),
        pub size: [f32; 2] = default_emitter().size
            => same(emitter_rule("size")),
        pub color: [[f32; 4]; 2] = default_emitter().color
            => same(emitter_rule("color")),
        pub emission: f32 = default_emitter().emission
            => same(emitter_rule("emission")),
        pub blend: SceneParticleBlend = default_emitter().blend => any,
        pub stretch: f32 = default_emitter().stretch
            => same(emitter_rule("stretch")),
        pub offset: [f32; 3] = default_emitter().offset
            => same(emitter_rule("offset")),
        pub radius: f32 = default_emitter().radius
            => same(emitter_rule("radius")),
        pub seed: u32 = default_emitter().seed => any,
    }
}

impl ExtraRules for SceneEmitter {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        emitter_extra(Vec3::from_array(self.direction), self.rate, self.lifetime)
    }
}

impl From<&ParticleEmitter> for SceneEmitter {
    fn from(emitter: &ParticleEmitter) -> Self {
        Self {
            rate: emitter.rate,
            lifetime: emitter.lifetime,
            speed: emitter.speed,
            direction: emitter.direction.to_array(),
            spread_degrees: emitter
                .authored_degrees
                .unwrap_or(emitter.spread.to_degrees()),
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
    extend(
        reach.from
            + emitter.offset.length()
            + emitter.radius
            + emitter.speed * emitter.lifetime
            + 0.5 * emitter.acceleration.length() * emitter.lifetime * emitter.lifetime,
    );
}

pub(crate) fn emitter_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(SceneEmitter)).unwrap_or(Value::Null)
}

pub(crate) fn emitter_example() -> Value {
    json!({})
}
