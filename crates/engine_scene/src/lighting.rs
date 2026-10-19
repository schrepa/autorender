//! Scene entity fields: local lights and particle
//! emitters. Each mirrors its runtime component field for
//! field, so inspected values paste back into scenes; angles are in degrees.

use engine_core::{
    LocalLight, MAX_SPOT_DEGREES, MAX_SPREAD_DEGREES, MIN_SPOT_DEGREES, ParticleBlend,
    ParticleEmitter, Spot, bevy_ecs::world::EntityWorldMut, glam::Vec3,
};
use engine_settings::{ExtraRules, InvalidSetting, Setting, settings};
use serde::{Deserialize, Serialize};

use crate::components::{Placement, Reach, SceneComponent};
use crate::instance::Lookups;
use crate::{SceneDocument, SceneError, SceneFiles};

settings! {
    /// A point or spot light on this entity.
    /// On an entity with a mesh it follows the mesh's displayed pose.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct SceneLight {
        /// Linear color.
        pub color: [f32; 3] = LocalLight::default().color.to_array()
            => same(LocalLight::rule_of("color")),
        /// Light at one world unit.
        pub intensity: f32 = LocalLight::default().intensity
            => same(LocalLight::rule_of("intensity")),
        /// World distance beyond which the light has no effect.
        pub range: f32 = LocalLight::default().range => same(LocalLight::rule_of("range")),
        /// Position in the entity's local space, scaled with it.
        pub offset: [f32; 3] = LocalLight::default().offset.to_array()
            => same(LocalLight::rule_of("offset")),
        /// A cone; omitted for light in every direction.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub spot: Option<SceneSpot> = None => any,
    }
}

settings! {
    extra_rules;
    /// The cone of a spot light.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct SceneSpot {
        /// Direction the cone points in the entity's local space; nonzero.
        pub direction: [f32; 3] = Spot::default().direction.to_array()
            => same(Spot::rule_of("direction")),
        /// Half-angle inside which the light is full, in degrees.
        pub inner_angle_degrees: f32 = Spot::default().inner_angle.to_degrees()
            => range(0.0, MAX_SPOT_DEGREES),
        /// Half-angle beyond which it is dark, in degrees: at least the inner
        /// angle.
        pub outer_angle_degrees: f32 = Spot::default().outer_angle.to_degrees()
            => range(MIN_SPOT_DEGREES, MAX_SPOT_DEGREES),
    }
}

impl ExtraRules for SceneSpot {
    /// The runtime spot's extra rules.
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        self.spot().check_extra()
    }
}

impl SceneSpot {
    /// The runtime spot, unchecked.
    fn spot(&self) -> Spot {
        Spot {
            direction: Vec3::from_array(self.direction),
            inner_angle: self.inner_angle_degrees.to_radians(),
            outer_angle: self.outer_angle_degrees.to_radians(),
        }
    }
}

impl SceneLight {
    /// The component; its rules are checked with the entity's.
    pub(crate) fn runtime(&self) -> LocalLight {
        let spot = self.spot.map(|spot| spot.spot());
        LocalLight {
            color: Vec3::from_array(self.color),
            intensity: self.intensity,
            range: self.range,
            offset: Vec3::from_array(self.offset),
            spot,
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
        }
    }
}

impl Setting for SceneParticleBlend {}

/// How particles combine with what is behind them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneParticleBlend {
    /// Covering by alpha, lit by the sun and sky: smoke, dust, rain.
    #[default]
    Alpha,
    /// Adding light, unlit: sparks, fire.
    Additive,
    /// Balls of mist, soft through their depth, lit by the sky and the
    /// sun they scatter forward: spray, steam, fog banks.
    Mist,
}

/// The engine's default emitter, as authored.
fn default_emitter() -> SceneEmitter {
    SceneEmitter::from(&ParticleEmitter::default())
}

settings! {
    extra_rules;
    /// A particle emitter on this entity.
    /// Omitted fields take the engine's defaults.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct SceneEmitter {
        /// Particles born per second.
        pub rate: f32 = default_emitter().rate
            => same(ParticleEmitter::rule_of("rate")),
        /// Seconds each particle lives.
        pub lifetime: f32 = default_emitter().lifetime
            => same(ParticleEmitter::rule_of("lifetime")),
        /// Initial speed in world units per second.
        pub speed: f32 = default_emitter().speed
            => same(ParticleEmitter::rule_of("speed")),
        /// Direction of emission in the entity's local space; nonzero.
        pub direction: [f32; 3] = default_emitter().direction
            => same(ParticleEmitter::rule_of("direction")),
        /// Half-angle of the cone of initial directions, in degrees; 180 emits
        /// in every direction.
        pub spread_degrees: f32 = default_emitter().spread_degrees
            => range(0.0, MAX_SPREAD_DEGREES),
        /// World acceleration: gravity, buoyancy or wind.
        pub acceleration: [f32; 3] = default_emitter().acceleration
            => same(ParticleEmitter::rule_of("acceleration")),
        /// Diameter at birth and at death, in world units.
        pub size: [f32; 2] = default_emitter().size
            => same(ParticleEmitter::rule_of("size")),
        /// Linear RGBA at birth and at death; alpha fades.
        pub color: [[f32; 4]; 2] = default_emitter().color
            => same(ParticleEmitter::rule_of("color")),
        /// Brightness multiple of the color, for glowing particles.
        pub emission: f32 = default_emitter().emission
            => same(ParticleEmitter::rule_of("emission")),
        /// How particles combine with what is behind them.
        pub blend: SceneParticleBlend = default_emitter().blend => any,
        /// Seconds of travel particles stretch along their motion.
        pub stretch: f32 = default_emitter().stretch
            => same(ParticleEmitter::rule_of("stretch")),
        /// Birth position in the entity's local space.
        pub offset: [f32; 3] = default_emitter().offset
            => same(ParticleEmitter::rule_of("offset")),
        /// Radius of the ball particles start in.
        pub radius: f32 = default_emitter().radius
            => same(ParticleEmitter::rule_of("radius")),
        /// Varies random choices between otherwise identical emitters.
        pub seed: u32 = default_emitter().seed => any,
    }
}

impl ExtraRules for SceneEmitter {
    /// The runtime emitter's extra rules.
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
    /// The component; its rules are checked with the entity's.
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

impl SceneComponent for SceneLight {
    type Prepared = LocalLight;

    const PLACEMENT: Placement = Placement::Anywhere;

    fn prepare(
        &self,
        _: &str,
        _: &SceneDocument,
        _: Option<&SceneFiles>,
    ) -> Result<LocalLight, SceneError> {
        Ok(self.runtime())
    }

    fn spawn(prepared: &LocalLight, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*prepared);
    }

    // Its light reaches its range.
    fn reach(prepared: &LocalLight, reach: &Reach<'_>, extend: &mut impl FnMut(f32)) {
        extend(reach.from + prepared.range);
    }
}

impl SceneComponent for SceneEmitter {
    type Prepared = ParticleEmitter;

    const PLACEMENT: Placement = Placement::Anywhere;

    fn prepare(
        &self,
        _: &str,
        _: &SceneDocument,
        _: Option<&SceneFiles>,
    ) -> Result<ParticleEmitter, SceneError> {
        Ok(self.runtime())
    }

    fn spawn(prepared: &ParticleEmitter, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*prepared);
    }

    // Its particles reach as far as they fly in their lifetime.
    fn reach(emitter: &ParticleEmitter, reach: &Reach<'_>, extend: &mut impl FnMut(f32)) {
        extend(
            reach.from
                + emitter.offset.length()
                + emitter.radius
                + emitter.speed * emitter.lifetime
                + 0.5 * emitter.acceleration.length() * emitter.lifetime * emitter.lifetime,
        );
    }
}
