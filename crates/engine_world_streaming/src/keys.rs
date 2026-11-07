use engine_core::parse_value;
use engine_places_rendering::lights::LocalLight;
use engine_places_rendering::particles::{ParticleEmitter, ParticleShape};
use engine_places_rendering::probes::ReflectionProbe;
use engine_places_world::streaming::{StreamedAt, StreamedKey, Streams};
use serde_json::Value;

pub(crate) const RENDERED: [StreamedKey; 3] = [
    StreamedKey {
        key: "light",
        streams: Streams::Reaching(light),
    },
    StreamedKey {
        key: "particles",
        streams: Streams::Reaching(particles),
    },
    StreamedKey {
        key: "reflection_probe",
        streams: Streams::Reaching(probe),
    },
];

fn light(value: &Value, at: &StreamedAt<'_>) -> f32 {
    parse_value::<LocalLight>(value.clone()).map_or(at.from, |light| at.from + light.range)
}

fn particles(value: &Value, at: &StreamedAt<'_>) -> f32 {
    let Ok(emitter) = parse_value::<ParticleEmitter>(value.clone()) else {
        return at.from;
    };
    let look = &emitter.look;
    let shape = match look.shape {
        ParticleShape::Box => look.extent.length(),
        ParticleShape::Ball => look.extent.max_element(),
    };
    at.from
        + emitter.offset.length()
        + shape
        + look.speed * look.lifetime
        + 0.5 * look.acceleration.length() * look.lifetime * look.lifetime
}

fn probe(value: &Value, at: &StreamedAt<'_>) -> f32 {
    parse_value::<ReflectionProbe>(value.clone()).map_or(at.from, |probe| at.from + probe.extent.length())
}
