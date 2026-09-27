//! Animation clips (decision 035): keyframed translation, rotation and scale
//! of target nodes, sampled as glTF 2.0 animations are.

use glam::{Quat, Vec3};

use crate::AssetError;

/// How a channel moves between keys (glTF sampler interpolation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyInterpolation {
    /// Holds each key's value until the next key.
    Step,
    /// Straight lines between keys; rotations take the shortest arc.
    Linear,
    /// Cubic Hermite curves through the keys, with an in-tangent and an
    /// out-tangent stored around each key's value.
    CubicSpline,
}

/// What a channel animates: one component of its target's local transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnimatedProperty {
    Translation,
    Rotation,
    Scale,
}

/// A sampled value of one channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnimationSample {
    Translation(Vec3),
    Rotation(Quat),
    Scale(Vec3),
}

/// Key values of one channel: vectors (translation, scale) as `[x, y, z]`,
/// rotations as quaternions `[x, y, z, w]`. Cubic-spline channels hold three
/// per key: in-tangent, value, out-tangent.
#[derive(Debug, Clone, PartialEq)]
pub enum Keyframes {
    Vectors(Box<[Vec3]>),
    Rotations(Box<[Quat]>),
}

/// One animated property of one target, keyed at strictly increasing times
/// in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationChannel {
    target: u32,
    property: AnimatedProperty,
    interpolation: KeyInterpolation,
    times: Box<[f32]>,
    keyframes: Keyframes,
}

impl AnimationChannel {
    /// A channel animating `property` of target `target` (an index into the
    /// targets a player binds). Times must be finite, not negative and
    /// strictly increasing; there is one value per time (three for cubic
    /// splines), finite, rotations nonzero (they are normalized).
    pub fn new(
        target: u32,
        property: AnimatedProperty,
        interpolation: KeyInterpolation,
        times: Vec<f32>,
        keyframes: Keyframes,
    ) -> Result<Self, AssetError> {
        let invalid = |reason| Err(AssetError::InvalidAnimation { reason });
        if times.is_empty() {
            return invalid("a channel needs at least one key");
        }
        if !times.iter().all(|time| time.is_finite() && *time >= 0.0)
            || times.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return invalid("key times must be finite, not negative and strictly increasing");
        }
        let per_key = if interpolation == KeyInterpolation::CubicSpline {
            3
        } else {
            1
        };
        let keyframes = match (property, keyframes) {
            (AnimatedProperty::Rotation, Keyframes::Rotations(rotations)) => {
                if rotations.len() != times.len() * per_key {
                    return invalid("a channel needs one value per key (three for cubic splines)");
                }
                let mut rotations = rotations.into_vec();
                for (index, rotation) in rotations.iter_mut().enumerate() {
                    if !rotation.is_finite() {
                        return invalid("rotations must be finite");
                    }
                    // Tangents keep their scale; values are unit quaternions.
                    if per_key == 1 || index % 3 == 1 {
                        if rotation.length_squared() == 0.0 {
                            return invalid("rotations must be nonzero");
                        }
                        *rotation = rotation.normalize();
                    }
                }
                Keyframes::Rotations(rotations.into_boxed_slice())
            }
            (
                AnimatedProperty::Translation | AnimatedProperty::Scale,
                Keyframes::Vectors(vectors),
            ) => {
                if vectors.len() != times.len() * per_key {
                    return invalid("a channel needs one value per key (three for cubic splines)");
                }
                if !vectors.iter().all(|vector| vector.is_finite()) {
                    return invalid("translations and scales must be finite");
                }
                Keyframes::Vectors(vectors)
            }
            _ => return invalid("rotations animate with quaternions, others with vectors"),
        };
        Ok(Self {
            target,
            property,
            interpolation,
            times: times.into_boxed_slice(),
            keyframes,
        })
    }

    pub fn target(&self) -> u32 {
        self.target
    }

    pub fn property(&self) -> AnimatedProperty {
        self.property
    }

    pub fn interpolation(&self) -> KeyInterpolation {
        self.interpolation
    }

    pub fn times(&self) -> &[f32] {
        &self.times
    }

    pub fn keyframes(&self) -> &Keyframes {
        &self.keyframes
    }

    /// The channel's value at `time` seconds: its first value before the
    /// first key and its last after the last.
    pub fn sample(&self, time: f32) -> AnimationSample {
        let times = &self.times;
        // The key at or before `time`, and how far toward the next it is.
        let next = times.partition_point(|key| *key <= time);
        let (key, fraction, span) = if next == 0 {
            (0, 0.0, 0.0)
        } else if next == times.len() {
            (times.len() - 1, 0.0, 0.0)
        } else {
            let span = times[next] - times[next - 1];
            (next - 1, (time - times[next - 1]) / span, span)
        };
        let has_next = fraction > 0.0;
        match &self.keyframes {
            Keyframes::Vectors(values) => {
                let value = match self.interpolation {
                    KeyInterpolation::Step => values[key],
                    KeyInterpolation::Linear if has_next => {
                        values[key].lerp(values[key + 1], fraction)
                    }
                    KeyInterpolation::Linear => values[key],
                    KeyInterpolation::CubicSpline if has_next => hermite(
                        values[key * 3 + 1],
                        values[key * 3 + 2] * span,
                        values[(key + 1) * 3 + 1],
                        values[(key + 1) * 3] * span,
                        fraction,
                    ),
                    KeyInterpolation::CubicSpline => values[key * 3 + 1],
                };
                match self.property {
                    AnimatedProperty::Scale => AnimationSample::Scale(value),
                    _ => AnimationSample::Translation(value),
                }
            }
            Keyframes::Rotations(values) => AnimationSample::Rotation(match self.interpolation {
                KeyInterpolation::Step => values[key],
                KeyInterpolation::Linear if has_next => {
                    values[key].slerp(values[key + 1], fraction)
                }
                KeyInterpolation::Linear => values[key],
                KeyInterpolation::CubicSpline if has_next => {
                    let value = |index: usize| {
                        let rotation = values[index];
                        glam::Vec4::new(rotation.x, rotation.y, rotation.z, rotation.w)
                    };
                    let curve = hermite4(
                        value(key * 3 + 1),
                        value(key * 3 + 2) * span,
                        value((key + 1) * 3 + 1),
                        value((key + 1) * 3) * span,
                        fraction,
                    );
                    let length = curve.length();
                    if length > 0.0 && length.is_finite() {
                        Quat::from_vec4(curve / length)
                    } else {
                        values[key * 3 + 1]
                    }
                }
                KeyInterpolation::CubicSpline => values[key * 3 + 1],
            }),
        }
    }

    fn byte_len(&self) -> usize {
        std::mem::size_of_val(self.times())
            + match &self.keyframes {
                Keyframes::Vectors(values) => std::mem::size_of_val(&values[..]),
                Keyframes::Rotations(values) => std::mem::size_of_val(&values[..]),
            }
    }
}

/// The cubic Hermite curve from `p0` (tangent `m0`) to `p1` (tangent `m1`)
/// at `t` in `[0, 1]`; tangents are scaled by the key span.
fn hermite(p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3, t: f32) -> Vec3 {
    let [a, b, c, d] = hermite_weights(t);
    p0 * a + m0 * b + p1 * c + m1 * d
}

fn hermite4(p0: glam::Vec4, m0: glam::Vec4, p1: glam::Vec4, m1: glam::Vec4, t: f32) -> glam::Vec4 {
    let [a, b, c, d] = hermite_weights(t);
    p0 * a + m0 * b + p1 * c + m1 * d
}

fn hermite_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        2.0 * t3 - 3.0 * t2 + 1.0,
        t3 - 2.0 * t2 + t,
        -2.0 * t3 + 3.0 * t2,
        t3 - t2,
    ]
}

/// An immutable set of channels animating up to [`Self::target_count`]
/// targets for [`Self::duration`] seconds (decision 035). Players bind the
/// targets to nodes, so one clip serves every instance of a model.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationClip {
    channels: Box<[AnimationChannel]>,
    target_count: u32,
    duration: f32,
}

impl AnimationClip {
    /// A clip of `channels`, at least one; its duration is its latest key.
    pub fn new(channels: Vec<AnimationChannel>) -> Result<Self, AssetError> {
        if channels.is_empty() {
            return Err(AssetError::InvalidAnimation {
                reason: "a clip needs at least one channel",
            });
        }
        let target_count = channels
            .iter()
            .map(|channel| channel.target + 1)
            .fold(0, u32::max);
        let duration = channels
            .iter()
            .map(|channel| *channel.times.last().expect("channels have keys"))
            .fold(
                0.0_f32,
                |longest, time| if time > longest { time } else { longest },
            );
        Ok(Self {
            channels: channels.into_boxed_slice(),
            target_count,
            duration,
        })
    }

    pub fn channels(&self) -> &[AnimationChannel] {
        &self.channels
    }

    /// One more than the largest target index a channel names.
    pub fn target_count(&self) -> u32 {
        self.target_count
    }

    /// The latest key time in seconds.
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// Owned key bytes; excludes allocation and `Arc` metadata.
    pub fn byte_len(&self) -> usize {
        self.channels.iter().map(AnimationChannel::byte_len).sum()
    }
}
