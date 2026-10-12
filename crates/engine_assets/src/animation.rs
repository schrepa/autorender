//! Animation clips: keyframed translation, rotation and scale
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
    // Linear rotations: the arc from each key to the next, derived once.
    arcs: Box<[Arc]>,
}

/// The spherical interpolation from one rotation key to the next, derived
/// as `Quat::slerp` derives it: the next key on the near side, and either
/// linear blending when the keys nearly coincide or the angle between them
/// and its sine's reciprocal. [`Arc::at`] then gives `slerp`'s bits.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Arc {
    end: Quat,
    theta: f32,
    inverse_sine: f32,
    linear: bool,
}

impl Arc {
    fn new(start: Quat, end: Quat) -> Self {
        let mut end = end;
        let mut dot = start.dot(end);
        if dot < 0.0 {
            end = -end;
            dot = -dot;
        }
        if dot > 1.0 - f32::EPSILON {
            return Self {
                end,
                theta: 0.0,
                inverse_sine: 0.0,
                linear: true,
            };
        }
        let theta = acos_approx(dot);
        Self {
            end,
            theta,
            inverse_sine: 1.0 / engine_math::sin(theta),
            linear: false,
        }
    }

    /// `start.slerp(next, s)` for the arc from `start` to `next`.
    #[inline]
    fn at(&self, start: Quat, s: f32) -> Quat {
        if self.linear {
            // `end` lies on `start`'s side, as `slerp` leaves it.
            start.lerp(self.end, s)
        } else {
            let scale1 = engine_math::sin(self.theta * (1.0 - s));
            let scale2 = engine_math::sin(self.theta * s);
            ((start * scale1) + (self.end * scale2)) * self.inverse_sine
        }
    }
}

/// glam's `acos` approximation, which `Quat::slerp` uses: identical
/// operations give identical bits.
fn acos_approx(v: f32) -> f32 {
    let nonnegative = v >= 0.0;
    let x = v.abs();
    let mut omx = 1.0 - x;
    if omx < 0.0 {
        omx = 0.0;
    }
    let root = omx.sqrt();
    #[expect(clippy::approx_constant, reason = "the polynomial's own coefficients")]
    let mut result =
        ((((((-0.001_262_491_1 * x + 0.006_670_09) * x - 0.017_088_126) * x + 0.030_891_88) * x
            - 0.050_174_303)
            * x
            + 0.088_978_99)
            * x
            - 0.214_598_8)
            * x
            + 1.570_796_3;
    result *= root;
    if nonnegative {
        result
    } else {
        core::f32::consts::PI - result
    }
}

/// Where a time falls among a channel's keys: the key at or before it, how
/// far toward the next key it is (0 before the first key and after the
/// last), and the time between those keys.
#[derive(Debug, Clone, Copy, PartialEq)]
struct KeyPosition {
    key: usize,
    fraction: f32,
    span: f32,
}

impl KeyPosition {
    fn find(times: &[f32], time: f32) -> Self {
        let next = times.partition_point(|key| *key <= time);
        if next == 0 {
            Self {
                key: 0,
                fraction: 0.0,
                span: 0.0,
            }
        } else if next == times.len() {
            Self {
                key: times.len() - 1,
                fraction: 0.0,
                span: 0.0,
            }
        } else {
            let span = times[next] - times[next - 1];
            Self {
                key: next - 1,
                fraction: (time - times[next - 1]) / span,
                span,
            }
        }
    }
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
        let arcs = match (&keyframes, interpolation) {
            (Keyframes::Rotations(rotations), KeyInterpolation::Linear) => rotations
                .windows(2)
                .map(|pair| Arc::new(pair[0], pair[1]))
                .collect(),
            _ => Box::default(),
        };
        Ok(Self {
            target,
            property,
            interpolation,
            times: times.into_boxed_slice(),
            keyframes,
            arcs,
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
        self.sample_at(KeyPosition::find(&self.times, time))
    }

    /// The channel's value at `position` among its keys.
    #[inline]
    fn sample_at(&self, position: KeyPosition) -> AnimationSample {
        let KeyPosition {
            key,
            fraction,
            span,
        } = position;
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
                    AnimatedProperty::Translation => AnimationSample::Translation(value),
                    AnimatedProperty::Scale => AnimationSample::Scale(value),
                    AnimatedProperty::Rotation => {
                        unreachable!(
                            "rotation channels hold quaternion keys (AnimationChannel::new)"
                        )
                    }
                }
            }
            Keyframes::Rotations(values) => AnimationSample::Rotation(match self.interpolation {
                KeyInterpolation::Step => values[key],
                // `values[key].slerp(values[key + 1], fraction)`.
                KeyInterpolation::Linear if has_next => self.arcs[key].at(values[key], fraction),
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
            + std::mem::size_of_val(&self.arcs[..])
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
/// targets for [`Self::duration`] seconds. Players bind the
/// targets to nodes, so one clip serves every instance of a model.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationClip {
    channels: Box<[AnimationChannel]>,
    // Whether each channel is keyed at the same times as the one before.
    shared_times: Box<[bool]>,
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
        let shared_times = channels
            .iter()
            .enumerate()
            .map(|(index, channel)| {
                index > 0 && channels[index - 1].times.as_ref() == channel.times.as_ref()
            })
            .collect();
        Ok(Self {
            channels: channels.into_boxed_slice(),
            shared_times,
            target_count,
            duration,
        })
    }

    pub fn channels(&self) -> &[AnimationChannel] {
        &self.channels
    }

    /// Samples every channel at `time` in order, passing each channel and
    /// its value ([`AnimationChannel::sample`]'s). Channels keyed at the
    /// same times as the channel before them share its key search.
    pub fn sample_channels(
        &self,
        time: f32,
        mut sampled: impl FnMut(&AnimationChannel, AnimationSample),
    ) {
        let mut position = None;
        for (channel, &shared) in self.channels.iter().zip(&self.shared_times) {
            let at = match position {
                Some(position) if shared => position,
                _ => KeyPosition::find(&channel.times, time),
            };
            position = Some(at);
            sampled(channel, channel.sample_at(at));
        }
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
