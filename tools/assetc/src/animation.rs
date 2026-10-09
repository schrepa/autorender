//! glTF skins and animations: skins keep their joint
//! nodes and inverse bind matrices; animations become clips in `EAN1` files
//! targeting node entities.

use engine_core::{AnimatedProperty, AnimationChannel, KeyInterpolation, Keyframes};
use engine_scene::{SceneAnimation, SceneDocument, SceneSkin, encode_animation_buffer};
use glam::{Quat, Vec3};
use gltf::{
    accessor::{DataType, Dimensions},
    animation::{Interpolation, Property, util::ReadOutputs},
};

use crate::{ImportError, ImportedFile, unsupported};

pub(crate) fn import(
    document: &gltf::Document,
    binary: &[u8],
    id: &str,
    output: &mut SceneDocument,
    files: &mut Vec<ImportedFile>,
) -> Result<(), ImportError> {
    let buffer = |buffer: gltf::Buffer<'_>| (buffer.index() == 0).then_some(binary);
    for skin in document.skins() {
        let path = format!("$.skins[{}]", skin.index());
        let joints: Vec<String> = skin
            .joints()
            .map(|joint| format!("{id}.node{}", joint.index()))
            .collect();
        if let Some(accessor) = skin.inverse_bind_matrices()
            && (accessor.data_type() != DataType::F32
                || accessor.dimensions() != Dimensions::Mat4
                || accessor.count() != joints.len())
        {
            return Err(ImportError::new(
                "invalid_skin",
                format!("{path}.inverseBindMatrices"),
                "inverse bind matrices are one FLOAT MAT4 per joint",
            ));
        }
        let inverse_binds = skin
            .reader(buffer)
            .read_inverse_bind_matrices()
            .map(|matrices| {
                matrices
                    .map(|columns| {
                        let mut values = [0.0; 16];
                        for (column, source) in
                            values.as_chunks_mut::<4>().0.iter_mut().zip(columns)
                        {
                            *column = source;
                        }
                        values
                    })
                    .collect()
            })
            .unwrap_or_default();
        output.skins.push(SceneSkin {
            id: format!("{id}.skin{}", skin.index()),
            joints,
            inverse_binds,
        });
    }
    for animation in document.animations() {
        let path = format!("$.animations[{}]", animation.index());
        let mut targets: Vec<String> = Vec::new();
        let mut channels = Vec::new();
        for channel in animation.channels() {
            let channel_path = format!("{path}.channels[{}]", channel.index());
            let target_id = format!("{id}.node{}", channel.target().node().index());
            let target = match targets.iter().position(|target| *target == target_id) {
                Some(target) => target,
                None => {
                    targets.push(target_id);
                    targets.len() - 1
                }
            };
            let property = match channel.target().property() {
                Property::Translation => AnimatedProperty::Translation,
                Property::Rotation => AnimatedProperty::Rotation,
                Property::Scale => AnimatedProperty::Scale,
                Property::MorphTargetWeights => {
                    return Err(unsupported(
                        format!("{channel_path}.target.path"),
                        "morph target weights are unsupported",
                    ));
                }
            };
            let sampler = channel.sampler();
            let interpolation = match sampler.interpolation() {
                Interpolation::Step => KeyInterpolation::Step,
                Interpolation::Linear => KeyInterpolation::Linear,
                Interpolation::CubicSpline => KeyInterpolation::CubicSpline,
            };
            let input = sampler.input();
            if input.data_type() != DataType::F32 || input.dimensions() != Dimensions::Scalar {
                return Err(ImportError::new(
                    "invalid_animation",
                    format!("{path}.samplers[{}].input", sampler.index()),
                    "key times are FLOAT scalars",
                ));
            }
            let output_accessor = sampler.output();
            let expected = if property == AnimatedProperty::Rotation {
                Dimensions::Vec4
            } else {
                Dimensions::Vec3
            };
            if output_accessor.dimensions() != expected
                || (property != AnimatedProperty::Rotation
                    && output_accessor.data_type() != DataType::F32)
            {
                return Err(ImportError::new(
                    "invalid_animation",
                    format!("{path}.samplers[{}].output", sampler.index()),
                    "translations and scales are FLOAT VEC3, rotations VEC4",
                ));
            }
            let reader = channel.reader(buffer);
            let times: Vec<f32> = reader
                .read_inputs()
                .ok_or_else(|| {
                    ImportError::new("invalid_accessor", &channel_path, "cannot read key times")
                })?
                .collect();
            let keyframes = match reader.read_outputs().ok_or_else(|| {
                ImportError::new("invalid_accessor", &channel_path, "cannot read key values")
            })? {
                ReadOutputs::Translations(values) | ReadOutputs::Scales(values) => {
                    Keyframes::Vectors(values.map(Vec3::from_array).collect())
                }
                ReadOutputs::Rotations(values) => {
                    Keyframes::Rotations(values.into_f32().map(Quat::from_array).collect())
                }
                ReadOutputs::MorphTargetWeights(_) => {
                    return Err(unsupported(
                        channel_path,
                        "morph target weights are unsupported",
                    ));
                }
            };
            let (times, keyframes) = distinct_keys(times, keyframes, interpolation);
            channels.push(
                AnimationChannel::new(target as u32, property, interpolation, times, keyframes)
                    .map_err(|error| {
                        ImportError::new("invalid_animation", &channel_path, error.to_string())
                    })?,
            );
        }
        if channels.is_empty() {
            return Err(ImportError::new(
                "invalid_animation",
                format!("{path}.channels"),
                "an animation needs at least one channel",
            ));
        }
        let file = ImportedFile::named(
            id,
            &format!("anim{}", animation.index()),
            "bin",
            encode_animation_buffer(&channels),
        );
        output.animations.push(SceneAnimation {
            id: format!("{id}.anim{}", animation.index()),
            name: animation
                .name()
                .filter(|name| name.len() <= 128 && !name.chars().any(char::is_control))
                .unwrap_or_default()
                .to_owned(),
            targets,
            channels: None,
            uri: Some(file.uri.clone()),
        });
        files.push(file);
    }
    Ok(())
}

/// Drops keys at or before an earlier key's time, which some exporters
/// write; the first of equal times is kept.
fn distinct_keys(
    times: Vec<f32>,
    keyframes: Keyframes,
    interpolation: KeyInterpolation,
) -> (Vec<f32>, Keyframes) {
    let per_key = if interpolation == KeyInterpolation::CubicSpline {
        3
    } else {
        1
    };
    let mut kept = Vec::with_capacity(times.len());
    let mut last = f32::NEG_INFINITY;
    for (index, time) in times.iter().enumerate() {
        if *time > last {
            kept.push(index);
            last = *time;
        }
    }
    if kept.len() == times.len() {
        return (times, keyframes);
    }
    let select = |count: usize| -> Vec<usize> {
        kept.iter()
            .flat_map(|key| {
                (key * per_key..(key + 1) * per_key).filter(move |value| *value < count)
            })
            .collect()
    };
    let keyframes = match keyframes {
        Keyframes::Vectors(values) => Keyframes::Vectors(
            select(values.len())
                .into_iter()
                .map(|value| values[value])
                .collect(),
        ),
        Keyframes::Rotations(values) => Keyframes::Rotations(
            select(values.len())
                .into_iter()
                .map(|value| values[value])
                .collect(),
        ),
    };
    (kept.into_iter().map(|key| times[key]).collect(), keyframes)
}
