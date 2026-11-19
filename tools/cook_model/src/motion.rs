use cook::Refusal;
use engine_rig::{Clip, Keys, RootTrack, RotationTrack, TranslationTrack};
use glam::{Mat4, Quat, Vec3, Vec4};
use gltf::animation::{Interpolation, Property, util::ReadOutputs};
use serde_json::{Value, json};

use crate::rig::Rig;
use crate::source::{Source, node_label};

struct Channel {
    joint: usize,
    property: Property,
    step: bool,
    times: Vec<f32>,
    values: Vec<Vec4>,
}

pub struct Cooked {
    pub clip: Clip,
    pub dropped: Vec<Value>,
    pub travel: Value,
}

fn property_name(property: Property) -> &'static str {
    match property {
        Property::Translation => "translation",
        Property::Rotation => "rotation",
        Property::Scale => "scale",
        Property::MorphTargetWeights => "weights",
    }
}

fn outputs(outputs: ReadOutputs<'_>) -> Vec<Vec4> {
    match outputs {
        ReadOutputs::Translations(values) | ReadOutputs::Scales(values) => values
            .map(|value| Vec3::from_array(value).extend(0.0))
            .collect(),
        ReadOutputs::Rotations(values) => values.into_f32().map(Vec4::from_array).collect(),
        ReadOutputs::MorphTargetWeights(values) => values
            .into_f32()
            .map(|value| Vec4::new(value, 0.0, 0.0, 0.0))
            .collect(),
    }
}

impl Channel {
    fn at(&self, seconds: f32) -> Vec4 {
        let last = self.times.len() - 1;
        let next = self.times.partition_point(|&time| time <= seconds);
        if next == 0 {
            return self.values[0];
        }
        if next > last {
            return self.values[last];
        }
        let first = next - 1;
        if self.step {
            return self.values[first];
        }
        let fraction = (seconds - self.times[first]) / (self.times[next] - self.times[first]);
        let (from, to) = (self.values[first], self.values[next]);
        match self.property {
            Property::Rotation => {
                Vec4::from(Quat::from_vec4(from).slerp(Quat::from_vec4(to), fraction))
            }
            Property::Translation | Property::Scale | Property::MorphTargetWeights => {
                from + (to - from) * fraction
            }
        }
    }
}

fn read_channels(
    source: &Source,
    rig: &Rig,
    animation: &gltf::Animation<'_>,
    dropped: &mut Vec<Value>,
) -> Result<Vec<Channel>, Refusal> {
    let label = animation.name().unwrap_or_default().to_owned();
    let mut channels = Vec::new();
    for channel in animation.channels() {
        let target = channel.target();
        let node = target.node();
        let property = target.property();
        let item = format!(
            "animation `{label}` {} {}",
            node_label(&node),
            property_name(property)
        );
        let step = match channel.sampler().interpolation() {
            Interpolation::Linear => false,
            Interpolation::Step => true,
            Interpolation::CubicSpline => {
                return Err(source.refuse("CubicSpline", item, "cubic spline keys are not cooked"));
            }
        };
        let Some(joint) = rig.joint_of_node(node.index()) else {
            dropped.push(json!({"bone": node.name(), "property": property_name(property), "reason": "not a joint"}));
            continue;
        };
        if property == Property::MorphTargetWeights {
            dropped.push(
                json!({"bone": node.name(), "property": "weights", "reason": "morph weights"}),
            );
            continue;
        }
        let reader = channel.reader(|held| source.data(&held));
        let times: Vec<f32> = reader
            .read_inputs()
            .ok_or_else(|| source.refuse("ChannelUnreadable", item.clone(), "no key times"))?
            .collect();
        let values =
            outputs(reader.read_outputs().ok_or_else(|| {
                source.refuse("ChannelUnreadable", item.clone(), "no key values")
            })?);
        if times.is_empty() || values.len() != times.len() {
            return Err(source.refuse(
                "ChannelUnreadable",
                item,
                format!("{} times, {} values", times.len(), values.len()),
            ));
        }
        channels.push(Channel {
            joint,
            property,
            step,
            times,
            values,
        });
    }
    Ok(channels)
}

struct Pose {
    rotations: Vec<Quat>,
    positions: Vec<Vec3>,
}

fn pose(
    source: &Source,
    rig: &Rig,
    seconds: f32,
    joint_channels: &[[Option<usize>; 3]],
    all: &[Channel],
) -> Pose {
    let count = rig.nodes.len();
    let mut rotations = Vec::with_capacity(count);
    let mut worlds: Vec<Mat4> = Vec::with_capacity(count);
    for (joint, held) in joint_channels.iter().enumerate() {
        let node = source.node(rig.nodes[joint]);
        let (rest_translation, rest_rotation, rest_scale) = node.transform().decomposed();
        let value = |slot: usize| held[slot].map(|index| all[index].at(seconds));
        let translation = value(0).map_or(Vec3::from_array(rest_translation), Vec4::truncate);
        let rotation = value(1)
            .map_or(Quat::from_array(rest_rotation), Quat::from_vec4)
            .normalize();
        let scale = value(2).map_or(Vec3::from_array(rest_scale), Vec4::truncate);
        let own = Mat4::from_scale_rotation_translation(scale, rotation, translation);
        let parent = rig.skeleton.joints()[joint].parent;
        let (world, turned) = match parent {
            Some(parent) => (worlds[parent] * own, rotations[parent] * rotation),
            None => (own, rotation),
        };
        worlds.push(world);
        rotations.push(turned);
    }
    let positions = worlds
        .iter()
        .map(|world| world.w_axis.truncate() * rig.scale)
        .collect();
    let rotations = rotations
        .iter()
        .zip(&rig.frames)
        .map(|(rotation, frame)| (*rotation * *frame).normalize())
        .collect();
    Pose {
        rotations,
        positions,
    }
}

fn grid(duration: f32, keys_per_second: f32) -> Vec<f32> {
    let mut times: Vec<f32> = (0u32..)
        .map(|key| key as f32 / keys_per_second)
        .take_while(|&time| time < duration)
        .collect();
    if times.last().is_none_or(|&last| last < duration) {
        times.push(duration);
    }
    times
}

fn same(a: Quat, b: Quat, tolerance: f32) -> bool {
    let b = if a.dot(b) < 0.0 { -b } else { b };
    (Vec4::from(a) - Vec4::from(b)).abs().max_element() <= tolerance
}

fn chain_step(
    rig: &Rig,
    joint: usize,
    joint_channels: &[[Option<usize>; 3]],
    all: &[Channel],
    slots: &[usize],
) -> Keys {
    let mut held = Some(joint);
    let mut any = false;
    while let Some(at) = held {
        for &slot in slots {
            if let Some(index) = joint_channels[at][slot] {
                any = true;
                if !all[index].step {
                    return Keys::Linear;
                }
            }
        }
        held = rig.skeleton.joints()[at].parent;
    }
    if any { Keys::Step } else { Keys::Linear }
}

fn dropped_channels(
    source: &Source,
    rig: &Rig,
    kept: &[bool],
    all: &[Channel],
    tolerance: f32,
    dropped: &mut Vec<Value>,
) {
    for channel in all {
        let node = source.node(rig.nodes[channel.joint]);
        let (translation, _, _) = node.transform().decomposed();
        let (rest, weight, name) = match channel.property {
            Property::Translation if !kept[channel.joint] => (
                Vec3::from_array(translation).extend(0.0),
                rig.scale,
                "translation",
            ),
            Property::Scale => (Vec4::new(1.0, 1.0, 1.0, 0.0), 1.0, "scale"),
            Property::Translation | Property::Rotation | Property::MorphTargetWeights => continue,
        };
        let animated = channel
            .values
            .iter()
            .any(|value| ((*value - rest) * weight).abs().max_element() > tolerance);
        dropped.push(json!({
            "bone": node.name(),
            "property": name,
            "reason": if animated { "animated" } else { "constant at rest" },
        }));
    }
}

struct Sampled<'a> {
    rig: &'a Rig,
    all: &'a [Channel],
    joint_channels: Vec<[Option<usize>; 3]>,
    poses: Vec<Pose>,
    tolerance: f32,
}

fn rotation_tracks(sampled: &Sampled<'_>, dropped: &mut Vec<Value>) -> Vec<RotationTrack> {
    let rig = sampled.rig;
    let keys = sampled.poses.len();
    let mut effective: Vec<Vec<Quat>> = Vec::with_capacity(rig.nodes.len());
    let mut rotations = Vec::new();
    for joint in 0..rig.nodes.len() {
        let inverse_rest = rig.skeleton.rest(joint).rotation.conjugate();
        let deltas: Vec<Quat> = sampled
            .poses
            .iter()
            .map(|pose| (pose.rotations[joint] * inverse_rest).normalize())
            .collect();
        let parent = rig.skeleton.joints()[joint].parent;
        let follows = deltas.iter().enumerate().all(|(key, &delta)| {
            let inherited = parent.map_or(Quat::IDENTITY, |parent| effective[parent][key]);
            same(delta, inherited, sampled.tolerance)
        });
        let name = rig.skeleton.joints()[joint].name.clone();
        if follows {
            dropped.push(json!({"bone": name, "property": "rotation", "reason": "follows parent"}));
            let inherited = match parent {
                Some(parent) => effective[parent].clone(),
                None => vec![Quat::IDENTITY; keys],
            };
            effective.push(inherited);
        } else {
            rotations.push(RotationTrack {
                bone: name,
                keys: chain_step(rig, joint, &sampled.joint_channels, sampled.all, &[1]),
                values: deltas.clone(),
            });
            effective.push(deltas);
        }
    }
    rotations
}

fn translation_tracks(
    sampled: &Sampled<'_>,
    hip: Option<usize>,
) -> (Vec<TranslationTrack>, Vec<bool>) {
    let rig = sampled.rig;
    let mut kept = vec![false; rig.nodes.len()];
    let mut tracks = Vec::new();
    for (joint, held) in kept.iter_mut().enumerate() {
        if hip == Some(joint) {
            *held = true;
            continue;
        }
        let parent = rig.skeleton.joints()[joint].parent;
        let rest = rig.skeleton.rest(joint).position;
        let rest_offset = parent.map_or(rest, |parent| {
            let above = rig.skeleton.rest(parent);
            above.rotation.conjugate() * (rest - above.position)
        });
        let values: Vec<Vec3> = sampled
            .poses
            .iter()
            .map(|pose| {
                let offset = parent.map_or(pose.positions[joint], |parent| {
                    pose.rotations[parent].conjugate()
                        * (pose.positions[joint] - pose.positions[parent])
                });
                offset - rest_offset
            })
            .collect();
        if values
            .iter()
            .all(|value| value.abs().max_element() <= sampled.tolerance)
        {
            continue;
        }
        *held = true;
        tracks.push(TranslationTrack {
            bone: rig.skeleton.joints()[joint].name.clone(),
            keys: chain_step(rig, joint, &sampled.joint_channels, sampled.all, &[0, 1]),
            values,
        });
    }
    (tracks, kept)
}

fn horizontal(value: Vec3) -> [f32; 2] {
    [value.x, value.z]
}

fn travel(deltas: &mut [Vec3], times: &[f32], height: f32, remove: bool) -> Value {
    let count = deltas.len() as f32;
    let duration = times.last().copied().unwrap_or_default();
    let flat = |value: Vec3| Vec3::new(value.x, 0.0, value.z);
    let mean = deltas
        .iter()
        .fold(Vec3::ZERO, |sum, delta| sum + flat(*delta))
        / count;
    let drift = flat(deltas[deltas.len() - 1] - deltas[0]);
    if remove {
        let share = |time: f32| if duration > 0.0 { time / duration } else { 0.0 };
        let mean_share = times.iter().map(|&time| share(time)).sum::<f32>() / count;
        for (delta, &time) in deltas.iter_mut().zip(times) {
            *delta -= mean + drift * (share(time) - mean_share);
        }
    }
    json!({
        "mean_m": horizontal(mean * height),
        "drift_m": horizontal(drift * height),
        "removed": remove,
    })
}

pub struct Settings<'a> {
    pub root_motion: Option<&'a str>,
    pub remove_travel: bool,
    pub keys_per_second: f32,
    pub tolerance: f32,
}

fn hip_joint(
    source: &Source,
    rig: &Rig,
    label: &str,
    root_motion: Option<&str>,
) -> Result<Option<(usize, Vec3)>, Refusal> {
    let Some(root_motion) = root_motion else {
        return Ok(None);
    };
    let hip = rig.skeleton.find(root_motion).ok_or_else(|| {
        source.refuse(
            "RootMotionBone",
            format!("animation `{label}`"),
            format!("no joint `{root_motion}`"),
        )
    })?;
    let rest = rig.skeleton.rest(hip).position;
    if rest.y <= 0.0 {
        return Err(source.refuse("HipHeight", format!("joint `{root_motion}`"), rest.y));
    }
    Ok(Some((hip, rest)))
}

fn root_track(
    sampled: &Sampled<'_>,
    hip: Option<(usize, Vec3)>,
    times: &[f32],
    remove_travel: bool,
    dropped: &mut Vec<Value>,
) -> (Option<RootTrack>, Value) {
    let Some((joint, rest)) = hip else {
        return (None, Value::Null);
    };
    let rig = sampled.rig;
    let height = rest.y;
    let mut deltas: Vec<Vec3> = sampled
        .poses
        .iter()
        .map(|pose| (pose.positions[joint] - rest) / height)
        .collect();
    let moved = travel(&mut deltas, times, height, remove_travel);
    let bone = rig.skeleton.joints()[joint].name.clone();
    let still = deltas
        .iter()
        .all(|delta| (*delta * height).abs().max_element() <= sampled.tolerance);
    if still {
        dropped
            .push(json!({"bone": bone, "property": "translation", "reason": "constant at rest"}));
        return (None, moved);
    }
    let keys = chain_step(rig, joint, &sampled.joint_channels, sampled.all, &[0, 1]);
    (
        Some(RootTrack {
            bone,
            keys,
            values: deltas,
        }),
        moved,
    )
}

pub fn cook(
    source: &Source,
    rig: &Rig,
    animation: &gltf::Animation<'_>,
    settings: &Settings<'_>,
) -> Result<Cooked, Refusal> {
    let label = animation.name().unwrap_or_default().to_owned();
    let mut dropped = Vec::new();
    let all = read_channels(source, rig, animation, &mut dropped)?;
    let duration = all
        .iter()
        .filter_map(|channel| channel.times.last().copied())
        .fold(
            0.0,
            |longest: f32, time| {
                if time > longest { time } else { longest }
            },
        );
    if all.is_empty() || duration <= 0.0 {
        return Err(source.refuse(
            "EmptyAnimation",
            format!("animation `{label}`"),
            "no joint channel with length",
        ));
    }
    let hip = hip_joint(source, rig, &label, settings.root_motion)?;
    let mut joint_channels = vec![[None; 3]; rig.nodes.len()];
    for (index, channel) in all.iter().enumerate() {
        let slot = match channel.property {
            Property::Translation => 0,
            Property::Rotation => 1,
            Property::Scale | Property::MorphTargetWeights => 2,
        };
        joint_channels[channel.joint][slot] = Some(index);
    }
    let times = grid(duration, settings.keys_per_second);
    let poses: Vec<Pose> = times
        .iter()
        .map(|&time| pose(source, rig, time, &joint_channels, &all))
        .collect();
    let sampled = Sampled {
        rig,
        all: &all,
        joint_channels,
        poses,
        tolerance: settings.tolerance,
    };
    let rotations = rotation_tracks(&sampled, &mut dropped);
    let (translations, kept) = translation_tracks(&sampled, hip.map(|(joint, _)| joint));
    dropped_channels(source, rig, &kept, &all, settings.tolerance, &mut dropped);
    let (root, travel) = root_track(&sampled, hip, &times, settings.remove_travel, &mut dropped);
    let clip = Clip::new(duration, times, rotations, translations, root)
        .map_err(|error| source.refuse("Clip", format!("animation `{label}`"), error))?;
    Ok(Cooked {
        clip,
        dropped,
        travel,
    })
}
