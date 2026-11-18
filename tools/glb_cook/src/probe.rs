use std::path::Path;

use engine_rig::{Clip, Posed, Skeleton, bind, blend};
use glam::{Quat, Vec3};
use serde_json::{Value, json};

use crate::fault::Fault;

fn read(path: &str) -> Result<Vec<u8>, Fault> {
    std::fs::read(Path::new(path))
        .map_err(|error| Fault::new("FileUnreadable", path, "file", error))
}

fn clip(path: &str) -> Result<Clip, Fault> {
    Clip::from_bytes(&read(path)?).map_err(|error| Fault::new("ClipInvalid", path, "file", error))
}

fn skeleton(path: &str) -> Result<Skeleton, Fault> {
    Skeleton::from_bytes(&read(path)?)
        .map_err(|error| Fault::new("SkeletonInvalid", path, "file", error))
}

fn number<T: std::str::FromStr<Err: std::fmt::Display>>(
    text: &str,
    item: &str,
) -> Result<T, Fault> {
    text.parse()
        .map_err(|error| Fault::new("Number", "arguments", item, format!("`{text}`: {error}")))
}

fn joint(skeleton: &Skeleton, path: &str, name: &str) -> Result<usize, Fault> {
    skeleton
        .find(name)
        .ok_or_else(|| Fault::new("JointMissing", path, "joint", name))
}

fn sampled(
    skeleton: &Skeleton,
    path: &str,
    seconds: f32,
) -> Result<(Vec<Posed>, Option<usize>, usize), Fault> {
    let clip = clip(path)?;
    let bound = bind(skeleton, &clip);
    let mut posed = vec![
        Posed {
            rotation: Quat::IDENTITY,
            position: Vec3::ZERO,
        };
        skeleton.len()
    ];
    bound
        .sample(&clip, seconds, skeleton, &mut posed)
        .map_err(|error| Fault::new("Sample", path, "sample", error))?;
    Ok((posed, bound.root_joint(), bound.skipped()))
}

pub fn pose(
    clip_path: &str,
    skeleton_path: &str,
    seconds: &str,
    name: &str,
) -> Result<Value, Fault> {
    let skeleton = skeleton(skeleton_path)?;
    let seconds: f32 = number(seconds, "seconds")?;
    let index = joint(&skeleton, skeleton_path, name)?;
    let (posed, _, skipped) = sampled(&skeleton, clip_path, seconds)?;
    Ok(json!({
        "joint": name,
        "seconds": seconds,
        "position": posed[index].position.to_array(),
        "skipped": skipped,
    }))
}

pub fn mixed(words: &[&str; 8]) -> Result<Value, Fault> {
    let [
        from_path,
        from_seconds,
        to_path,
        to_seconds,
        skeleton_path,
        top,
        feather,
        name,
    ] = *words;
    let skeleton = skeleton(skeleton_path)?;
    let index = joint(&skeleton, skeleton_path, name)?;
    let (from, root_motion, _) = sampled(&skeleton, from_path, number(from_seconds, "seconds a")?)?;
    let (to, _, _) = sampled(&skeleton, to_path, number(to_seconds, "seconds b")?)?;
    let weights = skeleton
        .mask(top, number(feather, "feather levels")?)
        .map_err(|error| Fault::new("Mask", skeleton_path, "mask joint", error))?;
    let mut out = from.clone();
    blend(&skeleton, &from, &to, &weights, root_motion, &mut out)
        .map_err(|error| Fault::new("Blend", skeleton_path, "blend", error))?;
    Ok(json!({
        "joint": name,
        "position": out[index].position.to_array(),
    }))
}
