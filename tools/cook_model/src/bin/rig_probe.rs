use std::process::ExitCode;

use cook::Refusal;
use engine_rig::{Clip, Posed, Skeleton, bind, blend};
use glam::{Quat, Vec3};
use serde_json::{Value, json};

fn read(path: &str) -> Result<Vec<u8>, Refusal> {
    std::fs::read(path).map_err(|error| Refusal::new("FileUnreadable", path, error))
}

fn clip(path: &str) -> Result<Clip, Refusal> {
    Clip::from_bytes(&read(path)?).map_err(|error| Refusal::new("ClipInvalid", path, error))
}

fn skeleton(path: &str) -> Result<Skeleton, Refusal> {
    Skeleton::from_bytes(&read(path)?)
        .map_err(|error| Refusal::new("SkeletonInvalid", path, error))
}

fn number<T: std::str::FromStr<Err: std::fmt::Display>>(
    text: &str,
    item: &str,
) -> Result<T, Refusal> {
    text.parse()
        .map_err(|error| Refusal::new("Number", "arguments", format!("{item} `{text}`: {error}")))
}

fn joint(skeleton: &Skeleton, path: &str, name: &str) -> Result<usize, Refusal> {
    skeleton
        .find(name)
        .ok_or_else(|| Refusal::new("JointMissing", path, name))
}

fn sampled(
    skeleton: &Skeleton,
    path: &str,
    seconds: f32,
) -> Result<(Vec<Posed>, Option<usize>, usize), Refusal> {
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
        .map_err(|error| Refusal::new("Sample", path, error))?;
    Ok((posed, bound.root_joint(), bound.skipped()))
}

fn pose(clip_path: &str, skeleton_path: &str, seconds: &str, name: &str) -> Result<Value, Refusal> {
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

fn mixed(words: &[&str; 8]) -> Result<Value, Refusal> {
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
        .mask(joint(&skeleton, skeleton_path, top)?, number(feather, "feather levels")?)
        .ok_or_else(|| Refusal::new("Mask", skeleton_path, top))?;
    let mut out = from.clone();
    blend(&skeleton, &from, &to, &weights, root_motion, &mut out)
        .map_err(|error| Refusal::new("Blend", skeleton_path, error))?;
    Ok(json!({
        "joint": name,
        "position": out[index].position.to_array(),
    }))
}

fn refused(no: &Refusal) {
    println!("{}", json!({"error": no.code, "file": no.file, "detail": no.detail}));
}

fn answer(result: Result<Value, Refusal>) -> ExitCode {
    match result {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(no) => {
            refused(&no);
            ExitCode::FAILURE
        }
    }
}

fn usage() -> ExitCode {
    refused(&Refusal::new(
        "Usage",
        "rig_probe",
        "rig_probe --pose <file.clip> <file.skel> <seconds> <joint name> | rig_probe --blend <a.clip> <seconds a> <b.clip> <seconds b> <file.skel> <mask joint> <feather levels> <joint name>",
    ));
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let words: Vec<&str> = arguments.iter().map(String::as_str).collect();
    match words.as_slice() {
        ["--pose", clip, skeleton, seconds, joint] => answer(pose(clip, skeleton, seconds, joint)),
        ["--blend", rest @ ..] => match <&[&str; 8]>::try_from(rest) {
            Ok(words) => answer(mixed(words)),
            Err(_) => usage(),
        },
        _ => usage(),
    }
}
