use std::collections::BTreeMap;

use cook::{Kind, Refusal, Step};
use engine_rig::{Clip as ClipFile, Skeleton};
use glam::{Mat4, Vec3};
use serde_json::{Value, json};

use crate::geometry::{self, budget, counted};
use crate::markers::{markers, write_markers};
use crate::material::{MaterialPlan, materials, write_materials, write_meshes, write_textures};
use crate::motion;
use crate::params::{
    Character, Clip, Form, Model, Static, check_clips, check_name, check_scale, map_kinds,
};
use crate::rig::{Rig, rig, rig_by_names};
use crate::source::Source;

pub static MODEL: Kind = Kind {
    name: "model",
    version: 1,
    run: model,
};

pub static FORM: Kind = Kind {
    name: "form",
    version: 1,
    run: form,
};

pub static CLIP: Kind = Kind {
    name: "clip",
    version: 1,
    run: clip,
};

fn sources<const N: usize>(step: &Step<'_>) -> Result<[Source; N], Refusal> {
    let inputs = step.inputs();
    let refusal = || {
        let detail = format!("{} inputs, {N} wanted", inputs.len());
        Refusal::new("Inputs", step.name(), detail)
    };
    if inputs.len() != N {
        return Err(refusal());
    }
    let read: Vec<Source> = inputs
        .iter()
        .map(|input| Source::read(input.name, input.bytes))
        .collect::<Result<_, _>>()?;
    read.try_into().map_err(|_| refusal())
}

fn survey(step: &mut Step<'_>, source: &Source, clips: Option<&BTreeMap<String, String>>) {
    let document = &source.gltf;
    let primitives: usize = document
        .meshes()
        .map(|mesh| mesh.primitives().count())
        .sum();
    let images: Vec<Value> = document
        .images()
        .map(|image| json!({"index": image.index(), "name": image.name()}))
        .collect();
    let animations: Vec<Value> = document
        .animations()
        .map(|animation| {
            let seconds = animation
                .channels()
                .filter_map(|channel| channel.sampler().input().max())
                .filter_map(|max| max.get(0).and_then(Value::as_f64))
                .fold(0.0, f64::max);
            let name = animation.name().unwrap_or_default();
            json!({
                "name": name,
                "seconds": seconds,
                "channels": animation.channels().count(),
                "taken_as": clips.and_then(|clips| clips.get(name)),
            })
        })
        .collect();
    step.note(json!({
        "file": source.file,
        "meshes": document.meshes().count(),
        "primitives": primitives,
        "extensions_used": document.extensions_used().collect::<Vec<_>>(),
        "images": images,
        "animations": animations,
    }));
}

fn rig_notes(
    step: &mut Step<'_>,
    source: &Source,
    rig: &Rig,
    joints: usize,
) -> Result<(), Refusal> {
    let frame_gap = rig
        .frames
        .iter()
        .map(|frame| 2.0 * frame.xyz().length().asin())
        .fold(
            0.0,
            |widest: f32, gap| if gap > widest { gap } else { widest },
        );
    step.note(json!({"joints": rig.skeleton.len(), "bind_gap_m": rig.bind_gap, "frame_gap_rad": frame_gap}));
    budget(source, "BudgetJoints", rig.skeleton.len(), joints)
}

fn same_rig(owner: &Rig, form: &Rig, tolerance: f32) -> Result<(), String> {
    let (held, made) = (owner.skeleton.joints(), form.skeleton.joints());
    if held.len() != made.len() {
        return Err(format!(
            "{} joints, the full mesh has {}",
            made.len(),
            held.len()
        ));
    }
    for (at, (full, reduced)) in held.iter().zip(made).enumerate() {
        if full.name != reduced.name || full.parent != reduced.parent {
            return Err(format!(
                "joint {at} is `{}` with parent {:?}, the full mesh has `{}` with parent {:?}",
                reduced.name, reduced.parent, full.name, full.parent
            ));
        }
        if !full
            .inverse_bind
            .abs_diff_eq(reduced.inverse_bind, tolerance)
        {
            return Err(format!(
                "joint `{}` has another inverse bind matrix (tolerance {tolerance})",
                full.name
            ));
        }
    }
    Ok(())
}

struct ClipPlan<'a> {
    clips: &'a BTreeMap<String, String>,
    root_motion: Option<&'a str>,
    remove_travel: &'a [String],
    keys_per_second: f32,
    tolerance: f32,
}

fn write_clips(
    step: &mut Step<'_>,
    source: &Source,
    rig: &Rig,
    plan: &ClipPlan<'_>,
) -> Result<(), Refusal> {
    for (animation_name, clip_name) in plan.clips {
        let animation = source
            .gltf
            .animations()
            .find(|animation| animation.name() == Some(animation_name))
            .ok_or_else(|| {
                let at = format!("animation `{animation_name}`");
                source.refuse("AnimationMissing", at, "not in the file")
            })?;
        let settings = motion::Settings {
            root_motion: plan.root_motion,
            remove_travel: plan.remove_travel.contains(animation_name),
            keys_per_second: plan.keys_per_second,
            tolerance: plan.tolerance,
        };
        let made = motion::cook(source, rig, &animation, &settings)?;
        let bytes = made.clip.to_bytes();
        let file = format!("{clip_name}.clip");
        if ClipFile::from_bytes(&bytes).as_ref() != Ok(&made.clip) {
            return Err(source.refuse("ClipRoundTrip", file, "bytes read back differ"));
        }
        for drop in made.dropped {
            step.note(json!({"dropped": drop, "clip": clip_name}));
        }
        step.note(json!({
            "clip": clip_name,
            "seconds": made.clip.duration(),
            "keys": made.clip.times().len(),
            "rotation_tracks": made.clip.rotations().len(),
            "translation_tracks": made.clip.translations().iter().map(|track| track.bone.as_str()).collect::<Vec<_>>(),
            "root_track": made.clip.root().is_some(),
            "travel": made.travel,
        }));
        step.write(&file, bytes);
    }
    Ok(())
}

fn model(step: &mut Step<'_>) -> Result<(), Refusal> {
    let name = step.name().to_owned();
    check_name(&name)?;
    let params: Model = step.params()?;
    let [source] = sources(step)?;
    match &params {
        Model::Character(character) => {
            survey(step, &source, Some(&character.clips));
            run_character(step, &source, &name, character)
        }
        Model::Static(fixed) => {
            survey(step, &source, None);
            run_static(step, &source, &name, fixed)
        }
    }
}

fn run_character(
    step: &mut Step<'_>,
    source: &Source,
    name: &str,
    character: &Character,
) -> Result<(), Refusal> {
    check_scale(name, character.scale, character.tolerance)?;
    let travel = &character.remove_travel;
    check_clips(name, &character.clips, travel, character.keys_per_second)?;
    let sizes = (character.texture_size, &character.map_sizes);
    let kinds = map_kinds(name, character.maps.as_deref(), &character.textures, sizes)?;
    let rig = rig(source, character.scale, character.tolerance)?;
    rig_notes(step, source, &rig, character.budgets.joints)?;
    let bytes = rig.skeleton.to_bytes();
    let file = format!("{name}.skel");
    if Skeleton::from_bytes(&bytes).as_ref() != Ok(&rig.skeleton) {
        return Err(source.refuse("SkeletonRoundTrip", file, "bytes read back differ"));
    }
    step.write(&file, bytes);
    let meshes = geometry::character(source, &rig, name)?;
    let budgets = &character.budgets;
    let influences = counted(step, source, &meshes, (budgets.triangles, budgets.vertices))?;
    budget(source, "BudgetInfluences", influences, budgets.influences)?;
    let plan = MaterialPlan {
        names: &character.material,
        size: character.texture_size,
        sizes: &character.map_sizes,
        kinds,
        textures: &character.textures,
        material_sizes: None,
        exact_alpha: &[],
        shared: None,
    };
    let made = materials(step, source, name, &meshes, &plan)?;
    write_textures(step, source, &made.maps, &plan)?;
    write_materials(step, &made.materials);
    write_meshes(step, &meshes);
    let space = Mat4::from_scale(Vec3::splat(rig.scale)) * rig.space.inverse();
    let found = markers(source, space, |node| rig.joint_of_node(node));
    write_markers(step, name, &found);
    let plan = ClipPlan {
        clips: &character.clips,
        root_motion: character.root_motion.as_deref(),
        remove_travel: &character.remove_travel,
        keys_per_second: character.keys_per_second,
        tolerance: character.tolerance,
    };
    write_clips(step, source, &rig, &plan)
}

fn run_static(
    step: &mut Step<'_>,
    source: &Source,
    name: &str,
    fixed: &Static,
) -> Result<(), Refusal> {
    check_scale(name, fixed.scale, fixed.tolerance)?;
    let sizes = (fixed.texture_size, &fixed.map_sizes);
    let kinds = map_kinds(name, fixed.maps.as_deref(), &fixed.textures, sizes)?;
    if let Some(wanted) = fixed
        .exact_alpha
        .iter()
        .find(|wanted| kinds.iter().all(|kind| kind.name() != *wanted))
    {
        let detail = format!("`{wanted}` is not a map this entry cooks");
        return Err(Refusal::new("ExactAlpha", name, detail));
    }
    let meshes = geometry::fixed(source, name, fixed.scale, fixed.tolerance)?;
    let budgets = (fixed.budgets.triangles, fixed.budgets.vertices);
    counted(step, source, &meshes, budgets)?;
    let plan = MaterialPlan {
        names: &fixed.material,
        size: fixed.texture_size,
        sizes: &fixed.map_sizes,
        kinds,
        textures: &fixed.textures,
        material_sizes: None,
        exact_alpha: &fixed.exact_alpha,
        shared: fixed.shared_maps.as_ref(),
    };
    let made = materials(step, source, name, &meshes, &plan)?;
    write_textures(step, source, &made.maps, &plan)?;
    write_materials(step, &made.materials);
    write_meshes(step, &meshes);
    let found = markers(source, Mat4::from_scale(Vec3::splat(fixed.scale)), |_| None);
    write_markers(step, name, &found);
    Ok(())
}

fn form(step: &mut Step<'_>) -> Result<(), Refusal> {
    let name = step.name().to_owned();
    check_name(&name)?;
    let form: Form = step.params()?;
    check_scale(&name, form.scale, form.tolerance)?;
    let [source, owner] = sources(step)?;
    survey(step, &source, None);
    let full = rig(&owner, form.scale, form.tolerance)?;
    let made = rig(&source, form.scale, form.tolerance)?;
    same_rig(&full, &made, form.tolerance)
        .map_err(|detail| source.refuse("FormRig", &name, detail))?;
    let meshes = geometry::character(&source, &made, &name)?;
    let budgets = &form.budgets;
    let influences = counted(
        step,
        &source,
        &meshes,
        (budgets.triangles, budgets.vertices),
    )?;
    budget(&source, "BudgetInfluences", influences, budgets.influences)?;
    step.note(json!({"form_of": {"character": form.form_of, "joints": full.skeleton.len()}}));
    write_meshes(step, &meshes);
    Ok(())
}

fn clip(step: &mut Step<'_>) -> Result<(), Refusal> {
    let name = step.name().to_owned();
    check_name(&name)?;
    let clip: Clip = step.params()?;
    check_scale(&name, clip.scale, clip.tolerance)?;
    check_clips(
        &name,
        &clip.clips,
        &clip.remove_travel,
        clip.keys_per_second,
    )?;
    let (source, owner) = match &clip.rig_from {
        None => {
            let [source] = sources(step)?;
            (source, None)
        }
        Some(owner) => {
            let [source, held] = sources(step)?;
            (source, Some((owner, held)))
        }
    };
    survey(step, &source, Some(&clip.clips));
    let rig = match owner {
        None => rig(&source, clip.scale, clip.tolerance)?,
        Some((owner, held)) => {
            let from = rig(&held, clip.scale, clip.tolerance)?;
            let made = rig_by_names(&source, &from, clip.tolerance)?;
            same_rig(&from, &made, clip.tolerance).map_err(|detail| {
                let detail = format!("the clip file's skeleton differs from `{owner}`: {detail}");
                source.refuse("ClipRig", "rig_from", detail)
            })?;
            made
        }
    };
    rig_notes(step, &source, &rig, clip.budgets.joints)?;
    let plan = ClipPlan {
        clips: &clip.clips,
        root_motion: clip.root_motion.as_deref(),
        remove_travel: &clip.remove_travel,
        keys_per_second: clip.keys_per_second,
        tolerance: clip.tolerance,
    };
    write_clips(step, &source, &rig, &plan)
}
