use std::collections::BTreeMap;
use std::path::Path;

use engine_rig::{Clip, Skeleton};
use glam::{Mat4, Vec3};
use serde_json::{Value, json};

use engine_scene::SceneMaterial;
use engine_textures::Image;

use crate::config::{Character, ClipEntry, Cook, Entry, MapEntry, MaterialNames, TextureVariant};
use crate::encode;
use crate::fault::Fault;
use crate::fragment::{CookedMaterial, assemble, material, validate};
use crate::geometry::{self, Built};
use crate::maps::{self, MapKind};
use crate::markers;
use crate::mips;
use crate::motion;
use crate::rig::{Rig, rig, rig_by_names};
use crate::source::Source;

#[derive(Default)]
pub struct Report {
    pub counts: BTreeMap<&'static str, Value>,
    pub notes: Vec<Value>,
    pub faults: Vec<Fault>,
    pub files: Vec<(String, Vec<u8>)>,
}

impl Report {
    fn take<T>(&mut self, result: Result<T, Fault>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(fault) => {
                self.faults.push(fault);
                None
            }
        }
    }

    fn budget(&mut self, file: &str, name: &'static str, count: usize, budget: usize) {
        if count > budget {
            self.faults.push(Fault::new(
                name,
                file,
                "budgets",
                format!("{count} > {budget}"),
            ));
        }
    }
}

fn survey(source: &Source, report: &mut Report, clips: Option<&BTreeMap<String, String>>) {
    let document = &source.gltf;
    report
        .counts
        .insert("meshes", json!(document.meshes().count()));
    report.counts.insert(
        "primitives",
        json!(
            document
                .meshes()
                .map(|mesh| mesh.primitives().count())
                .sum::<usize>()
        ),
    );
    report.counts.insert(
        "extensions_used",
        json!(document.extensions_used().collect::<Vec<_>>()),
    );
    let images: Vec<Value> = document
        .images()
        .map(|image| match maps::decode(source, &image) {
            Ok(decoded) => json!({"index": image.index(), "name": image.name(), "width": decoded.width, "height": decoded.height}),
            Err(fault) => json!({"index": image.index(), "name": image.name(), "error": fault.name}),
        })
        .collect();
    report.counts.insert("images", json!(images));
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
    report.counts.insert("animations", json!(animations));
}

fn mesh_counts(file: &str, report: &mut Report, meshes: &[Built], budgets: (usize, usize)) {
    let vertices: usize = meshes.iter().map(|mesh| mesh.vertices).sum();
    let triangles: usize = meshes.iter().map(|mesh| mesh.triangles).sum();
    let influences = meshes
        .iter()
        .map(|mesh| mesh.influences)
        .max()
        .unwrap_or_default();
    report.counts.insert("vertices", json!(vertices));
    report.counts.insert("triangles", json!(triangles));
    report.counts.insert("max_influences", json!(influences));
    report.budget(file, "BudgetTriangles", triangles, budgets.0);
    report.budget(file, "BudgetVertices", vertices, budgets.1);
}

pub struct MaterialPlan<'a> {
    pub names: &'a MaterialNames,
    pub size: u32,
    pub sizes: &'a BTreeMap<String, u32>,
    pub kinds: Vec<MapKind>,
    pub textures: &'a [TextureVariant],
}

impl<'a> MaterialPlan<'a> {
    fn new(
        names: &'a MaterialNames,
        (size, sizes): (u32, &'a BTreeMap<String, u32>),
        maps: Option<&[String]>,
        textures: &'a [TextureVariant],
    ) -> Self {
        let kinds = match maps {
            None => MapKind::ALL.to_vec(),
            Some(maps) => MapKind::ALL
                .into_iter()
                .filter(|kind| maps.iter().any(|map| map == kind.name()))
                .collect(),
        };
        Self {
            names,
            size,
            sizes,
            kinds,
            textures,
        }
    }

    fn size(&self, kind: MapKind) -> u32 {
        self.sizes.get(kind.name()).copied().unwrap_or(self.size)
    }
}

fn cook_materials(
    cook: &Cook,
    source: &Source,
    report: &mut Report,
    name: &str,
    meshes: &[Built],
    plan: &MaterialPlan<'_>,
) {
    let mut used: Vec<usize> = meshes.iter().map(|mesh| mesh.material).collect();
    used.sort_unstable();
    used.dedup();
    let stem = |index: usize| {
        if used.len() == 1 {
            name.to_owned()
        } else {
            format!("{name}_{index}")
        }
    };
    let mut cooked = Vec::new();
    let mut scene = Vec::new();
    for &index in &used {
        let held = source
            .gltf
            .materials()
            .nth(index)
            .expect("material_index_from_document");
        let item = format!("materials[{index}]");
        if held.occlusion_texture().is_some() {
            report.notes.push(json!({"note": "OcclusionIgnored", "file": source.file, "item": item, "detail": "ORM red is written as 1"}));
        }
        if held.double_sided() {
            report.notes.push(json!({"warning": "DoubleSided", "file": source.file, "item": item, "detail": "doubleSided true: cull none"}));
        }
        let maps: Vec<_> = plan
            .kinds
            .iter()
            .filter_map(|&kind| {
                report.take(maps::map(source, &held, kind, &stem(index), plan.size(kind)))
            })
            .collect();
        if maps.len() != plan.kinds.len() {
            continue;
        }
        let made = CookedMaterial {
            stem: stem(index),
            maps,
        };
        if let Some(material) = report.take(material(source, &held, &made, plan.names)) {
            scene.push(material);
            cooked.push(made);
        }
    }
    let mapping: BTreeMap<&str, String> = meshes
        .iter()
        .map(|mesh| (mesh.stem.as_str(), stem(mesh.material)))
        .collect();
    report.counts.insert("mesh_materials", json!(mapping));
    report.counts.insert("materials", json!(used.len()));
    if !report.faults.is_empty() {
        return;
    }
    if plan.textures.is_empty() {
        let fragment = assemble((&cook.uri_prefix, ".png"), meshes, &cooked, scene);
        let images: Vec<(&str, &Image)> = cooked
            .iter()
            .flat_map(|material| &material.maps)
            .map(|map| (map.stem.as_str(), &map.decoded))
            .collect();
        if let Some(text) = report.take(validate(
            (&source.file, name),
            &fragment,
            (&cook.uri_prefix, ".png"),
            meshes,
            &images,
        )) {
            push_meshes(report, meshes);
            for map in cooked.into_iter().flat_map(|material| material.maps) {
                report.files.push((format!("{}.png", map.stem), map.png));
            }
            report
                .files
                .push((format!("{name}.fragment.json"), text.into_bytes()));
        }
        return;
    }
    write_variants(
        cook,
        source,
        report,
        (name, meshes),
        (&cooked, &scene),
        plan.textures,
    );
}

fn write_variants(
    cook: &Cook,
    source: &Source,
    report: &mut Report,
    (name, meshes): (&str, &[Built]),
    (cooked, scene): (&[CookedMaterial], &[SceneMaterial]),
    textures: &[TextureVariant],
) {
    let mut variants = Vec::new();
    let mut written = Vec::new();
    for variant in textures {
        let ending = format!(".{}.ktx2", variant.name);
        let mut images = Vec::new();
        let mut rows = Vec::new();
        let (mut file_bytes, mut gpu_bytes, mut seconds) = (0, 0, 0.0);
        for map in cooked.iter().flat_map(|material| &material.maps) {
            let encoding = variant
                .encoding(map.kind.name())
                .expect("variant_maps_validated");
            if encoding.srgb() != (map.kind == MapKind::BaseColor) {
                report.notes.push(json!({"warning": "TextureSpace", "file": source.file, "item": map.stem, "detail": format!("variant `{}` stores {} as {}", variant.name, map.kind.name(), encoding.name())}));
            }
            let chain = mips::chain(
                map.kind,
                &map.decoded,
                [map.sampling.wrap_u, map.sampling.wrap_v],
            );
            let Some(encoded) =
                report.take(encode::encode(map.kind, encoding, &chain).map_err(|error| {
                    source.fault(
                        "TextureEncode",
                        format!("{} {}", map.stem, encoding.name()),
                        error,
                    )
                }))
            else {
                continue;
            };
            file_bytes += encoded.file.len();
            gpu_bytes += encoded.gpu_bytes;
            seconds += encoded.seconds;
            rows.push(json!({"map": map.stem, "encoding": encoding.name(), "levels": chain.len(), "file_bytes": encoded.file.len(), "gpu_bytes": encoded.gpu_bytes, "encode_seconds": encoded.seconds, "quality": encoded.quality}));
            written.push((format!("{}{ending}", map.stem), encoded.file));
            images.push((map.stem.clone(), encoded.level_zero));
        }
        if !report.faults.is_empty() {
            return;
        }
        let fragment = assemble((&cook.uri_prefix, &ending), meshes, cooked, scene.to_vec());
        let held: Vec<(&str, &Image)> = images
            .iter()
            .map(|(stem, image)| (stem.as_str(), image))
            .collect();
        let Some(text) = report.take(validate(
            (&source.file, name),
            &fragment,
            (&cook.uri_prefix, &ending),
            meshes,
            &held,
        )) else {
            return;
        };
        written.push((
            format!("{name}.{}.fragment.json", variant.name),
            text.into_bytes(),
        ));
        variants.push(json!({"variant": variant.name, "file_bytes": file_bytes, "gpu_bytes": gpu_bytes, "encode_seconds": seconds, "maps": rows}));
    }
    report.counts.insert("textures", json!(variants));
    push_meshes(report, meshes);
    report.files.extend(written);
}

fn push_meshes(report: &mut Report, meshes: &[Built]) {
    for mesh in meshes {
        report
            .files
            .push((format!("{}.emb", mesh.stem), mesh.bytes.clone()));
    }
}

struct ClipPlan<'a> {
    clips: &'a BTreeMap<String, String>,
    root_motion: Option<&'a str>,
    remove_travel: &'a [String],
}

fn cook_clips(cook: &Cook, source: &Source, report: &mut Report, rig: &Rig, plan: &ClipPlan<'_>) {
    let mut cooked = Vec::new();
    for (animation_name, clip_name) in plan.clips {
        let Some(animation) = source
            .gltf
            .animations()
            .find(|animation| animation.name() == Some(animation_name))
        else {
            report.faults.push(source.fault(
                "AnimationMissing",
                format!("animation `{animation_name}`"),
                "not in the file",
            ));
            continue;
        };
        let settings = motion::Settings {
            root_motion: plan.root_motion,
            remove_travel: plan.remove_travel.contains(animation_name),
            keys_per_second: cook.keys_per_second,
            tolerance: cook.tolerance,
        };
        let Some(made) = report.take(motion::cook(source, rig, &animation, &settings)) else {
            continue;
        };
        let bytes = made.clip.to_bytes();
        if Clip::from_bytes(&bytes).as_ref() != Ok(&made.clip) {
            report.faults.push(source.fault(
                "ClipRoundTrip",
                format!("{clip_name}.clip"),
                "bytes read back differ",
            ));
        }
        for drop in made.dropped {
            report
                .notes
                .push(json!({"dropped": drop, "clip": clip_name}));
        }
        cooked.push(json!({
            "clip": clip_name,
            "seconds": made.clip.duration(),
            "keys": made.clip.times().len(),
            "rotation_tracks": made.clip.rotations().len(),
            "translation_tracks": made.clip.translations().iter().map(|track| track.bone.as_str()).collect::<Vec<_>>(),
            "root_track": made.clip.root().is_some(),
            "travel": made.travel,
        }));
        report.files.push((format!("{clip_name}.clip"), bytes));
    }
    report.counts.insert("clips", json!(cooked));
}

fn rig_counts(
    source: &Source,
    report: &mut Report,
    rig: &Rig,
    name: &str,
    budget: usize,
    write: bool,
) {
    report.counts.insert("joints", json!(rig.skeleton.len()));
    report.counts.insert("bind_gap_m", json!(rig.bind_gap));
    let frame_gap = rig
        .frames
        .iter()
        .map(|frame| 2.0 * frame.xyz().length().asin())
        .fold(
            0.0,
            |widest: f32, gap| if gap > widest { gap } else { widest },
        );
    report.counts.insert("frame_gap_rad", json!(frame_gap));
    report.budget(&source.file, "BudgetJoints", rig.skeleton.len(), budget);
    if write {
        let bytes = rig.skeleton.to_bytes();
        if Skeleton::from_bytes(&bytes).as_ref() != Ok(&rig.skeleton) {
            report.faults.push(source.fault(
                "SkeletonRoundTrip",
                format!("{name}.skel"),
                "bytes read back differ",
            ));
        }
        report.files.push((format!("{name}.skel"), bytes));
    }
}

fn push_markers(report: &mut Report, name: &str, markers: &Value) {
    report.counts.insert(
        "markers",
        json!(markers["markers"].as_array().map_or(0, Vec::len)),
    );
    report.files.push((
        format!("{name}.markers.json"),
        serde_json::to_string_pretty(markers)
            .expect("markers_serialize")
            .into_bytes(),
    ));
}

fn run_character(cook: &Cook, source: &Source, report: &mut Report, character: &Character) {
    let Some(rig) = report.take(rig(source, character.scale, cook.tolerance)) else {
        return;
    };
    rig_counts(
        source,
        report,
        &rig,
        &character.name,
        character.budgets.joints,
        true,
    );
    if let Some(meshes) = report.take(geometry::character(source, &rig, &character.name)) {
        mesh_counts(
            &source.file,
            report,
            &meshes,
            (character.budgets.triangles, character.budgets.vertices),
        );
        let influences = meshes
            .iter()
            .map(|mesh| mesh.influences)
            .max()
            .unwrap_or_default();
        report.budget(
            &source.file,
            "BudgetInfluences",
            influences,
            character.budgets.influences,
        );
        let plan = MaterialPlan::new(
            &character.material,
            (character.texture_size, &character.map_sizes),
            character.maps.as_deref(),
            &character.textures,
        );
        cook_materials(cook, source, report, &character.name, &meshes, &plan);
    }
    let space = Mat4::from_scale(Vec3::splat(rig.scale)) * rig.space.inverse();
    let found = markers::markers(source, space, |node| rig.joint_of_node(node));
    push_markers(report, &character.name, &found);
    cook_clips(
        cook,
        source,
        report,
        &rig,
        &ClipPlan {
            clips: &character.clips,
            root_motion: character.root_motion.as_deref(),
            remove_travel: &character.remove_travel,
        },
    );
}

fn run_map(cook: &Cook, source: &Source, report: &mut Report, map: &MapEntry) {
    let prefixes: Vec<&str> = map.merge.iter().map(|rule| rule.prefix.as_str()).collect();
    if let Some(drawn) = report.take(geometry::map_drawn(
        source,
        &map.name,
        map.scale,
        cook.tolerance,
        &prefixes,
    )) {
        mesh_counts(
            &source.file,
            report,
            &drawn.built,
            (map.budgets.triangles, map.budgets.vertices),
        );
        let (widest, vertices) = drawn.widest_object;
        report.counts.insert(
            "widest_object",
            json!({"node": widest, "vertices": vertices}),
        );
        report.budget(
            &source.file,
            "BudgetObjectVertices",
            vertices,
            map.budgets.object_vertices,
        );
        report.budget(
            &source.file,
            "BudgetMaterials",
            drawn.built.len(),
            map.budgets.materials,
        );
        let plan = MaterialPlan::new(
            &map.material,
            (map.texture_size, &map.map_sizes),
            map.maps.as_deref(),
            &map.textures,
        );
        cook_materials(cook, source, report, &map.name, &drawn.built, &plan);
    }
    let mut merged_counts = Vec::new();
    for rule in &map.merge {
        let Some(merged) = report.take(geometry::map_merged(
            source,
            &rule.prefix,
            &rule.output,
            map.scale,
            cook.tolerance,
        )) else {
            continue;
        };
        report.budget(
            &source.file,
            "BudgetMergedTriangles",
            merged.built.triangles,
            rule.triangles,
        );
        merged_counts.push(json!({
            "output": merged.output,
            "meshes": merged.meshes.len(),
            "vertices": merged.built.vertices,
            "triangles": merged.built.triangles,
        }));
        let listing = json!({"meshes": merged.meshes});
        report
            .files
            .push((format!("{}.emb", merged.output), merged.built.bytes));
        report.files.push((
            format!("{}.json", merged.output),
            serde_json::to_string_pretty(&listing)
                .expect("merge_listing_serializes")
                .into_bytes(),
        ));
    }
    report.counts.insert("merged", json!(merged_counts));
    let found = markers::markers(source, Mat4::from_scale(Vec3::splat(map.scale)), |_| None);
    push_markers(report, &map.name, &found);
}

fn clip_rig(cook: &Cook, base: &Path, source: &Source, clip: &ClipEntry) -> Result<Rig, Fault> {
    let Some(name) = &clip.rig_from else {
        return rig(source, 1.0, cook.tolerance);
    };
    let character = cook
        .entries
        .iter()
        .find_map(|entry| match entry {
            Entry::Character(character) if character.name == *name => Some(character),
            Entry::Character(_)
            | Entry::Clip(_)
            | Entry::Static(_)
            | Entry::Map(_)
            | Entry::Picture(_) => None,
        })
        .ok_or_else(|| {
            source.fault(
                "RigFrom",
                format!("entries.{}.rig_from", clip.name),
                format!("no character entry `{name}`"),
            )
        })?;
    let owner = Source::load(&base.join(&character.source))?;
    let from = rig(&owner, character.scale, cook.tolerance)?;
    rig_by_names(source, &from, cook.tolerance)
}

pub fn run(cook: &Cook, base: &Path, entry: &Entry, path: &Path) -> Report {
    if let Entry::Picture(picture) = entry {
        return crate::picture::run(picture, path);
    }
    let mut report = Report::default();
    let Some(source) = report.take(Source::load(path)) else {
        return report;
    };
    survey(&source, &mut report, entry.clips());
    match entry {
        Entry::Character(character) => run_character(cook, &source, &mut report, character),
        Entry::Clip(clip) => {
            let Some(rig) = report.take(clip_rig(cook, base, &source, clip)) else {
                return report;
            };
            rig_counts(
                &source,
                &mut report,
                &rig,
                &clip.name,
                clip.budgets.joints,
                false,
            );
            let plan = ClipPlan {
                clips: &clip.clips,
                root_motion: Some(&clip.root_motion),
                remove_travel: &clip.remove_travel,
            };
            cook_clips(cook, &source, &mut report, &rig, &plan);
        }
        Entry::Static(fixed) => {
            if let Some(meshes) = report.take(geometry::fixed(
                &source,
                &fixed.name,
                fixed.scale,
                cook.tolerance,
            )) {
                mesh_counts(
                    &source.file,
                    &mut report,
                    &meshes,
                    (fixed.budgets.triangles, fixed.budgets.vertices),
                );
                let plan = MaterialPlan::new(
                    &fixed.material,
                    (fixed.texture_size, &fixed.map_sizes),
                    fixed.maps.as_deref(),
                    &fixed.textures,
                );
                cook_materials(cook, &source, &mut report, &fixed.name, &meshes, &plan);
            }
        }
        Entry::Map(map) => run_map(cook, &source, &mut report, map),
        Entry::Picture(_) => {}
    }
    report
}
