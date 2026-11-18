use std::collections::BTreeMap;
use std::path::Path;

use engine_rig::{Clip, Skeleton};
use glam::{Mat4, Vec3};
use serde_json::{Value, json};

use engine_scene::{SceneMap, SceneMaterial, SceneSlot};
use engine_textures::Image;

use crate::config::{
    BakedPlan, Character, ClipEntry, Cook, Entry, FormEntry, MapEntry, MaterialNames, SetsPlan,
    TextureVariant, ValueMap, ValueRule,
};
use crate::encode;
use crate::fault::Fault;
use crate::fragment::{
    CookedMaterial, FragmentDrawn, FragmentEntity, LevelPiece, LevelSets, assemble, material,
    validate,
};
use crate::geometry::{self, Built};
use crate::maps::{self, MapKind};
use crate::markers;
use crate::mips;
use crate::motion;
use crate::rig::{Rig, rig, rig_by_names};
use crate::sets;
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
    let colored: BTreeMap<&str, usize> = meshes
        .iter()
        .filter(|mesh| mesh.colored)
        .map(|mesh| (mesh.stem.as_str(), mesh.vertices))
        .collect();
    report
        .counts
        .insert("color_stream_vertices", json!(colored));
    report.budget(file, "BudgetTriangles", triangles, budgets.0);
    report.budget(file, "BudgetVertices", vertices, budgets.1);
}

pub struct Baked<'a> {
    pub plans: &'a BTreeMap<String, BakedPlan>,
    pub value_maps: &'a BTreeMap<String, ValueMap>,
    pub value_rules: &'a BTreeMap<String, ValueRule>,
    pub pictures: Vec<(String, String, Image)>,
}

pub struct MaterialPlan<'a> {
    pub names: &'a MaterialNames,
    pub size: u32,
    pub sizes: &'a BTreeMap<String, u32>,
    pub kinds: Vec<MapKind>,
    pub textures: &'a [TextureVariant],
    pub material_sizes: Option<&'a BTreeMap<String, BTreeMap<String, u32>>>,
    pub baked: Option<Baked<'a>>,
    pub extra: Vec<FragmentEntity>,
    pub exact_alpha: &'a [String],
    pub map_stem: Option<&'a str>,
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
            material_sizes: None,
            extra: Vec::new(),
            exact_alpha: &[],
            map_stem: None,
            baked: None,
        }
    }

    fn size(&self, material: Option<&str>, kind: MapKind) -> u32 {
        material
            .and_then(|name| self.material_sizes?.get(name)?.get(kind.name()).copied())
            .or_else(|| self.sizes.get(kind.name()).copied())
            .unwrap_or(self.size)
    }
}

fn extras_values(source: &Source, index: usize) -> Result<BTreeMap<String, f32>, Fault> {
    let item = format!("materials[{index}].extras");
    match source.extras("materials", index) {
        Value::Null => Ok(BTreeMap::new()),
        Value::Object(extras) => extras
            .into_iter()
            .map(|(key, value)| match value.as_f64() {
                Some(number) if number.is_finite() => Ok((key, number as f32)),
                Some(_) | None => Err(source.fault(
                    "MaterialExtras",
                    item.clone(),
                    format!("`{key}` is {value}; material extras become numeric values"),
                )),
            })
            .collect(),
        other => Err(source.fault("MaterialExtras", item, format!("{other} is not an object"))),
    }
}

fn bake(
    source: &Source,
    (index, meshes): (usize, &[Built]),
    (material, slot): (&mut SceneMaterial, &str),
    plans: &BTreeMap<String, BakedPlan>,
) -> Result<(), Fault> {
    let item = format!("materials[{index}]");
    let held = |flag: fn(&Built) -> bool| {
        meshes
            .iter()
            .any(|mesh| mesh.material == index && flag(mesh))
    };
    let streams = [
        ("uv1", held(|mesh| mesh.uv1)),
        ("color", held(|mesh| mesh.colored)),
    ];
    for (stream, carried) in streams {
        let Some(plan) = plans.get(stream).filter(|_| carried) else {
            continue;
        };
        let values = material.values.get_or_insert_with(BTreeMap::new);
        for (key, value) in &plan.values {
            if values
                .insert(key.clone(), *value)
                .is_some_and(|held| held != *value)
            {
                return Err(source.fault(
                    "BakedValue",
                    item,
                    format!("`{key}` is set twice (material extras and `baked.{stream}`)"),
                ));
            }
        }
        let Some(SceneSlot::Filled(fill)) = material
            .slots
            .as_mut()
            .and_then(|slots| slots.get_mut(slot))
        else {
            return Err(source.fault("BakedMap", item, "the material has no filled slot"));
        };
        for (name, baked) in &plan.maps {
            let mut sampling = baked.sampling.clone();
            sampling.insert("id".into(), json!(baked.picture));
            let map: SceneMap = serde_json::from_value(Value::Object(sampling))
                .map_err(|error| source.fault("BakedMap", item.clone(), error))?;
            if fill.maps.insert(name.clone(), map).is_some() {
                return Err(source.fault("BakedMap", item, format!("map `{name}` is set twice")));
            }
        }
    }
    Ok(())
}

fn value_maps(
    source: &Source,
    index: usize,
    (material, slot): (&mut SceneMaterial, &str),
    rules: &BTreeMap<String, ValueMap>,
) -> Result<(), Fault> {
    let item = format!("materials[{index}]");
    for (name, rule) in rules {
        let held = material.values.clone().unwrap_or_default();
        let value = |key: &str| held.get(key).copied();
        let on: Vec<&str> = rule
            .when
            .iter()
            .map(String::as_str)
            .filter(|key| value(key).is_some_and(|number| number > 0.0))
            .collect();
        if on.is_empty() {
            continue;
        }
        let short: Vec<&str> = rule
            .needs
            .iter()
            .map(String::as_str)
            .filter(|key| !value(key).is_some_and(|number| number.is_finite() && number > 0.0))
            .collect();
        if !short.is_empty() {
            return Err(source.fault(
                "ValueMapNeeds",
                item,
                format!("{on:?} above 0 turn on map `{name}`, which also needs {short:?} above 0"),
            ));
        }
        let Some(SceneSlot::Filled(fill)) = material
            .slots
            .as_mut()
            .and_then(|slots| slots.get_mut(slot))
        else {
            return Err(source.fault(
                "ValueMap",
                item,
                format!("{on:?} above 0 need map `{name}` but the material has no filled slot"),
            ));
        };
        let mut sampling = rule.sampling.clone();
        sampling.insert("id".into(), json!(rule.picture));
        let map: SceneMap = serde_json::from_value(Value::Object(sampling))
            .map_err(|error| source.fault("ValueMap", item.clone(), error))?;
        if fill.maps.insert(name.clone(), map).is_some() {
            return Err(source.fault("ValueMap", item, format!("map `{name}` is set twice")));
        }
    }
    Ok(())
}

type MapKey = (usize, MapKind, u32, [u32; 2]);

pub struct Made<'a> {
    pub cooked: Vec<CookedMaterial>,
    pub scene: Vec<SceneMaterial>,
    pub pictures: Vec<(String, String)>,
    pub picture_images: Vec<(String, &'a Image)>,
    pub entities: Vec<FragmentEntity>,
}

fn entities(
    source: &Source,
    meshes: &[Built],
    mapping: &BTreeMap<&str, String>,
    plans: &BTreeMap<String, BakedPlan>,
) -> Result<Vec<FragmentEntity>, Fault> {
    meshes
        .iter()
        .map(|mesh| {
            let stream = if mesh.colored { "color" } else { "uv1" };
            let plan = plans.get(stream).ok_or_else(|| {
                source.fault(
                    "BakedKind",
                    format!("mesh {}", mesh.stem),
                    format!(
                        "the mesh carries {stream} but `baked.{stream}` names no kind to draw it"
                    ),
                )
            })?;
            Ok(FragmentEntity {
                id: mesh.stem.clone(),
                drawn: Some(FragmentDrawn {
                    kind: plan.kind.clone(),
                    mesh: mesh.stem.clone(),
                    material: mapping[mesh.stem.as_str()].clone(),
                }),
                level_piece: mesh.largest_m.map(|largest_m| LevelPiece { largest_m }),
                level_sets: None,
                transform: Some(BTreeMap::new()),
            })
        })
        .collect()
}

fn cook_one(
    (source, report): (&Source, &mut Report),
    (index, stem): (usize, &dyn Fn(usize) -> String),
    (meshes, plan): (&[Built], &MaterialPlan<'_>),
    shared: &mut BTreeMap<MapKey, String>,
) -> Option<(CookedMaterial, SceneMaterial)> {
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
    let mut maps = Vec::new();
    let mut borrowed = Vec::new();
    let plain = plan
        .kinds
        .iter()
        .all(|&kind| maps::texture_of(&held, kind).is_none());
    if plain {
        report.notes.push(json!({"note": "PlainMaterial", "file": source.file, "item": item, "detail": "the material names none of the cooked maps; it is written without maps"}));
    }
    for &kind in plan.kinds.iter().filter(|_| !plain) {
        let size = plan.size(held.name(), kind);
        let key = maps::key(&held, kind, size);
        if let Some(owner) = key.and_then(|key| shared.get(&key)) {
            let (texture, tex_coord) =
                maps::texture_of(&held, kind).expect("keyed_maps_have_textures");
            borrowed.push((
                kind,
                format!("{owner}_{}", kind.name()),
                maps::sampling(&texture, tex_coord),
            ));
            continue;
        }
        let map_stem = plan.map_stem.map_or_else(|| stem(index), str::to_owned);
        if let Some(map) = report.take(maps::map(source, &held, kind, &map_stem, size)) {
            if let Some(key) = key {
                shared.insert(key, map_stem);
            }
            maps.push(map);
        }
    }
    if !plain && maps.len() + borrowed.len() != plan.kinds.len() {
        return None;
    }
    let made = CookedMaterial {
        stem: stem(index),
        maps,
        borrowed,
    };
    let mut material = report.take(material(source, &held, &made, plan.names))?;
    let extras = report.take(extras_values(source, index))?;
    if !extras.is_empty() {
        let values = material.values.get_or_insert_with(BTreeMap::new);
        for (key, value) in extras {
            if let Some(held) = values.insert(key.clone(), value)
                && held.to_bits() != value.to_bits()
            {
                report.faults.push(source.fault(
                    "MaterialExtras",
                    format!("materials[{index}].extras.{key}"),
                    format!("{value} here, but the glTF material itself writes {held}"),
                ));
                return None;
            }
        }
    }
    if let Some(baked) = &plan.baked {
        report.take(bake(
            source,
            (index, meshes),
            (&mut material, &plan.names.slot),
            baked.plans,
        ))?;
        report.take(value_maps(
            source,
            index,
            (&mut material, &plan.names.slot),
            baked.value_maps,
        ))?;
        for (key, rule) in baked.value_rules {
            let held = material
                .values
                .as_ref()
                .and_then(|values| values.get(key))
                .copied();
            if let Some(value) = held
                && !rule.holds(value)
            {
                report.faults.push(source.fault(
                    "ValueRule",
                    format!("materials[{index}].values.{key}"),
                    format!("{value} breaks the rule {rule:?}"),
                ));
                return None;
            }
        }
    }
    Some((made, material))
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
    let mut cooked: Vec<CookedMaterial> = Vec::new();
    let mut scene = Vec::new();
    let mut shared = BTreeMap::new();
    for &index in &used {
        if let Some((made, material)) = cook_one(
            (source, report),
            (index, &stem),
            (meshes, plan),
            &mut shared,
        ) {
            scene.push(material);
            cooked.push(made);
        }
    }
    let named: Vec<&str> = used
        .iter()
        .filter_map(|&index| source.gltf.materials().nth(index)?.name())
        .collect();
    for wanted in plan.material_sizes.into_iter().flat_map(BTreeMap::keys) {
        if !named.contains(&wanted.as_str()) {
            report.faults.push(source.fault(
                "MaterialSizes",
                format!("material_sizes.{wanted}"),
                "no drawn material has this name",
            ));
        }
    }
    for variant in plan.textures {
        for wanted in variant.material_sizes.keys() {
            if !named.contains(&wanted.as_str()) {
                report.faults.push(source.fault(
                    "MaterialSizes",
                    format!("textures.{}.material_sizes.{wanted}", variant.name),
                    "no drawn material has this name",
                ));
            }
        }
    }
    let mapping: BTreeMap<&str, String> = meshes
        .iter()
        .map(|mesh| (mesh.stem.as_str(), stem(mesh.material)))
        .collect();
    report.counts.insert("mesh_materials", json!(mapping));
    report.counts.insert("materials", json!(used.len()));
    let borrowed: usize = cooked.iter().map(|material| material.borrowed.len()).sum();
    report.counts.insert("shared_maps", json!(borrowed));
    if !report.faults.is_empty() {
        return;
    }
    let pictures = plan.baked.iter().flat_map(|baked| &baked.pictures);
    let drawn = match &plan.baked {
        Some(baked) => match entities(source, meshes, &mapping, baked.plans) {
            Ok(mut drawn) => {
                drawn.extend(plan.extra.iter().cloned());
                drawn
            }
            Err(fault) => {
                report.faults.push(fault);
                return;
            }
        },
        None => Vec::new(),
    };
    let made = Made {
        cooked,
        scene,
        entities: drawn,
        pictures: pictures
            .clone()
            .map(|(id, uri, _)| (id.clone(), uri.clone()))
            .collect(),
        picture_images: pictures
            .map(|(_, uri, image)| (uri.clone(), image))
            .collect(),
    };
    if plan.textures.is_empty() {
        write_png(cook, source, report, (name, meshes), made);
    } else {
        write_variants(
            cook,
            source,
            report,
            (name, meshes),
            &made,
            (plan.textures, plan.exact_alpha),
        );
    }
}

fn write_png(
    cook: &Cook,
    source: &Source,
    report: &mut Report,
    (name, meshes): (&str, &[Built]),
    made: Made<'_>,
) {
    let fragment = assemble(
        (&cook.uri_prefix, ".png"),
        meshes,
        &made.cooked,
        (made.scene, &made.pictures, made.entities),
    );
    let images: Vec<(String, &Image)> = made
        .cooked
        .iter()
        .flat_map(|material| &material.maps)
        .map(|map| (format!("{}{}.png", cook.uri_prefix, map.stem), &map.decoded))
        .chain(made.picture_images)
        .collect();
    if let Some(text) = report.take(validate(
        (&source.file, name),
        &fragment,
        &cook.uri_prefix,
        meshes,
        &images,
    )) {
        push_meshes(report, meshes);
        for map in made.cooked.iter().flat_map(|material| &material.maps) {
            report
                .files
                .push((format!("{}.png", map.stem), map.png.clone()));
        }
        report
            .files
            .push((format!("{name}.fragment.json"), text.into_bytes()));
    }
}

fn write_variants(
    cook: &Cook,
    source: &Source,
    report: &mut Report,
    (name, meshes): (&str, &[Built]),
    made: &Made<'_>,
    (textures, exact_alpha): (&[TextureVariant], &[String]),
) {
    let (cooked, scene, pictures, picture_images) = (
        &made.cooked,
        &made.scene,
        &made.pictures,
        &made.picture_images,
    );
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
            let limit = variant
                .material_sizes
                .get(&map.material)
                .and_then(|sizes| sizes.get(map.kind.name()));
            let top = limit.map_or(0, |&limit| {
                chain
                    .iter()
                    .position(|level| level.width <= limit && level.height <= limit)
                    .unwrap_or(chain.len() - 1)
            });
            let chain = &chain[top..];
            let exact = exact_alpha.iter().any(|name| name == map.kind.name());
            let Some(encoded) = report.take(
                encode::encode(map.kind, encoding, chain, exact).map_err(|error| {
                    source.fault(
                        "TextureEncode",
                        format!("{} {}", map.stem, encoding.name()),
                        error,
                    )
                }),
            ) else {
                continue;
            };
            file_bytes += encoded.file.len();
            gpu_bytes += encoded.gpu_bytes;
            seconds += encoded.seconds;
            rows.push(json!({"map": map.stem, "encoding": encoding.name(), "levels": chain.len(), "file_bytes": encoded.file.len(), "gpu_bytes": encoded.gpu_bytes, "encode_seconds": encoded.seconds, "quality": encoded.quality}));
            written.push((format!("{}{ending}", map.stem), encoded.file));
            images.push((
                format!("{}{}{ending}", cook.uri_prefix, map.stem),
                encoded.level_zero,
            ));
        }
        if !report.faults.is_empty() {
            return;
        }
        let fragment = assemble(
            (&cook.uri_prefix, &ending),
            meshes,
            cooked,
            (scene.to_vec(), pictures, made.entities.clone()),
        );
        let held: Vec<(String, &Image)> = images
            .iter()
            .map(|(uri, image)| (uri.clone(), image))
            .chain(
                picture_images
                    .iter()
                    .map(|(uri, image)| (uri.clone(), *image)),
            )
            .collect();
        let Some(text) = report.take(validate(
            (&source.file, name),
            &fragment,
            &cook.uri_prefix,
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
    keys_per_second: f32,
    tolerance: f32,
}

fn cook_clips(source: &Source, report: &mut Report, rig: &Rig, plan: &ClipPlan<'_>) {
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
            keys_per_second: plan.keys_per_second,
            tolerance: plan.tolerance,
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
        source,
        report,
        &rig,
        &ClipPlan {
            clips: &character.clips,
            root_motion: character.root_motion.as_deref(),
            remove_travel: &character.remove_travel,
            keys_per_second: cook.keys_per_second,
            tolerance: cook.tolerance,
        },
    );
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

fn run_form(cook: &Cook, base: &Path, source: &Source, report: &mut Report, form: &FormEntry) {
    let owner = cook.entries.iter().find_map(|entry| match entry {
        Entry::Character(character) if character.name == form.form_of => Some(character),
        Entry::Character(_)
        | Entry::Form(_)
        | Entry::Clip(_)
        | Entry::Static(_)
        | Entry::Map(_)
        | Entry::Picture(_) => None,
    });
    let Some(owner) = owner else {
        report.faults.push(source.fault(
            "FormOf",
            format!("entries.{}.form_of", form.name),
            format!("no character entry `{}`", form.form_of),
        ));
        return;
    };
    let Some(full) = report.take(
        Source::load(&base.join(&owner.source))
            .and_then(|held| rig(&held, owner.scale, cook.tolerance)),
    ) else {
        return;
    };
    let Some(made) = report.take(rig(source, owner.scale, cook.tolerance)) else {
        return;
    };
    if let Err(detail) = same_rig(&full, &made, cook.tolerance) {
        report
            .faults
            .push(source.fault("FormRig", format!("entries.{}", form.name), detail));
        return;
    }
    let Some(meshes) = report.take(geometry::character(source, &made, &form.name)) else {
        return;
    };
    mesh_counts(
        &source.file,
        report,
        &meshes,
        (form.budgets.triangles, form.budgets.vertices),
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
        form.budgets.influences,
    );
    report.counts.insert(
        "form_of",
        json!({"character": owner.name, "joints": full.skeleton.len()}),
    );
    push_meshes(report, &meshes);
}

fn baked_pictures(
    cook: &Cook,
    base: &Path,
    map: &MapEntry,
) -> Result<Vec<(String, String, Image)>, Fault> {
    let mut pictures: Vec<(String, String, Image)> = Vec::new();
    for (_, wanted, variant) in map.pictures() {
        if pictures.iter().any(|(id, _, _)| *id == wanted) {
            continue;
        }
        let picture = cook
            .entries
            .iter()
            .find_map(|entry| match entry {
                Entry::Picture(picture) if picture.name == wanted => Some(picture),
                Entry::Character(_)
                | Entry::Form(_)
                | Entry::Clip(_)
                | Entry::Static(_)
                | Entry::Map(_)
                | Entry::Picture(_) => None,
            })
            .expect("baked_pictures_validated");
        let path = base.join(&picture.source);
        let file = path.display().to_string();
        let bytes = std::fs::read(&path)
            .map_err(|error| Fault::new("PictureUnreadable", &file, "file", error))?;
        let image = engine_textures::decode_image(&file, &bytes)
            .map_err(|error| Fault::new("PictureUndecodable", &file, "file", error))?;
        pictures.push((
            wanted.to_owned(),
            format!("{}{wanted}.{variant}.ktx2", cook.uri_prefix),
            image,
        ));
    }
    Ok(pictures)
}

struct SetsMade {
    entity: FragmentEntity,
    file: (String, Vec<u8>),
    key_file: (String, Vec<u8>),
    summary: Value,
}

fn cached_sets(
    folder: &Path,
    (stem, key_name): (&str, &str),
    key: &str,
) -> Option<(Vec<u8>, Value)> {
    let text = std::fs::read_to_string(folder.join(key_name)).ok()?;
    let held: Value = serde_json::from_str(&text).ok()?;
    if held["key"].as_str() != Some(key) {
        return None;
    }
    let bytes = std::fs::read(folder.join(stem)).ok()?;
    if held["summary"]["file_bytes"].as_u64() != Some(bytes.len() as u64) {
        return None;
    }
    Some((bytes, held["summary"].clone()))
}

fn visible_sets(
    cook: &Cook,
    base: &Path,
    source: &Source,
    (name, plan): (&str, &SetsPlan),
    meshes: &[Built],
) -> Result<SetsMade, Fault> {
    let path = base.join(&plan.data);
    let file = path.display().to_string();
    let fault = |detail: &dyn std::fmt::Display| {
        Fault::new(
            "VisibleSets",
            &file,
            format!("entries.{name}.visible_sets"),
            detail,
        )
    };
    let text = std::fs::read_to_string(&path).map_err(|error| fault(&error))?;
    let data: Value = serde_json::from_str(&text).map_err(|error| fault(&error))?;
    let materials: Vec<gltf::Material<'_>> = source.gltf.materials().collect();
    for wanted in &plan.see_through {
        if !meshes
            .iter()
            .any(|mesh| materials[mesh.material].name() == Some(wanted.as_str()))
        {
            return Err(fault(&format!(
                "see_through `{wanted}`: no drawn material has this name"
            )));
        }
    }
    let mut chunks = Vec::with_capacity(meshes.len());
    for mesh in meshes {
        let held = engine_assets::Mesh::from_record(&mesh.bytes).map_err(|error| fault(&error))?;
        let material = &materials[mesh.material];
        let see_through = plan
            .see_through
            .iter()
            .any(|wanted| material.name() == Some(wanted.as_str()));
        chunks.push(sets::Chunk {
            id: mesh.stem.clone(),
            positions: held
                .vertices()
                .iter()
                .map(|vertex| vertex.position)
                .collect(),
            indices: held.indices().to_vec(),
            occludes: !see_through && material.alpha_mode() == gltf::material::AlphaMode::Opaque,
            two_sided: material.double_sided(),
        });
    }
    let stem = format!("{name}.sets.bin");
    let key_name = format!("{name}.sets.key.json");
    let tool = std::env::current_exe()
        .and_then(std::fs::read)
        .map_err(|error| fault(&error))?;
    let mut described = serde_json::to_value(plan).map_err(|error| fault(&error))?;
    if let Some(fields) = described.as_object_mut() {
        fields.remove("data");
    }
    let key = sets::key(&described, &data, &chunks, &tool);
    let folder = base.join(&cook.output);
    let (bytes, summary, cached) = match cached_sets(&folder, (&stem, &key_name), &key) {
        Some((bytes, summary)) => (bytes, summary, true),
        None => {
            let built = sets::build(plan, &data, &chunks).map_err(|detail| fault(&detail))?;
            (built.bytes, built.summary, false)
        }
    };
    let key_text = serde_json::to_string_pretty(&json!({"key": key, "summary": summary}))
        .map_err(|error| fault(&error))?;
    let mut summary = summary;
    summary["cached"] = json!(cached);
    summary["key"] = json!(key);
    let entity = FragmentEntity {
        id: format!("{name}_sets"),
        drawn: None,
        level_piece: None,
        level_sets: Some(LevelSets {
            file: format!("{}{stem}", cook.uri_prefix),
        }),
        transform: None,
    };
    Ok(SetsMade {
        entity,
        file: (stem, bytes),
        key_file: (key_name, key_text.into_bytes()),
        summary,
    })
}

fn run_map(cook: &Cook, base: &Path, source: &Source, report: &mut Report, map: &MapEntry) {
    let prefixes: Vec<&str> = map.merge.iter().map(|rule| rule.prefix.as_str()).collect();
    if let Some(drawn) = report.take(geometry::map_drawn(
        source,
        &map.name,
        map.scale,
        cook.tolerance,
        (&prefixes, map.chunks.as_ref()),
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
        let mut materials: Vec<usize> = drawn.built.iter().map(|mesh| mesh.material).collect();
        materials.sort_unstable();
        materials.dedup();
        report.budget(
            &source.file,
            "BudgetMaterials",
            materials.len(),
            map.budgets.materials,
        );
        if let Some(chunks) = &drawn.chunks {
            report.counts.insert("chunks", chunks.clone());
        }
        let mut plan = MaterialPlan::new(
            &map.material,
            (map.texture_size, &map.map_sizes),
            map.maps.as_deref(),
            &map.textures,
        );
        plan.material_sizes = Some(&map.material_sizes);
        if let Some(sets) = &map.visible_sets {
            let Some(made) = report.take(visible_sets(
                cook,
                base,
                source,
                (&map.name, sets),
                &drawn.built,
            )) else {
                return;
            };
            report.counts.insert("visible_sets", made.summary);
            report.files.push(made.file);
            report.files.push(made.key_file);
            plan.extra.push(made.entity);
        }
        if !map.baked.is_empty() || !map.value_maps.is_empty() || !map.value_rules.is_empty() {
            let Some(pictures) = report.take(baked_pictures(cook, base, map)) else {
                return;
            };
            plan.baked = Some(Baked {
                plans: &map.baked,
                value_maps: &map.value_maps,
                value_rules: &map.value_rules,
                pictures,
            });
        }
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
            | Entry::Form(_)
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
    let made = rig_by_names(source, &from, cook.tolerance)?;
    same_rig(&from, &made, cook.tolerance).map_err(|detail| {
        source.fault(
            "ClipRig",
            format!("entries.{}.rig_from", clip.name),
            format!("the clip file's skeleton differs from `{name}`: {detail}"),
        )
    })?;
    Ok(made)
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
                root_motion: clip.root_motion.as_deref(),
                remove_travel: &clip.remove_travel,
                keys_per_second: clip.keys_per_second.unwrap_or(cook.keys_per_second),
                tolerance: clip.tolerance.unwrap_or(cook.tolerance),
            };
            cook_clips(&source, &mut report, &rig, &plan);
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
                let mut plan = MaterialPlan::new(
                    &fixed.material,
                    (fixed.texture_size, &fixed.map_sizes),
                    fixed.maps.as_deref(),
                    &fixed.textures,
                );
                plan.exact_alpha = &fixed.exact_alpha;
                plan.map_stem = fixed.map_stem.as_deref();
                for wanted in &fixed.exact_alpha {
                    if !plan.kinds.iter().any(|kind| kind.name() == wanted) {
                        report.faults.push(source.fault(
                            "ExactAlpha",
                            format!("entries.{}.exact_alpha", fixed.name),
                            format!("`{wanted}` is not a map this entry cooks"),
                        ));
                    }
                }
                cook_materials(cook, &source, &mut report, &fixed.name, &meshes, &plan);
                let found =
                    markers::markers(&source, Mat4::from_scale(Vec3::splat(fixed.scale)), |_| {
                        None
                    });
                push_markers(&mut report, &fixed.name, &found);
            }
        }
        Entry::Map(map) => run_map(cook, base, &source, &mut report, map),
        Entry::Form(form) => run_form(cook, base, &source, &mut report, form),
        Entry::Picture(_) => {}
    }
    report
}
