use std::collections::BTreeMap;
use std::sync::Arc;

use cook::{Kind, Refusal, Step};
use cook_model::{
    Built, MapKind, MapPlan, Maps, Material, MaterialNames, MaterialPlan, Source, TextureVariant,
    budget, counted, identifier, item, map_kinds, markers, materials, write_markers,
    write_materials, write_meshes, write_textures,
};
use engine::scene::SceneMap;
use engine_core::MeshFile;
use engine_meshpack::POSITION;
use glam::{Mat4, Vec3};
use gltf::material::AlphaMode;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::chunks::ChunkPlan;
use crate::meshes::{self, Drawn};
use crate::sets::{Occluder, occluders_file};

pub static LEVEL: Kind = Kind {
    name: "level",
    version: 2,
    run: level,
};

const BAKED_STREAMS: [&str; 2] = ["lightmap", "color"];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Level {
    scale: f32,
    texture_size: u32,
    #[serde(default)]
    map_sizes: BTreeMap<String, u32>,
    material: MaterialNames,
    maps: Option<Vec<String>>,
    #[serde(default)]
    textures: Vec<TextureVariant>,
    #[serde(default)]
    material_sizes: BTreeMap<String, BTreeMap<String, u32>>,
    #[serde(default)]
    baked: BTreeMap<String, Baked>,
    #[serde(default)]
    value_maps: BTreeMap<String, ValueMap>,
    #[serde(default)]
    value_rules: BTreeMap<String, ValueRule>,
    merge: Vec<MergeRule>,
    chunks: Option<ChunkPlan>,
    budgets: LevelBudgets,
    tolerance: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Baked {
    kind: String,
    #[serde(default)]
    maps: BTreeMap<String, Picked>,
    #[serde(default)]
    values: BTreeMap<String, f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Picked {
    picture: String,
    sampling: Map<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ValueMap {
    picture: String,
    sampling: Map<String, Value>,
    when: Vec<String>,
    #[serde(default)]
    needs: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ValueRule {
    one_of: Option<Vec<f32>>,
    min: Option<f32>,
    max: Option<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MergeRule {
    prefix: String,
    output: String,
    triangles: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LevelBudgets {
    triangles: usize,
    vertices: usize,
    materials: usize,
    object_vertices: usize,
}

#[derive(Serialize)]
struct LevelPiece {
    largest_m: f32,
}

impl ValueRule {
    fn holds(&self, value: f32) -> bool {
        value.is_finite()
            && self
                .one_of
                .as_ref()
                .is_none_or(|allowed| allowed.contains(&value))
            && self.min.is_none_or(|least| value >= least)
            && self.max.is_none_or(|most| value <= most)
    }
}

fn positive(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

fn checked(name: &str, level: &Level) -> Result<Vec<MapKind>, Refusal> {
    let refuse = |code, detail: String| Err(Refusal::new(code, name, detail));
    if !identifier(name) {
        return refuse("EntryName", "not a lowercase identifier".to_owned());
    }
    if !positive(level.scale) {
        return refuse("Scale", level.scale.to_string());
    }
    if !(level.tolerance.is_finite() && level.tolerance >= 0.0) {
        return refuse("Tolerance", level.tolerance.to_string());
    }
    if let Some(rule) = level
        .merge
        .iter()
        .find(|rule| !identifier(&rule.output) || rule.prefix.is_empty())
    {
        let detail = format!("prefix `{}`, output `{}`", rule.prefix, rule.output);
        return refuse("MergeRule", detail);
    }
    if let Some(plan) = &level.chunks {
        if !positive(plan.cell_m) {
            return refuse("ChunkCell", plan.cell_m.to_string());
        }
        if let Some(bounds) = &plan.detail_classes_m
            && (bounds.is_empty()
                || !bounds.iter().copied().all(positive)
                || bounds.windows(2).any(|pair| pair[0] >= pair[1]))
        {
            let detail = format!("{bounds:?}: a non-empty list of rising positive sizes");
            return refuse("DetailClasses", detail);
        }
    }
    let cooked = |map: &str| MapKind::ALL.iter().any(|kind| kind.name() == map);
    for (stream, plan) in &level.baked {
        if !BAKED_STREAMS.contains(&stream.as_str()) {
            return refuse("BakedStream", format!("`{stream}`: one of {BAKED_STREAMS:?}"));
        }
        if let Some(map) = plan
            .maps
            .iter()
            .find(|(map, held)| cooked(map.as_str()) || held.sampling.contains_key("id"))
            .map(|(map, _)| map)
        {
            let detail = format!("baked.{stream}.maps.{map}: the map name must not be a cooked map and the sampling must not name an id");
            return refuse("BakedMap", detail);
        }
    }
    for (map, held) in &level.value_maps {
        let baked = level.baked.values().any(|plan| plan.maps.contains_key(map));
        if baked || cooked(map.as_str()) || held.sampling.contains_key("id") || held.when.is_empty() {
            let detail = format!("value_maps.{map}: the map name must not be a cooked or baked map; the sampling must not name an id; `when` names at least one value");
            return refuse("ValueMap", detail);
        }
    }
    let sizes = level
        .map_sizes
        .iter()
        .chain(level.material_sizes.values().flatten());
    map_kinds(name, level.maps.as_deref(), &level.textures, (level.texture_size, sizes))
}

fn put_map(
    source: &Source,
    material: &mut Material,
    (code, slot, name): (&'static str, &str, &str),
    (picture, sampling): (&str, &Map<String, Value>),
) -> Result<(), Refusal> {
    let at = format!("materials[{}]", material.index);
    let Some(fill) = material.slots.get_mut(slot) else {
        return Err(source.refuse(code, at, "the material has no filled slot"));
    };
    let mut sampling = sampling.clone();
    sampling.insert("id".into(), json!(picture));
    let map: SceneMap = serde_json::from_value(Value::Object(sampling))
        .map_err(|error| source.refuse(code, at.clone(), error))?;
    if fill.maps.insert(name.to_owned(), item(&map)).is_some() {
        return Err(source.refuse(code, at, format!("map `{name}` is set twice")));
    }
    Ok(())
}

fn baked(
    source: &Source,
    material: &mut Material,
    meshes: &[Built],
    level: &Level,
) -> Result<(), Refusal> {
    let (index, slot) = (material.index, level.material.slot.as_str());
    let at = format!("materials[{index}]");
    let held = |flag: fn(&Built) -> bool| {
        meshes
            .iter()
            .any(|mesh| mesh.material == index && flag(mesh))
    };
    let streams = [
        ("lightmap", held(|mesh| mesh.uv1)),
        ("color", held(|mesh| mesh.colored)),
    ];
    for (stream, carried) in streams {
        let Some(plan) = level.baked.get(stream).filter(|_| carried) else {
            continue;
        };
        let values = material.values.get_or_insert_with(BTreeMap::new);
        for (key, value) in &plan.values {
            if values
                .insert(key.clone(), *value)
                .is_some_and(|held| held != *value)
            {
                let detail = format!("`{key}` is set twice (material extras and `baked.{stream}`)");
                return Err(source.refuse("BakedValue", at, detail));
            }
        }
        for (name, map) in &plan.maps {
            let picked = (map.picture.as_str(), &map.sampling);
            put_map(source, material, ("BakedMap", slot, name), picked)?;
        }
    }
    for (name, rule) in &level.value_maps {
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
            let detail =
                format!("{on:?} above 0 turn on map `{name}`, which also needs {short:?} above 0");
            return Err(source.refuse("ValueMapNeeds", at, detail));
        }
        let picked = (rule.picture.as_str(), &rule.sampling);
        put_map(source, material, ("ValueMap", slot, name), picked)?;
    }
    for (key, rule) in &level.value_rules {
        let held = material.values.as_ref().and_then(|values| values.get(key));
        if let Some(&value) = held
            && !rule.holds(value)
        {
            let at = format!("materials[{index}].values.{key}");
            return Err(source.refuse("ValueRule", at, format!("{value} breaks the rule {rule:?}")));
        }
    }
    Ok(())
}

fn pieces(
    step: &mut Step<'_>,
    source: &Source,
    drawn: &Drawn,
    (of_mesh, baked): (&BTreeMap<String, String>, &BTreeMap<String, Baked>),
) -> Result<(), Refusal> {
    for mesh in &drawn.built {
        let stream = if mesh.colored { "color" } else { "lightmap" };
        let plan = baked.get(stream).ok_or_else(|| {
            let detail = format!("the mesh carries {stream} but `baked.{stream}` names no kind to draw it");
            source.refuse("BakedKind", format!("mesh {}", mesh.stem), detail)
        })?;
        let material = &of_mesh[&mesh.stem];
        let drawn_as = json!({"kind": plan.kind, "mesh": mesh.stem, "material": material});
        let mut entity = json!({"id": mesh.stem, "drawn": drawn_as, "transform": {}});
        if let Some(&largest_m) = drawn.largest_m.get(&mesh.stem) {
            entity["level_piece"] = item(&LevelPiece { largest_m });
        }
        step.item("entities", &mesh.stem, entity);
    }
    Ok(())
}

fn occluders(source: &Source, meshes: &[Built]) -> Result<Vec<u8>, Refusal> {
    let materials: Vec<gltf::Material<'_>> = source.gltf.materials().collect();
    let mut held = Vec::with_capacity(meshes.len());
    for mesh in meshes {
        let refused = |error| source.refuse("MeshReadBack", &mesh.stem, error);
        let read = MeshFile::read(Arc::from(mesh.bytes.as_slice()), &mesh.stem).map_err(refused)?;
        let material = &materials[mesh.material];
        held.push(Occluder {
            id: mesh.stem.clone(),
            material: material.name().unwrap_or_default().to_owned(),
            opaque: material.alpha_mode() == AlphaMode::Opaque,
            two_sided: material.double_sided(),
            positions: read.floats3(POSITION, &mesh.stem).map_err(refused)?,
            indices: read.index_values(),
        });
    }
    Ok(occluders_file(&held))
}

fn merged(step: &mut Step<'_>, source: &Source, level: &Level) -> Result<(), Refusal> {
    for rule in &level.merge {
        let size = (level.scale, level.tolerance);
        let merged = meshes::merged(source, (&rule.prefix, &rule.output), size)?;
        let triangles = merged.triangles;
        budget(source, "BudgetMergedTriangles", triangles, rule.triangles)?;
        let (count, vertices) = (merged.meshes.len(), merged.vertices);
        step.note(json!({"merged": {"output": rule.output, "meshes": count, "vertices": vertices, "triangles": triangles}}));
        let listing = serde_json::to_string_pretty(&json!({"meshes": merged.meshes}))
            .expect("merge_listing_serializes");
        step.write(&format!("{}.emb", rule.output), merged.bytes);
        step.write(&format!("{}.json", rule.output), listing.into_bytes());
    }
    Ok(())
}

fn level(step: &mut Step<'_>) -> Result<(), Refusal> {
    let name = step.name().to_owned();
    let level: Level = step.params()?;
    let kinds = checked(&name, &level)?;
    let source = {
        let [input] = step.inputs() else {
            return Err(Refusal::new("LevelInputs", &name, "one glTF file"));
        };
        Source::read(input.name, input.bytes)?
    };
    let prefixes: Vec<&str> = level.merge.iter().map(|rule| rule.prefix.as_str()).collect();
    let size = (level.scale, level.tolerance);
    let drawn = meshes::drawn(&source, &name, size, (&prefixes, level.chunks.as_ref()))?;
    let budgets = &level.budgets;
    counted(step, &source, &drawn.built, (budgets.triangles, budgets.vertices))?;
    let (widest, vertices) = &drawn.widest_object;
    step.note(json!({"widest_object": {"node": widest, "vertices": vertices}, "chunks": drawn.chunks}));
    budget(&source, "BudgetObjectVertices", *vertices, budgets.object_vertices)?;
    let mut used: Vec<usize> = drawn.built.iter().map(|mesh| mesh.material).collect();
    used.sort_unstable();
    used.dedup();
    budget(&source, "BudgetMaterials", used.len(), budgets.materials)?;
    let maps = MapPlan {
        size: level.texture_size,
        sizes: &level.map_sizes,
        material_sizes: Some(&level.material_sizes),
        textures: &level.textures,
        exact_alpha: &[],
    };
    let plan = MaterialPlan {
        names: &level.material,
        kinds,
        maps: Maps::Own(maps),
    };
    let mut made = materials(step, &source, &name, &drawn.built, &plan)?;
    for material in &mut made.materials {
        baked(&source, material, &drawn.built, &level)?;
    }
    write_textures(step, &source, &made.maps, &maps)?;
    write_materials(step, &made.materials);
    write_meshes(step, &drawn.built);
    pieces(step, &source, &drawn, (&made.of_mesh, &level.baked))?;
    step.pass(&format!("{name}.occluders.bin"), occluders(&source, &drawn.built)?);
    merged(step, &source, &level)?;
    let found = markers(&source, Mat4::from_scale(Vec3::splat(level.scale)), |_| None);
    write_markers(step, &name, &found);
    Ok(())
}
