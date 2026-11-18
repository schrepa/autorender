use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::fault::Fault;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cook {
    pub output: String,
    pub uri_prefix: String,
    pub keys_per_second: f32,
    pub tolerance: f32,
    pub entries: Vec<Entry>,
    #[serde(default)]
    pub scenes: Vec<SceneVariant>,
    #[serde(default)]
    pub scene_parts: Vec<ScenePart>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenePart {
    pub id: String,
    pub scene: String,
    pub needs: Vec<String>,
    #[serde(default)]
    pub textures: Vec<serde_json::Value>,
    #[serde(default)]
    pub materials: Vec<serde_json::Value>,
    #[serde(default)]
    pub meshes: Vec<serde_json::Value>,
    #[serde(default)]
    pub entities: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneVariant {
    pub from: String,
    pub to: String,
    pub replaces: String,
    pub variant: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Entry {
    Character(Character),
    Form(FormEntry),
    Clip(ClipEntry),
    Static(StaticEntry),
    Map(MapEntry),
    Picture(PictureEntry),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormEntry {
    pub name: String,
    pub source: String,
    pub form_of: String,
    pub budgets: FormBudgets,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormBudgets {
    pub triangles: usize,
    pub vertices: usize,
    pub influences: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PictureEntry {
    pub name: String,
    pub source: String,
    pub encodings: BTreeMap<String, Encoding>,
    pub mips: bool,
    pub wrap: [Wrap; 2],
    #[serde(default)]
    pub optional: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Wrap {
    ClampToEdge,
    Repeat,
    MirroredRepeat,
}

impl Wrap {
    pub const fn mode(self) -> gltf::texture::WrappingMode {
        match self {
            Self::ClampToEdge => gltf::texture::WrappingMode::ClampToEdge,
            Self::Repeat => gltf::texture::WrappingMode::Repeat,
            Self::MirroredRepeat => gltf::texture::WrappingMode::MirroredRepeat,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BakedPlan {
    pub kind: String,
    #[serde(default)]
    pub maps: BTreeMap<String, BakedMap>,
    #[serde(default)]
    pub values: BTreeMap<String, f32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BakedMap {
    pub picture: String,
    pub variant: String,
    pub sampling: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueMap {
    pub picture: String,
    pub variant: String,
    pub sampling: serde_json::Map<String, serde_json::Value>,
    pub when: Vec<String>,
    #[serde(default)]
    pub needs: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueRule {
    pub one_of: Option<Vec<f32>>,
    pub min: Option<f32>,
    pub max: Option<f32>,
}

impl ValueRule {
    pub fn holds(&self, value: f32) -> bool {
        value.is_finite()
            && self
                .one_of
                .as_ref()
                .is_none_or(|allowed| allowed.contains(&value))
            && self.min.is_none_or(|least| value >= least)
            && self.max.is_none_or(|most| value <= most)
    }
}

pub const BAKED_STREAMS: [&str; 2] = ["uv1", "color"];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialNames {
    pub slot: String,
    pub module: String,
    pub lighting: String,
    pub emissive_value: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Character {
    pub name: String,
    pub source: String,
    pub scale: f32,
    pub texture_size: u32,
    #[serde(default)]
    pub map_sizes: BTreeMap<String, u32>,
    pub root_motion: Option<String>,
    pub clips: BTreeMap<String, String>,
    #[serde(default)]
    pub remove_travel: Vec<String>,
    pub material: MaterialNames,
    pub maps: Option<Vec<String>>,
    #[serde(default)]
    pub textures: Vec<TextureVariant>,
    pub budgets: CharacterBudgets,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CharacterBudgets {
    pub triangles: usize,
    pub vertices: usize,
    pub joints: usize,
    pub influences: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClipEntry {
    pub name: String,
    pub source: String,
    pub root_motion: Option<String>,
    pub rig_from: Option<String>,
    pub clips: BTreeMap<String, String>,
    #[serde(default)]
    pub remove_travel: Vec<String>,
    pub keys_per_second: Option<f32>,
    pub tolerance: Option<f32>,
    pub budgets: ClipBudgets,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClipBudgets {
    pub joints: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticEntry {
    pub name: String,
    pub source: String,
    pub scale: f32,
    pub texture_size: u32,
    #[serde(default)]
    pub map_sizes: BTreeMap<String, u32>,
    pub material: MaterialNames,
    pub maps: Option<Vec<String>>,
    #[serde(default)]
    pub textures: Vec<TextureVariant>,
    #[serde(default)]
    pub exact_alpha: Vec<String>,
    pub map_stem: Option<String>,
    pub budgets: StaticBudgets,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticBudgets {
    pub triangles: usize,
    pub vertices: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MergeRule {
    pub prefix: String,
    pub output: String,
    pub triangles: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapEntry {
    pub name: String,
    pub source: String,
    pub scale: f32,
    pub texture_size: u32,
    #[serde(default)]
    pub map_sizes: BTreeMap<String, u32>,
    pub material: MaterialNames,
    pub maps: Option<Vec<String>>,
    #[serde(default)]
    pub textures: Vec<TextureVariant>,
    #[serde(default)]
    pub material_sizes: BTreeMap<String, BTreeMap<String, u32>>,
    #[serde(default)]
    pub baked: BTreeMap<String, BakedPlan>,
    #[serde(default)]
    pub value_maps: BTreeMap<String, ValueMap>,
    #[serde(default)]
    pub value_rules: BTreeMap<String, ValueRule>,
    pub merge: Vec<MergeRule>,
    pub chunks: Option<ChunkPlan>,
    pub visible_sets: Option<Box<SetsPlan>>,
    pub budgets: MapBudgets,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SetsPlan {
    pub data: String,
    pub cell_m: f32,
    pub eye_inset_m: f32,
    pub eye_heights_m: Vec<f32>,
    pub top_m: f32,
    pub padding_cells: u32,
    pub face_step_m: f32,
    pub probe_m: f32,
    pub target_gap_m: f32,
    pub ground: String,
    pub occluders: OccluderPlan,
    pub see_through: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OccluderPlan {
    pub names: Vec<String>,
    pub surfaces: Vec<String>,
    pub shrink_m: f32,
    #[serde(default)]
    pub starts_m: BTreeMap<String, f32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkPlan {
    pub cell_m: f32,
    pub fold_triangles: usize,
    pub detail_classes_m: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapBudgets {
    pub triangles: usize,
    pub vertices: usize,
    pub materials: usize,
    pub object_vertices: usize,
}

pub const MAP_NAMES: [&str; 3] = ["base_color", "normal", "orm"];

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Encoding {
    Rgba8Srgb,
    Rgba8,
    Bc7Srgb,
    Bc7,
    Bc5,
}

impl Encoding {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rgba8Srgb => "rgba8_srgb",
            Self::Rgba8 => "rgba8",
            Self::Bc7Srgb => "bc7_srgb",
            Self::Bc7 => "bc7",
            Self::Bc5 => "bc5",
        }
    }

    pub const fn srgb(self) -> bool {
        matches!(self, Self::Rgba8Srgb | Self::Bc7Srgb)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureVariant {
    pub name: String,
    #[serde(default)]
    pub material_sizes: BTreeMap<String, BTreeMap<String, u32>>,
    pub base_color: Option<Encoding>,
    pub normal: Option<Encoding>,
    pub orm: Option<Encoding>,
}

impl TextureVariant {
    pub fn encoding(&self, map: &str) -> Option<Encoding> {
        match map {
            "base_color" => self.base_color,
            "normal" => self.normal,
            "orm" => self.orm,
            _ => None,
        }
    }
}

impl Entry {
    pub fn name(&self) -> &str {
        match self {
            Self::Character(entry) => &entry.name,
            Self::Form(entry) => &entry.name,
            Self::Clip(entry) => &entry.name,
            Self::Static(entry) => &entry.name,
            Self::Map(entry) => &entry.name,
            Self::Picture(entry) => &entry.name,
        }
    }

    pub fn source(&self) -> &str {
        match self {
            Self::Character(entry) => &entry.source,
            Self::Form(entry) => &entry.source,
            Self::Clip(entry) => &entry.source,
            Self::Static(entry) => &entry.source,
            Self::Map(entry) => &entry.source,
            Self::Picture(entry) => &entry.source,
        }
    }

    pub fn clips(&self) -> Option<&BTreeMap<String, String>> {
        match self {
            Self::Character(entry) => Some(&entry.clips),
            Self::Clip(entry) => Some(&entry.clips),
            Self::Form(_) | Self::Static(_) | Self::Map(_) | Self::Picture(_) => None,
        }
    }

    pub fn maps(&self) -> Option<&[String]> {
        match self {
            Self::Character(entry) => entry.maps.as_deref(),
            Self::Static(entry) => entry.maps.as_deref(),
            Self::Map(entry) => entry.maps.as_deref(),
            Self::Form(_) | Self::Clip(_) | Self::Picture(_) => None,
        }
    }

    pub fn map_sizes(&self) -> Option<&BTreeMap<String, u32>> {
        match self {
            Self::Character(entry) => Some(&entry.map_sizes),
            Self::Static(entry) => Some(&entry.map_sizes),
            Self::Map(entry) => Some(&entry.map_sizes),
            Self::Form(_) | Self::Clip(_) | Self::Picture(_) => None,
        }
    }

    pub fn textures(&self) -> &[TextureVariant] {
        match self {
            Self::Character(entry) => &entry.textures,
            Self::Static(entry) => &entry.textures,
            Self::Map(entry) => &entry.textures,
            Self::Form(_) | Self::Clip(_) | Self::Picture(_) => &[],
        }
    }

    pub fn remove_travel(&self) -> &[String] {
        match self {
            Self::Character(entry) => &entry.remove_travel,
            Self::Clip(entry) => &entry.remove_travel,
            Self::Form(_) | Self::Static(_) | Self::Map(_) | Self::Picture(_) => &[],
        }
    }
}

pub fn identifier(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_lowercase())
        && characters.all(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
        })
}

fn positive(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

impl MapEntry {
    pub fn pictures(&self) -> Vec<(String, &str, &str)> {
        let mut out: Vec<(String, &str, &str)> = self
            .baked
            .iter()
            .flat_map(|(stream, plan)| {
                plan.maps.iter().map(move |(name, baked)| {
                    (
                        format!("baked.{stream}.maps.{name}"),
                        baked.picture.as_str(),
                        baked.variant.as_str(),
                    )
                })
            })
            .collect();
        out.extend(self.value_maps.iter().map(|(name, held)| {
            (
                format!("value_maps.{name}"),
                held.picture.as_str(),
                held.variant.as_str(),
            )
        }));
        out
    }
}

fn value_map_faults(file: &str, cook: &Cook, map: &MapEntry, faults: &mut Vec<Fault>) {
    for (slot_map, held) in &map.value_maps {
        let item = format!("entries.{}.value_maps.{slot_map}", map.name);
        let found = cook.entries.iter().any(|entry| match entry {
            Entry::Picture(picture) => {
                picture.name == held.picture && picture.encodings.contains_key(&held.variant)
            }
            Entry::Character(_)
            | Entry::Form(_)
            | Entry::Clip(_)
            | Entry::Static(_)
            | Entry::Map(_) => false,
        });
        let baked = map
            .baked
            .values()
            .any(|plan| plan.maps.contains_key(slot_map));
        if !found
            || baked
            || MAP_NAMES.contains(&slot_map.as_str())
            || held.sampling.contains_key("id")
            || held.when.is_empty()
        {
            faults.push(Fault::new("ValueMap", file, item, format!("picture `{}` variant `{}` must be a picture entry of this cook; the map name must not be a cooked or baked map; the sampling must not name an id; `when` names at least one value", held.picture, held.variant)));
        }
    }
}

fn baked_faults(file: &str, cook: &Cook, map: &MapEntry, faults: &mut Vec<Fault>) {
    value_map_faults(file, cook, map, faults);
    for (stream, plan) in &map.baked {
        let item = format!("entries.{}.baked.{stream}", map.name);
        if !BAKED_STREAMS.contains(&stream.as_str()) {
            faults.push(Fault::new(
                "BakedStream",
                file,
                item.clone(),
                format!("`{stream}`: one of {BAKED_STREAMS:?}"),
            ));
        }
        for (slot_map, baked) in &plan.maps {
            let found = cook.entries.iter().any(|entry| match entry {
                Entry::Picture(picture) => {
                    picture.name == baked.picture && picture.encodings.contains_key(&baked.variant)
                }
                Entry::Character(_)
                | Entry::Form(_)
                | Entry::Clip(_)
                | Entry::Static(_)
                | Entry::Map(_) => false,
            });
            if !found || MAP_NAMES.contains(&slot_map.as_str()) || baked.sampling.contains_key("id")
            {
                faults.push(Fault::new("BakedMap", file, format!("{item}.maps.{slot_map}"), format!("picture `{}` variant `{}` must be a picture entry of this cook; the map name must not be a cooked map and the sampling must not name an id", baked.picture, baked.variant)));
            }
        }
    }
}

fn size_faults(file: &str, entry: &Entry, faults: &mut Vec<Fault>) {
    let item = format!("entries.{}.map_sizes", entry.name());
    let cooked: Vec<&str> = match entry.maps() {
        Some(maps) => maps.iter().map(String::as_str).collect(),
        None => MAP_NAMES.to_vec(),
    };
    let per_material = match entry {
        Entry::Map(map) => map.material_sizes.values().flatten().collect(),
        Entry::Character(_)
        | Entry::Form(_)
        | Entry::Clip(_)
        | Entry::Static(_)
        | Entry::Picture(_) => Vec::new(),
    };
    for (map, &size) in entry.map_sizes().into_iter().flatten().chain(per_material) {
        if !cooked.contains(&map.as_str()) || size == 0 {
            faults.push(Fault::new(
                "MapSize",
                file,
                item.clone(),
                format!("`{map}`: {size}; each key a cooked map, each size above zero"),
            ));
        }
    }
}

fn entry_faults(file: &str, entry: &Entry, faults: &mut Vec<Fault>) {
    let name = entry.name();
    let item = |field: &str| format!("entries.{name}.{field}");
    if !identifier(name) {
        faults.push(Fault::new(
            "EntryName",
            file,
            item("name"),
            "not a lowercase identifier",
        ));
    }
    for clip in entry.clips().into_iter().flat_map(BTreeMap::values) {
        if !identifier(clip) {
            faults.push(Fault::new(
                "ClipName",
                file,
                item("clips"),
                format!("`{clip}` is not a lowercase identifier"),
            ));
        }
    }
    for travel in entry.remove_travel() {
        if entry
            .clips()
            .is_none_or(|clips| !clips.contains_key(travel))
        {
            faults.push(Fault::new(
                "RemoveTravel",
                file,
                item("remove_travel"),
                format!("`{travel}` is not in clips"),
            ));
        }
    }
    let variants = entry.textures();
    for (index, variant) in variants.iter().enumerate() {
        if !identifier(&variant.name)
            || variants[..index]
                .iter()
                .any(|held| held.name == variant.name)
        {
            faults.push(Fault::new(
                "TextureVariantName",
                file,
                item("textures"),
                format!(
                    "`{}` is not a lowercase identifier or is named twice",
                    variant.name
                ),
            ));
        }
        let cooked: Vec<&str> = match entry.maps() {
            Some(maps) => maps.iter().map(String::as_str).collect(),
            None => MAP_NAMES.to_vec(),
        };
        for map in MAP_NAMES {
            if cooked.contains(&map) != variant.encoding(map).is_some() {
                faults.push(Fault::new(
                    "TextureVariantMaps",
                    file,
                    item("textures"),
                    format!("variant `{}` must name an encoding for exactly the cooked maps; `{map}` differs", variant.name),
                ));
            }
        }
    }
    size_faults(file, entry, faults);
    let maps = entry.maps().unwrap_or_default();
    for (index, map) in maps.iter().enumerate() {
        if !MAP_NAMES.contains(&map.as_str()) || maps[..index].contains(map) {
            faults.push(Fault::new(
                "MapName",
                file,
                item("maps"),
                format!("`{map}`: each of {MAP_NAMES:?} at most once"),
            ));
        }
    }
    if let Entry::Picture(picture) = entry
        && (picture.encodings.is_empty()
            || picture
                .encodings
                .iter()
                .any(|(variant, &encoding)| !identifier(variant) || encoding == Encoding::Bc5))
    {
        faults.push(Fault::new(
            "PictureEncoding",
            file,
            item("encodings"),
            "one or more lowercase variant names, each with a four-channel encoding (not bc5)",
        ));
    }
    if let Entry::Map(map) = entry {
        for rule in &map.merge {
            if !identifier(&rule.output) || rule.prefix.is_empty() {
                faults.push(Fault::new(
                    "MergeRule",
                    file,
                    item("merge"),
                    format!("prefix `{}`, output `{}`", rule.prefix, rule.output),
                ));
            }
        }
    }
    let (scale, size) = match entry {
        Entry::Character(entry) => (Some(entry.scale), Some(entry.texture_size)),
        Entry::Static(entry) => (Some(entry.scale), Some(entry.texture_size)),
        Entry::Map(entry) => (Some(entry.scale), Some(entry.texture_size)),
        Entry::Form(_) | Entry::Clip(_) | Entry::Picture(_) => (None, None),
    };
    if let Some(scale) = scale
        && !positive(scale)
    {
        faults.push(Fault::new("Scale", file, item("scale"), scale));
    }
    if size == Some(0) {
        faults.push(Fault::new("TextureSize", file, item("texture_size"), 0));
    }
}

pub fn missing_baked_pictures(cook: &Cook, base: &Path) -> Vec<Fault> {
    let mut faults = Vec::new();
    for entry in &cook.entries {
        let Entry::Map(map) = entry else { continue };
        for (item, wanted, _) in map.pictures() {
            let source = cook.entries.iter().find_map(|held| match held {
                Entry::Picture(picture) if picture.name == wanted => Some(&picture.source),
                Entry::Character(_)
                | Entry::Form(_)
                | Entry::Clip(_)
                | Entry::Static(_)
                | Entry::Map(_)
                | Entry::Picture(_) => None,
            });
            if let Some(source) = source
                && !base.join(source).is_file()
            {
                faults.push(Fault::new(
                    "BakedPictureMissing",
                    &base.join(source).display().to_string(),
                    format!("entries.{}.{item}", map.name),
                    format!("the map `{}` needs picture `{wanted}` and its source is not there; nothing was cooked (make it, or remove `{item}` to cook without it)", map.name),
                ));
            }
        }
    }
    faults
}

pub fn read(path: &Path) -> Result<Cook, Vec<Fault>> {
    let file = path.display().to_string();
    let text = std::fs::read_to_string(path)
        .map_err(|error| vec![Fault::new("CookUnreadable", &file, "file", error)])?;
    let cook: Cook = serde_json::from_str(&text)
        .map_err(|error| vec![Fault::new("CookInvalid", &file, "file", error)])?;
    let mut faults = Vec::new();
    if !positive(cook.keys_per_second) {
        faults.push(Fault::new(
            "KeysPerSecond",
            &file,
            "keys_per_second",
            cook.keys_per_second,
        ));
    }
    if !cook.tolerance.is_finite() || cook.tolerance < 0.0 {
        faults.push(Fault::new("Tolerance", &file, "tolerance", cook.tolerance));
    }
    for entry in &cook.entries {
        if let Entry::Map(map) = entry
            && let Some(plan) = &map.chunks
            && !positive(plan.cell_m)
        {
            let item = format!("entries.{}.chunks.cell_m", map.name);
            faults.push(Fault::new("ChunkCell", &file, item, plan.cell_m));
        }
        if let Entry::Map(map) = entry
            && let Some(bounds) = map
                .chunks
                .as_ref()
                .and_then(|plan| plan.detail_classes_m.as_ref())
            && (bounds.is_empty()
                || !bounds.iter().copied().all(positive)
                || bounds.windows(2).any(|pair| pair[0] >= pair[1]))
        {
            let item = format!("entries.{}.chunks.detail_classes_m", map.name);
            faults.push(Fault::new(
                "DetailClasses",
                &file,
                item,
                format!("{bounds:?}: a non-empty list of rising positive sizes"),
            ));
        }
        let Entry::Clip(clip) = entry else {
            continue;
        };
        if let Some(rate) = clip.keys_per_second
            && !positive(rate)
        {
            let item = format!("entries.{}.keys_per_second", clip.name);
            faults.push(Fault::new("KeysPerSecond", &file, item, rate));
        }
        if let Some(tolerance) = clip.tolerance
            && !(tolerance.is_finite() && tolerance >= 0.0)
        {
            let item = format!("entries.{}.tolerance", clip.name);
            faults.push(Fault::new("Tolerance", &file, item, tolerance));
        }
    }
    for (index, scene) in cook.scenes.iter().enumerate() {
        if !identifier(&scene.replaces)
            || !identifier(&scene.variant)
            || scene.replaces == scene.variant
            || scene.from == scene.to
        {
            faults.push(Fault::new(
                "SceneVariant",
                &file,
                format!("scenes[{index}]"),
                "two different lowercase variant names and two different files",
            ));
        }
    }
    for (index, entry) in cook.entries.iter().enumerate() {
        if cook.entries[..index]
            .iter()
            .any(|earlier| earlier.name() == entry.name())
        {
            faults.push(Fault::new("EntryTwice", &file, "entries", entry.name()));
        }
        entry_faults(&file, entry, &mut faults);
    }
    for entry in &cook.entries {
        if let Entry::Map(map) = entry {
            baked_faults(&file, &cook, map, &mut faults);
        }
    }
    for part in &cook.scene_parts {
        let known = |name: &String| {
            cook.entries.iter().any(|entry| match entry {
                Entry::Picture(picture) => picture.name == *name,
                Entry::Character(_)
                | Entry::Form(_)
                | Entry::Clip(_)
                | Entry::Static(_)
                | Entry::Map(_) => false,
            })
        };
        let unnamed = [
            &part.textures,
            &part.materials,
            &part.meshes,
            &part.entities,
        ]
        .into_iter()
        .flatten()
        .any(|item| !item["id"].is_string());
        if part.needs.iter().any(|name| !known(name)) || unnamed {
            faults.push(Fault::new(
                "ScenePart",
                &file,
                format!("scene_parts.{}", part.id),
                "`needs` names picture entries of this cook and every item has a string `id`",
            ));
        }
    }
    if faults.is_empty() {
        Ok(cook)
    } else {
        Err(faults)
    }
}
