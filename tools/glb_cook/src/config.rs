use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::fault::Fault;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cook {
    pub output: String,
    pub uri_prefix: String,
    pub keys_per_second: f32,
    pub tolerance: f32,
    pub entries: Vec<Entry>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Entry {
    Character(Character),
    Clip(ClipEntry),
    Static(StaticEntry),
    Map(MapEntry),
    Picture(PictureEntry),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PictureEntry {
    pub name: String,
    pub source: String,
    pub encoding: Encoding,
    pub mips: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialNames {
    pub slot: String,
    pub module: String,
    pub lighting: String,
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
    pub root_motion: String,
    pub rig_from: Option<String>,
    pub clips: BTreeMap<String, String>,
    #[serde(default)]
    pub remove_travel: Vec<String>,
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
    pub merge: Vec<MergeRule>,
    pub budgets: MapBudgets,
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
            Self::Clip(entry) => &entry.name,
            Self::Static(entry) => &entry.name,
            Self::Map(entry) => &entry.name,
            Self::Picture(entry) => &entry.name,
        }
    }

    pub fn source(&self) -> &str {
        match self {
            Self::Character(entry) => &entry.source,
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
            Self::Static(_) | Self::Map(_) | Self::Picture(_) => None,
        }
    }

    pub fn maps(&self) -> Option<&[String]> {
        match self {
            Self::Character(entry) => entry.maps.as_deref(),
            Self::Static(entry) => entry.maps.as_deref(),
            Self::Map(entry) => entry.maps.as_deref(),
            Self::Clip(_) | Self::Picture(_) => None,
        }
    }

    pub fn map_sizes(&self) -> Option<&BTreeMap<String, u32>> {
        match self {
            Self::Character(entry) => Some(&entry.map_sizes),
            Self::Static(entry) => Some(&entry.map_sizes),
            Self::Map(entry) => Some(&entry.map_sizes),
            Self::Clip(_) | Self::Picture(_) => None,
        }
    }

    pub fn textures(&self) -> &[TextureVariant] {
        match self {
            Self::Character(entry) => &entry.textures,
            Self::Static(entry) => &entry.textures,
            Self::Map(entry) => &entry.textures,
            Self::Clip(_) | Self::Picture(_) => &[],
        }
    }

    pub fn remove_travel(&self) -> &[String] {
        match self {
            Self::Character(entry) => &entry.remove_travel,
            Self::Clip(entry) => &entry.remove_travel,
            Self::Static(_) | Self::Map(_) | Self::Picture(_) => &[],
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
    let cooked: Vec<&str> = match entry.maps() {
        Some(maps) => maps.iter().map(String::as_str).collect(),
        None => MAP_NAMES.to_vec(),
    };
    for (map, &size) in entry.map_sizes().into_iter().flatten() {
        if !cooked.contains(&map.as_str()) || size == 0 {
            faults.push(Fault::new(
                "MapSize",
                file,
                item("map_sizes"),
                format!("`{map}`: {size}; each key a cooked map, each size above zero"),
            ));
        }
    }
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
        && picture.encoding == Encoding::Bc5
    {
        faults.push(Fault::new(
            "PictureEncoding",
            file,
            item("encoding"),
            "bc5 keeps two channels; a picture keeps four",
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
        Entry::Clip(_) | Entry::Picture(_) => (None, None),
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
    for (index, entry) in cook.entries.iter().enumerate() {
        if cook.entries[..index]
            .iter()
            .any(|earlier| earlier.name() == entry.name())
        {
            faults.push(Fault::new("EntryTwice", &file, "entries", entry.name()));
        }
        entry_faults(&file, entry, &mut faults);
    }
    if faults.is_empty() {
        Ok(cook)
    } else {
        Err(faults)
    }
}
