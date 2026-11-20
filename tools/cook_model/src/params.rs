use std::collections::BTreeMap;

use cook::Refusal;
use cook_texture::Encoding;
use engine_scene::SceneCull;
use serde::Deserialize;

use crate::maps::MapKind;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialNames {
    pub slot: String,
    pub module: String,
    pub lighting: String,
    pub emissive_value: Option<String>,
    pub cull: Option<SceneCull>,
}

#[derive(Deserialize)]
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
    pub const fn encoding(&self, kind: MapKind) -> Option<Encoding> {
        match kind {
            MapKind::BaseColor => self.base_color,
            MapKind::Normal => self.normal,
            MapKind::Orm => self.orm,
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Model {
    Character(Character),
    Static(Static),
    Borrowing(Borrowing),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Character {
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
    pub keys_per_second: f32,
    pub tolerance: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CharacterBudgets {
    pub triangles: usize,
    pub vertices: usize,
    pub joints: usize,
    pub influences: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Static {
    pub scale: f32,
    pub texture_size: u32,
    #[serde(default)]
    pub map_sizes: BTreeMap<String, u32>,
    pub material: MaterialNames,
    pub maps: Option<Vec<String>>,
    pub textures: Vec<TextureVariant>,
    #[serde(default)]
    pub exact_alpha: Vec<String>,
    pub budgets: StaticBudgets,
    pub tolerance: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Borrowing {
    pub scale: f32,
    pub maps_of: String,
    pub material: MaterialNames,
    pub maps: Option<Vec<String>>,
    pub budgets: StaticBudgets,
    pub tolerance: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialMaps {
    pub material: String,
    pub texture_size: u32,
    #[serde(default)]
    pub map_sizes: BTreeMap<String, u32>,
    pub maps: Option<Vec<String>>,
    pub textures: Vec<TextureVariant>,
    #[serde(default)]
    pub exact_alpha: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticBudgets {
    pub triangles: usize,
    pub vertices: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Form {
    pub form_of: String,
    pub scale: f32,
    pub tolerance: f32,
    pub budgets: FormBudgets,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormBudgets {
    pub triangles: usize,
    pub vertices: usize,
    pub influences: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Clip {
    pub root_motion: Option<String>,
    pub rig_from: Option<String>,
    pub scale: f32,
    pub clips: BTreeMap<String, String>,
    #[serde(default)]
    pub remove_travel: Vec<String>,
    pub keys_per_second: f32,
    pub tolerance: f32,
    pub budgets: ClipBudgets,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClipBudgets {
    pub joints: usize,
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

pub fn check_name(name: &str) -> Result<(), Refusal> {
    if identifier(name) {
        Ok(())
    } else {
        Err(Refusal::new(
            "EntryName",
            name,
            "not a lowercase identifier",
        ))
    }
}

pub fn check_scale(name: &str, scale: f32, tolerance: f32) -> Result<(), Refusal> {
    if !positive(scale) {
        return Err(Refusal::new("Scale", name, scale));
    }
    if !(tolerance.is_finite() && tolerance >= 0.0) {
        return Err(Refusal::new("Tolerance", name, tolerance));
    }
    Ok(())
}

pub fn check_clips(
    name: &str,
    clips: &BTreeMap<String, String>,
    remove_travel: &[String],
    keys_per_second: f32,
) -> Result<(), Refusal> {
    if let Some(clip) = clips.values().find(|clip| !identifier(clip)) {
        let detail = format!("`{clip}` is not a lowercase identifier");
        return Err(Refusal::new("ClipName", name, detail));
    }
    if let Some(travel) = remove_travel.iter().find(|held| !clips.contains_key(*held)) {
        let detail = format!("`{travel}` is not in clips");
        return Err(Refusal::new("RemoveTravel", name, detail));
    }
    if !positive(keys_per_second) {
        return Err(Refusal::new("KeysPerSecond", name, keys_per_second));
    }
    Ok(())
}

pub fn map_names(name: &str, maps: Option<&[String]>) -> Result<Vec<MapKind>, Refusal> {
    let maps: Vec<&str> = match maps {
        Some(maps) => maps.iter().map(String::as_str).collect(),
        None => MapKind::ALL.map(MapKind::name).to_vec(),
    };
    for (index, map) in maps.iter().enumerate() {
        if MapKind::ALL.iter().all(|kind| kind.name() != *map) || maps[..index].contains(map) {
            let detail = format!("`{map}`: each of base_color, normal, orm at most once");
            return Err(Refusal::new("MapName", name, detail));
        }
    }
    Ok(MapKind::ALL
        .into_iter()
        .filter(|kind| maps.contains(&kind.name()))
        .collect())
}

pub fn check_exact_alpha(
    name: &str,
    exact_alpha: &[String],
    kinds: &[MapKind],
) -> Result<(), Refusal> {
    match exact_alpha
        .iter()
        .find(|wanted| kinds.iter().all(|kind| kind.name() != *wanted))
    {
        Some(wanted) => {
            let detail = format!("`{wanted}` is not a map this entry cooks");
            Err(Refusal::new("ExactAlpha", name, detail))
        }
        None => Ok(()),
    }
}

pub fn map_kinds<'s>(
    name: &str,
    maps: Option<&[String]>,
    textures: &[TextureVariant],
    (size, sizes): (u32, impl IntoIterator<Item = (&'s String, &'s u32)>),
) -> Result<Vec<MapKind>, Refusal> {
    let kinds = map_names(name, maps)?;
    if size == 0 {
        return Err(Refusal::new("TextureSize", name, 0));
    }
    for (map, &size) in sizes {
        if kinds.iter().all(|kind| kind.name() != map) || size == 0 {
            let detail = format!("`{map}`: {size}; each key a cooked map, each size above zero");
            return Err(Refusal::new("MapSize", name, detail));
        }
    }
    if textures.is_empty() {
        let detail = "name at least one texture variant: scenes load only KTX2 textures";
        return Err(Refusal::new("TextureVariants", name, detail));
    }
    for (index, variant) in textures.iter().enumerate() {
        if !identifier(&variant.name)
            || textures[..index]
                .iter()
                .any(|held| held.name == variant.name)
        {
            let detail = format!(
                "`{}` is not a lowercase identifier or is named twice",
                variant.name
            );
            return Err(Refusal::new("TextureVariantName", name, detail));
        }
        if let Some(kind) = MapKind::ALL
            .into_iter()
            .find(|&kind| kinds.contains(&kind) != variant.encoding(kind).is_some())
        {
            let detail = format!(
                "variant `{}` must name an encoding for exactly the cooked maps; `{}` differs",
                variant.name,
                kind.name()
            );
            return Err(Refusal::new("TextureVariantMaps", name, detail));
        }
    }
    Ok(kinds)
}
