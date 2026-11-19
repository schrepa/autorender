use std::collections::BTreeMap;

use cook::{Refusal, Step};
use gltf::material::AlphaMode;
use gltf::texture::{MagFilter, MinFilter, WrappingMode};
use serde::Serialize;
use serde_json::{Value, json};

use crate::geometry::Built;
use crate::maps::{self, Map, MapKind, Sampling, wrap};
use crate::params::{MaterialNames, SharedMaps, TextureVariant};
use crate::source::Source;

#[derive(Clone, Serialize)]
pub struct Material {
    #[serde(skip)]
    pub index: usize,
    pub id: String,
    pub color: [f32; 4],
    pub normal_scale: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emissive: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha_mode: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha_cutoff: Option<f32>,
    pub cull: &'static str,
    pub slots: BTreeMap<String, Slot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<BTreeMap<String, f32>>,
    pub lighting: String,
}

#[derive(Clone, Serialize)]
pub struct Slot {
    pub module: String,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub maps: BTreeMap<String, Value>,
}

#[derive(Serialize)]
struct MapRef {
    id: String,
    wrap_u: &'static str,
    wrap_v: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    mag_filter: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_filter: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mipmap_filter: Option<&'static str>,
    tex_coord: u32,
}

pub struct MaterialPlan<'a> {
    pub names: &'a MaterialNames,
    pub size: u32,
    pub sizes: &'a BTreeMap<String, u32>,
    pub kinds: Vec<MapKind>,
    pub textures: &'a [TextureVariant],
    pub material_sizes: Option<&'a BTreeMap<String, BTreeMap<String, u32>>>,
    pub exact_alpha: &'a [String],
    pub shared: Option<&'a SharedMaps>,
}

pub struct Materials {
    pub maps: Vec<Map>,
    pub materials: Vec<Material>,
    pub of_mesh: BTreeMap<String, String>,
}

struct Cooked {
    stem: String,
    maps: Vec<Map>,
    borrowed: Vec<(MapKind, String, Sampling)>,
}

type MapKey = (usize, MapKind, u32, [u32; 2]);

pub fn item<T: Serialize>(value: &T) -> Value {
    let text = serde_json::to_string(value).expect("items_serialize");
    serde_json::from_str(&text).expect("items_parse")
}

impl MaterialPlan<'_> {
    fn size(&self, material: Option<&str>, kind: MapKind) -> u32 {
        material
            .and_then(|name| self.material_sizes?.get(name)?.get(kind.name()).copied())
            .or_else(|| self.sizes.get(kind.name()).copied())
            .unwrap_or(self.size)
    }
}

const fn wrap_name(mode: WrappingMode) -> &'static str {
    match mode {
        WrappingMode::Repeat => "repeat",
        WrappingMode::MirroredRepeat => "mirrored_repeat",
        WrappingMode::ClampToEdge => "clamp_to_edge",
    }
}

fn map_ref(id: String, sampling: &Sampling) -> Value {
    let (min_filter, mipmap_filter) = match sampling.min {
        None => (None, None),
        Some(MinFilter::Nearest) => (Some("nearest"), Some("none")),
        Some(MinFilter::Linear) => (Some("linear"), Some("none")),
        Some(MinFilter::NearestMipmapNearest) => (Some("nearest"), Some("nearest")),
        Some(MinFilter::LinearMipmapNearest) => (Some("linear"), Some("nearest")),
        Some(MinFilter::NearestMipmapLinear) => (Some("nearest"), Some("linear")),
        Some(MinFilter::LinearMipmapLinear) => (Some("linear"), Some("linear")),
    };
    item(&MapRef {
        id,
        wrap_u: wrap_name(sampling.wrap_u),
        wrap_v: wrap_name(sampling.wrap_v),
        mag_filter: sampling.mag.map(|filter| match filter {
            MagFilter::Nearest => "nearest",
            MagFilter::Linear => "linear",
        }),
        min_filter,
        mipmap_filter,
        tex_coord: sampling.tex_coord,
    })
}

fn material(
    source: &Source,
    held: &gltf::Material<'_>,
    cooked: &Cooked,
    names: &MaterialNames,
) -> Result<Material, Refusal> {
    let index = held.index().unwrap_or_default();
    let item = format!("materials[{index}]");
    let [red, green, blue, alpha] = held.pbr_metallic_roughness().base_color_factor();
    if alpha != 1.0 {
        let detail = format!("base colour alpha factor {alpha}");
        return Err(source.refuse("AlphaFactor", item, detail));
    }
    let (alpha_mode, alpha_cutoff) = match held.alpha_mode() {
        AlphaMode::Opaque => (None, None),
        AlphaMode::Mask => (Some("mask"), held.alpha_cutoff()),
        AlphaMode::Blend => {
            return Err(source.refuse("BlendAlpha", item, "BLEND alpha is refused"));
        }
    };
    if held.emissive_texture().is_some() {
        return Err(source.refuse("EmissiveMap", item, "emissive textures are not cooked"));
    }
    let emissive = held.emissive_factor();
    let values = match (held.emissive_strength(), &names.emissive_value) {
        (None, _) => None,
        (Some(strength), Some(name)) => Some(BTreeMap::from([(name.clone(), strength)])),
        (Some(strength), None) => {
            return Err(source.refuse(
                "EmissiveStrength",
                item,
                format!("emissive strength {strength}: the entry's `material.emissive_value` names no value for it"),
            ));
        }
    };
    let maps = cooked
        .maps
        .iter()
        .map(|map| (map.kind, &map.stem, &map.sampling))
        .chain(
            cooked
                .borrowed
                .iter()
                .map(|(kind, stem, sampling)| (*kind, stem, sampling)),
        )
        .map(|(kind, stem, sampling)| (kind.name().to_owned(), map_ref(stem.clone(), sampling)))
        .collect();
    let slots = BTreeMap::from([(
        names.slot.clone(),
        Slot {
            module: names.module.clone(),
            maps,
        },
    )]);
    Ok(Material {
        index,
        id: cooked.stem.clone(),
        color: [red, green, blue, 1.0],
        normal_scale: held.normal_texture().map_or(1.0, |normal| normal.scale()),
        emissive: (emissive != [0.0; 3]).then_some(emissive),
        alpha_mode,
        alpha_cutoff,
        cull: if held.double_sided() { "none" } else { "back" },
        slots,
        values,
        lighting: names.lighting.clone(),
    })
}

fn extras_values(source: &Source, index: usize) -> Result<BTreeMap<String, f32>, Refusal> {
    let item = format!("materials[{index}].extras");
    match source.extras("materials", index) {
        Value::Null => Ok(BTreeMap::new()),
        Value::Object(extras) => extras
            .into_iter()
            .map(|(key, value)| match value.as_f64() {
                Some(number) if number.is_finite() => Ok((key, number as f32)),
                Some(_) | None => Err(source.refuse(
                    "MaterialExtras",
                    item.clone(),
                    format!("`{key}` is {value}; material extras become numeric values"),
                )),
            })
            .collect(),
        other => Err(source.refuse("MaterialExtras", item, format!("{other} is not an object"))),
    }
}

fn cook_one(
    step: &mut Step<'_>,
    source: &Source,
    (index, stem): (usize, String),
    plan: &MaterialPlan<'_>,
    shared: &mut BTreeMap<MapKey, String>,
) -> Result<(Cooked, Material), Refusal> {
    let held = source
        .gltf
        .materials()
        .nth(index)
        .expect("material_index_from_document");
    let item = format!("materials[{index}]");
    if held.occlusion_texture().is_some() {
        step.note(json!({"note": "OcclusionIgnored", "file": source.file, "item": item, "detail": "ORM red is written as 1"}));
    }
    if held.double_sided() {
        step.note(json!({"warning": "DoubleSided", "file": source.file, "item": item, "detail": "doubleSided true: cull none"}));
    }
    let plain = plan
        .kinds
        .iter()
        .all(|&kind| maps::texture_of(&held, kind).is_none());
    if plain {
        step.note(json!({"note": "PlainMaterial", "file": source.file, "item": item, "detail": "the material names none of the cooked maps; it is written without maps"}));
    }
    let mut cooked = Cooked {
        stem,
        maps: Vec::new(),
        borrowed: Vec::new(),
    };
    for &kind in plan.kinds.iter().filter(|_| !plain) {
        let size = plan.size(held.name(), kind);
        let key = maps::key(&held, kind, size);
        let owner = match plan.shared {
            Some(SharedMaps::Use(owner)) => Some(owner),
            Some(SharedMaps::Make(_)) | None => key.and_then(|key| shared.get(&key)),
        };
        if let Some(owner) = owner {
            let (texture, tex_coord) = maps::texture_of(&held, kind).ok_or_else(|| {
                let at = format!("{item} {}", kind.name());
                source.refuse("MissingMap", at, "the asset's material has no such texture")
            })?;
            let borrowed = format!("{owner}_{}", kind.name());
            let sampling = maps::sampling(&texture, tex_coord);
            cooked.borrowed.push((kind, borrowed, sampling));
            continue;
        }
        let map_stem = match plan.shared {
            Some(SharedMaps::Make(stem)) => stem.clone(),
            Some(SharedMaps::Use(_)) | None => cooked.stem.clone(),
        };
        let map = maps::map(source, &held, kind, &map_stem, size)?;
        if let Some(key) = key {
            shared.insert(key, map_stem);
        }
        cooked.maps.push(map);
    }
    let mut material = material(source, &held, &cooked, plan.names)?;
    let extras = extras_values(source, index)?;
    if !extras.is_empty() {
        let values = material.values.get_or_insert_with(BTreeMap::new);
        for (key, value) in extras {
            if let Some(held) = values.insert(key.clone(), value)
                && held.to_bits() != value.to_bits()
            {
                return Err(source.refuse(
                    "MaterialExtras",
                    format!("materials[{index}].extras.{key}"),
                    format!("{value} here, but the glTF material itself writes {held}"),
                ));
            }
        }
    }
    Ok((cooked, material))
}

pub fn materials(
    step: &mut Step<'_>,
    source: &Source,
    name: &str,
    meshes: &[Built],
    plan: &MaterialPlan<'_>,
) -> Result<Materials, Refusal> {
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
    let (mut maps, mut materials, mut borrowed) = (Vec::new(), Vec::new(), 0);
    let mut shared = BTreeMap::new();
    for &index in &used {
        let (cooked, material) = cook_one(step, source, (index, stem(index)), plan, &mut shared)?;
        borrowed += cooked.borrowed.len();
        maps.extend(cooked.maps);
        materials.push(material);
    }
    let named: Vec<&str> = used
        .iter()
        .filter_map(|&index| source.gltf.materials().nth(index)?.name())
        .collect();
    let sized = plan.material_sizes.into_iter().flat_map(BTreeMap::keys);
    let per_variant = plan
        .textures
        .iter()
        .flat_map(|variant| variant.material_sizes.keys());
    if let Some(wanted) = sized
        .chain(per_variant)
        .find(|wanted| !named.contains(&wanted.as_str()))
    {
        let at = format!("material_sizes.{wanted}");
        return Err(source.refuse("MaterialSizes", at, "no drawn material has this name"));
    }
    let of_mesh: BTreeMap<String, String> = meshes
        .iter()
        .map(|mesh| (mesh.stem.clone(), stem(mesh.material)))
        .collect();
    step.note(json!({"mesh_materials": of_mesh, "materials": used.len(), "shared_maps": borrowed}));
    Ok(Materials {
        maps,
        materials,
        of_mesh,
    })
}

pub fn write_textures(
    step: &mut Step<'_>,
    source: &Source,
    maps: &[Map],
    plan: &MaterialPlan<'_>,
) -> Result<(), Refusal> {
    for map in maps {
        let wraps = [wrap(map.sampling.wrap_u), wrap(map.sampling.wrap_v)];
        let chain = cook_texture::chain(map.kind.texels(), &map.decoded, wraps);
        let exact = plan.exact_alpha.iter().any(|name| name == map.kind.name());
        let mut variants = serde_json::Map::new();
        for variant in plan.textures {
            let encoding = variant.encoding(map.kind).expect("variant_maps_checked");
            if encoding.srgb() != (map.kind == MapKind::BaseColor) {
                step.note(json!({"warning": "TextureSpace", "file": source.file, "item": map.stem, "detail": format!("variant `{}` stores {} as {}", variant.name, map.kind.name(), encoding.name())}));
            }
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
            let levels = &chain[top..];
            let encoded = cook_texture::encode(map.kind.texels(), encoding, levels, exact)
                .map_err(|error| {
                    let at = format!("{} {}", map.stem, encoding.name());
                    source.refuse("TextureEncode", at, error)
                })?;
            step.note(json!({"map": map.stem, "variant": variant.name, "encoding": encoding.name(), "levels": levels.len(), "file_bytes": encoded.file.len(), "gpu_bytes": encoded.gpu_bytes, "encode_seconds": encoded.seconds, "quality": encoded.quality}));
            let file = format!("{}.{}.ktx2", map.stem, variant.name);
            step.write(&file, encoded.file);
            variants.insert(variant.name.clone(), Value::String(file));
        }
        step.item(
            "textures",
            &map.stem,
            json!({"id": map.stem, "variants": variants}),
        );
    }
    Ok(())
}

pub fn write_materials(step: &mut Step<'_>, materials: &[Material]) {
    for material in materials {
        step.item("materials", &material.id, item(material));
    }
}

pub fn write_meshes(step: &mut Step<'_>, meshes: &[Built]) {
    for mesh in meshes {
        let file = format!("{}.emb", mesh.stem);
        let entry = json!({"id": mesh.stem, "source": {"kind": "buffer", "uri": file}});
        step.item("meshes", &mesh.stem, entry);
        step.write(&file, mesh.bytes.clone());
    }
}
