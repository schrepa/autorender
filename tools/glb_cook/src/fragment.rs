use std::collections::BTreeMap;

use engine_scene::{
    MeshSource, SceneAlphaMode, SceneCull, SceneDocument, SceneFiles, SceneFilter, SceneKinds,
    SceneMap, SceneMaterial, SceneMesh, SceneMipmapFilter, SceneSlot, SceneSlotFill, SceneTexture,
    SceneTextureMap, SceneWrap,
};
use gltf::material::AlphaMode;
use gltf::texture::{MagFilter, MinFilter, WrappingMode};
use serde::Serialize;

use crate::config::MaterialNames;
use crate::fault::Fault;
use crate::geometry::Built;
use crate::maps::{Map, MapKind, Sampling};
use crate::source::Source;

#[derive(Serialize)]
pub struct Fragment {
    pub textures: Vec<SceneTexture>,
    pub materials: Vec<SceneMaterial>,
    pub meshes: Vec<SceneMesh>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<FragmentEntity>,
}

#[derive(Clone, Serialize)]
pub struct FragmentEntity {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drawn: Option<FragmentDrawn>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level_piece: Option<LevelPiece>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level_sets: Option<LevelSets>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform: Option<BTreeMap<String, f32>>,
}

#[derive(Clone, Serialize)]
pub struct LevelSets {
    pub file: String,
}

#[derive(Clone, Serialize)]
pub struct LevelPiece {
    pub largest_m: f32,
}

#[derive(Clone, Serialize)]
pub struct FragmentDrawn {
    pub kind: String,
    pub mesh: String,
    pub material: String,
}

pub struct CookedMaterial {
    pub stem: String,
    pub maps: Vec<Map>,
    pub borrowed: Vec<(MapKind, String, Sampling)>,
}

fn wrap(mode: WrappingMode) -> SceneWrap {
    match mode {
        WrappingMode::Repeat => SceneWrap::Repeat,
        WrappingMode::MirroredRepeat => SceneWrap::MirroredRepeat,
        WrappingMode::ClampToEdge => SceneWrap::ClampToEdge,
    }
}

fn scene_map(id: String, sampling: &Sampling) -> SceneMap {
    let (min_filter, mipmap_filter) = match sampling.min {
        None => (None, None),
        Some(MinFilter::Nearest) => (Some(SceneFilter::Nearest), Some(SceneMipmapFilter::None)),
        Some(MinFilter::Linear) => (Some(SceneFilter::Linear), Some(SceneMipmapFilter::None)),
        Some(MinFilter::NearestMipmapNearest) => {
            (Some(SceneFilter::Nearest), Some(SceneMipmapFilter::Nearest))
        }
        Some(MinFilter::LinearMipmapNearest) => {
            (Some(SceneFilter::Linear), Some(SceneMipmapFilter::Nearest))
        }
        Some(MinFilter::NearestMipmapLinear) => {
            (Some(SceneFilter::Nearest), Some(SceneMipmapFilter::Linear))
        }
        Some(MinFilter::LinearMipmapLinear) => {
            (Some(SceneFilter::Linear), Some(SceneMipmapFilter::Linear))
        }
    };
    SceneMap::Map(SceneTextureMap {
        id,
        wrap_u: Some(wrap(sampling.wrap_u)),
        wrap_v: Some(wrap(sampling.wrap_v)),
        mag_filter: sampling.mag.map(|filter| match filter {
            MagFilter::Nearest => SceneFilter::Nearest,
            MagFilter::Linear => SceneFilter::Linear,
        }),
        min_filter,
        mipmap_filter,
        offset: None,
        rotation: None,
        scale: None,
        tex_coord: Some(sampling.tex_coord),
    })
}

pub fn material(
    source: &Source,
    held: &gltf::Material<'_>,
    cooked: &CookedMaterial,
    names: &MaterialNames,
) -> Result<SceneMaterial, Fault> {
    let item = format!("materials[{}]", held.index().unwrap_or_default());
    let [red, green, blue, alpha] = held.pbr_metallic_roughness().base_color_factor();
    if alpha != 1.0 {
        return Err(source.fault(
            "AlphaFactor",
            item,
            format!("base colour alpha factor {alpha}"),
        ));
    }
    let (alpha_mode, alpha_cutoff) = match held.alpha_mode() {
        AlphaMode::Opaque => (None, None),
        AlphaMode::Mask => (Some(SceneAlphaMode::Mask), held.alpha_cutoff()),
        AlphaMode::Blend => return Err(source.fault("BlendAlpha", item, "BLEND alpha is refused")),
    };
    if held.emissive_texture().is_some() {
        return Err(source.fault("EmissiveMap", item, "emissive textures are not cooked"));
    }
    let emissive = held.emissive_factor();
    let values = match (held.emissive_strength(), &names.emissive_value) {
        (None, _) => None,
        (Some(strength), Some(name)) => Some(BTreeMap::from([(name.clone(), strength)])),
        (Some(strength), None) => {
            return Err(source.fault(
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
        .map(|(kind, stem, sampling)| (kind.name().to_owned(), scene_map(stem.clone(), sampling)))
        .collect();
    let slots = BTreeMap::from([(
        names.slot.clone(),
        SceneSlot::Filled(SceneSlotFill {
            module: names.module.clone(),
            maps,
        }),
    )]);
    Ok(SceneMaterial {
        id: cooked.stem.clone(),
        color: Some([red, green, blue, 1.0]),
        surface: None,
        opacity: None,
        normal_scale: Some(held.normal_texture().map_or(1.0, |normal| normal.scale())),
        emissive: (emissive != [0.0; 3]).then_some(emissive),
        alpha_mode,
        alpha_cutoff,
        cull: Some(if held.double_sided() {
            SceneCull::None
        } else {
            SceneCull::Back
        }),
        depth_test: None,
        depth_write: None,
        depth_bias: None,
        slots: Some(slots),
        values,
        lighting: Some(names.lighting.clone()),
    })
}

pub fn assemble(
    (prefix, ending): (&str, &str),
    meshes: &[Built],
    materials: &[CookedMaterial],
    (scene_materials, pictures, entities): (
        Vec<SceneMaterial>,
        &[(String, String)],
        Vec<FragmentEntity>,
    ),
) -> Fragment {
    let texture = |id: &str, uri: String| SceneTexture {
        id: id.to_owned(),
        width: None,
        height: None,
        rgba8: None,
        uri: Some(uri),
    };
    Fragment {
        textures: materials
            .iter()
            .flat_map(|material| &material.maps)
            .map(|map| texture(&map.stem, format!("{prefix}{}{ending}", map.stem)))
            .chain(pictures.iter().map(|(id, uri)| texture(id, uri.clone())))
            .collect(),
        materials: scene_materials,
        entities,
        meshes: meshes
            .iter()
            .map(|mesh| SceneMesh {
                id: mesh.stem.clone(),
                source: MeshSource::Buffer {
                    uri: format!("{prefix}{}.emb", mesh.stem),
                },
            })
            .collect(),
    }
}

pub fn validate(
    (file, name): (&str, &str),
    fragment: &Fragment,
    prefix: &str,
    meshes: &[Built],
    textures: &[(String, &[u8])],
) -> Result<String, Fault> {
    let fault = |detail: &dyn std::fmt::Display| {
        Fault::new("FragmentInvalid", file, format!("{name} fragment"), detail)
    };
    let text = serde_json::to_string_pretty(fragment).map_err(|error| fault(&error))?;
    let mut document: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| fault(&error))?;
    document["id"] = serde_json::json!(name);
    document["revision"] = serde_json::json!(1);
    document["entities"] = serde_json::json!([]);
    let document =
        SceneDocument::from_json(&document.to_string()).map_err(|error| fault(&error))?;
    let mut files = SceneFiles::new();
    for mesh in meshes {
        files.add_buffer(&format!("{prefix}{}.emb", mesh.stem), mesh.bytes.clone());
    }
    for (uri, bytes) in textures {
        files
            .add_texture_file(uri, bytes.to_vec())
            .map_err(|error| fault(&error))?;
    }
    document
        .validate_with(&SceneKinds::default(), &files)
        .map_err(|error| fault(&error))?;
    Ok(text)
}
