use std::collections::BTreeMap;

use engine_scene::{
    SceneAlphaMode, SceneCull, SceneDocument, SceneFilter, SceneMap, SceneMaterial,
    SceneMipmapFilter, SceneSlot, SceneSlotFill, SceneTexture, SceneTextureMap, SceneWrap,
};
use gltf::texture::{MagFilter, MinFilter, WrappingMode};
use serde_json::Value;

use engine_core::{
    BASE_COLOR_MAP, EMISSIVE_MAP, Error, ErrorCode, Expected, MAPPED, MAPPED_SLOTS, NORMAL_MAP,
    SurfaceSlot,
};

use crate::{ImportedFile, unsupported};

const ROUGHNESS: &str = "roughness";
const METALNESS: &str = "metalness";
const OCCLUSION: &str = "occlusion";
const OCCLUSION_STRENGTH: &str = "occlusion_strength";
const METALLIC_ROUGHNESS_MAP: &str = "metallic_roughness";
const OCCLUSION_MAP: &str = "occlusion";

const GLTF_SLOTS: [(&str, &[&str]); 3] = [
    (ROUGHNESS, &[METALLIC_ROUGHNESS_MAP]),
    (METALNESS, &[METALLIC_ROUGHNESS_MAP]),
    (OCCLUSION, &[OCCLUSION_MAP]),
];

const KTX2_IDENTIFIER: [u8; 12] = [
    0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
];

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub(crate) fn import(
    document: &gltf::Document,
    json: &Value,
    binary: &[u8],
    id: &str,
    scene: &mut SceneDocument,
    files: &mut Vec<ImportedFile>,
) -> Result<(), Error> {
    for image in document.images() {
        let path = format!("$.images[{}]", image.index());
        let gltf::image::Source::View { view, mime_type } = image.source() else {
            return Err(unsupported(format!("{path}.uri")).expected(Expected::Absent));
        };
        let extension = match mime_type {
            "image/png" => "png",
            "image/jpeg" => "jpg",
            "image/ktx2" => "ktx2",
            _ => {
                return Err(unsupported(format!("{path}.mimeType"))
                    .got(mime_type)
                    .variants(["image/png", "image/jpeg", "image/ktx2"]));
            }
        };
        if view.stride().is_some() {
            return Err(unsupported(format!("{path}.bufferView")));
        }
        let bytes = &binary[view.offset()..view.offset() + view.length()];
        if extension == "ktx2" && !bytes.starts_with(&KTX2_IDENTIFIER) {
            return Err(Error::new(ErrorCode::InvalidTexture, path));
        }
        let file = ImportedFile::named(
            id,
            &format!("image{}", image.index()),
            extension,
            bytes.to_vec(),
        );
        scene.textures.push(SceneTexture::file(
            format!("{id}.image{}", image.index()),
            file.uri.clone(),
        ));
        files.push(file);
    }
    for material in document.materials() {
        let index = material.index().expect("material_has_index");
        let path = format!("$.materials[{index}]");
        let source = &json["materials"][index];
        let pbr = material.pbr_metallic_roughness();
        let [red, green, blue, alpha] = pbr.base_color_factor();
        let opacity =
            Some(if alpha < 0.001 { 0.001 } else { alpha }).filter(|opacity| *opacity < 1.0);
        let (opacity, alpha_mode, alpha_cutoff) = match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => (None, None, None),
            gltf::material::AlphaMode::Mask => (
                opacity,
                Some(SceneAlphaMode::Mask),
                Some(material.alpha_cutoff().unwrap_or(0.5)),
            ),
            gltf::material::AlphaMode::Blend => (opacity, Some(SceneAlphaMode::Blend), None),
        };
        let map = |field: &str, info: &Value, texture: Option<gltf::Texture<'_>>| {
            texture
                .map(|texture| texture_map(info, &texture, id, &format!("{path}.{field}")))
                .transpose()
        };
        let pbr_source = &source["pbrMetallicRoughness"];
        let texture = map(
            "pbrMetallicRoughness.baseColorTexture",
            &pbr_source["baseColorTexture"],
            pbr.base_color_texture().map(|info| info.texture()),
        )?;
        let metallic_roughness_texture = map(
            "pbrMetallicRoughness.metallicRoughnessTexture",
            &pbr_source["metallicRoughnessTexture"],
            pbr.metallic_roughness_texture().map(|info| info.texture()),
        )?;
        let normal = material.normal_texture();
        let normal_texture = map(
            "normalTexture",
            &source["normalTexture"],
            normal.as_ref().map(|info| info.texture()),
        )?;
        let occlusion = material.occlusion_texture();
        let occlusion_texture = map(
            "occlusionTexture",
            &source["occlusionTexture"],
            occlusion.as_ref().map(|info| info.texture()),
        )?;
        let emissive_texture = map(
            "emissiveTexture",
            &source["emissiveTexture"],
            material.emissive_texture().map(|info| info.texture()),
        )?;
        let strength = source["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"]
            .as_f64()
            .unwrap_or(1.0) as f32;
        let emissive = material.emissive_factor().map(|value| value * strength);
        let slots = mapped_slots(
            [
                (BASE_COLOR_MAP, texture),
                (METALLIC_ROUGHNESS_MAP, metallic_roughness_texture),
                (NORMAL_MAP, normal_texture),
                (OCCLUSION_MAP, occlusion_texture),
                (EMISSIVE_MAP, emissive_texture),
            ],
            alpha_mode == Some(SceneAlphaMode::Mask),
        );
        scene.materials.push(SceneMaterial {
            id: format!("{id}.material{index}"),
            provided: None,
            color: Some([red, green, blue, 1.0]),
            surface: Some(BTreeMap::from([
                (ROUGHNESS.to_owned(), pbr.roughness_factor()),
                (METALNESS.to_owned(), pbr.metallic_factor()),
            ])),
            opacity,
            normal_scale: normal
                .map(|normal| normal.scale())
                .filter(|scale| *scale != 1.0),
            emissive: (emissive != [0.0; 3]).then_some(emissive),
            alpha_mode,
            alpha_cutoff,
            cull: material.double_sided().then_some(SceneCull::None),
            depth_test: None,
            depth_write: None,
            depth_bias: None,
            slots,
            values: occlusion
                .map(|occlusion| occlusion.strength())
                .filter(|strength| *strength != 1.0)
                .map(|strength| BTreeMap::from([(OCCLUSION_STRENGTH.to_owned(), strength)])),
            lighting: None,
        });
    }
    Ok(())
}

fn mapped_slots(
    maps: [(&str, Option<SceneMap>); 5],
    masked: bool,
) -> Option<BTreeMap<String, SceneSlot>> {
    let mut slots: BTreeMap<String, SceneSlot> = BTreeMap::new();
    for (name, map) in maps {
        let Some(map) = map else {
            continue;
        };
        let reading = MAPPED_SLOTS
            .iter()
            .chain(&GLTF_SLOTS)
            .filter(|(slot, read)| {
                read.first() == Some(&name) && (masked || *slot != SurfaceSlot::Cutoff.name())
            });
        for (at, (slot, _)) in reading.enumerate() {
            let filled = if at == 0 {
                SceneSlot::Filled(SceneSlotFill {
                    module: MAPPED.to_owned(),
                    maps: BTreeMap::from([(name.to_owned(), map.clone())]),
                })
            } else {
                SceneSlot::Module(MAPPED.to_owned())
            };
            slots.insert((*slot).to_owned(), filled);
        }
    }
    (!slots.is_empty()).then_some(slots)
}

fn texture_map(
    info: &Value,
    texture: &gltf::Texture<'_>,
    id: &str,
    path: &str,
) -> Result<SceneMap, Error> {
    let Some(image) = texture.source().map(|image| image.index()) else {
        return Err(
            unsupported(format!("$.textures[{}].source", texture.index()))
                .expected(Expected::Present),
        );
    };
    let transform = &info["extensions"]["KHR_texture_transform"];
    let tex_coord = transform["texCoord"]
        .as_u64()
        .or_else(|| info["texCoord"].as_u64())
        .unwrap_or(0);
    if tex_coord > 1 {
        return Err(unsupported(format!("{path}.texCoord"))
            .at_most(1)
            .got(tex_coord));
    }
    let sampler = texture.sampler();
    let wrap = |mode| match mode {
        WrappingMode::Repeat => None,
        WrappingMode::MirroredRepeat => Some(SceneWrap::MirroredRepeat),
        WrappingMode::ClampToEdge => Some(SceneWrap::ClampToEdge),
    };
    let nearest = Some(SceneFilter::Nearest);
    let (min_filter, mipmap_filter) = match sampler.min_filter() {
        None | Some(MinFilter::LinearMipmapLinear) => (None, None),
        Some(MinFilter::Nearest) => (nearest, Some(SceneMipmapFilter::None)),
        Some(MinFilter::Linear) => (None, Some(SceneMipmapFilter::None)),
        Some(MinFilter::NearestMipmapNearest) => (nearest, Some(SceneMipmapFilter::Nearest)),
        Some(MinFilter::LinearMipmapNearest) => (None, Some(SceneMipmapFilter::Nearest)),
        Some(MinFilter::NearestMipmapLinear) => (nearest, None),
    };
    let number = |value: &Value| value.as_f64().map(|value| value as f32);
    let pair = |value: &Value| Some([number(&value[0])?, number(&value[1])?]);
    let map = SceneTextureMap {
        id: format!("{id}.image{image}"),
        wrap_u: wrap(sampler.wrap_s()),
        wrap_v: wrap(sampler.wrap_t()),
        mag_filter: matches!(sampler.mag_filter(), Some(MagFilter::Nearest))
            .then_some(SceneFilter::Nearest),
        min_filter,
        mipmap_filter,
        offset: pair(&transform["offset"]),
        rotation: number(&transform["rotation"]),
        scale: pair(&transform["scale"]),
        tex_coord: (tex_coord == 1).then_some(1),
    };
    let plain = SceneTextureMap {
        id: map.id.clone(),
        wrap_u: None,
        wrap_v: None,
        mag_filter: None,
        min_filter: None,
        mipmap_filter: None,
        offset: None,
        rotation: None,
        scale: None,
        tex_coord: None,
    };
    Ok(if map == plain {
        SceneMap::Id(map.id)
    } else {
        SceneMap::Map(map)
    })
}
