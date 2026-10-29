use engine_scene::{
    SceneAlphaMode, SceneDocument, SceneFilter, SceneMap, SceneMaterial, SceneMipmapFilter,
    SceneSurface, SceneTexture, SceneTextureMap, SceneWrap,
};
use gltf::texture::{MagFilter, MinFilter, WrappingMode};
use serde_json::Value;

use engine_core::{Error, ErrorCode, Expected};

use crate::{ImportedFile, unsupported};

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
            gltf::material::AlphaMode::Blend => {
                return Err(unsupported(format!("{path}.alphaMode"))
                    .got("BLEND")
                    .variants(["OPAQUE", "MASK"]));
            }
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
        scene.materials.push(SceneMaterial {
            id: format!("{id}.material{index}"),
            color: [red, green, blue, 1.0],
            texture,
            surface: Some(SceneSurface {
                roughness: pbr.roughness_factor(),
                metalness: pbr.metallic_factor(),
                emission: 0.0,
                ripples: 0.0,
            }),
            opacity,
            metallic_roughness_texture,
            normal_texture,
            normal_scale: normal
                .map(|normal| normal.scale())
                .filter(|scale| *scale != 1.0),
            occlusion_texture,
            occlusion_strength: occlusion
                .map(|occlusion| occlusion.strength())
                .filter(|strength| *strength != 1.0),
            emissive: (emissive != [0.0; 3]).then_some(emissive),
            emissive_texture,
            alpha_mode,
            alpha_cutoff,
            double_sided: material.double_sided().then_some(true),
            foliage: None,
            world_tiling: None,
            top: None,
            grounding: None,
        });
    }
    Ok(())
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
