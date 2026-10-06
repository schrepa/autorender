//! glTF 2.0 metallic-roughness materials, mapped one to one onto scene
//! version 8 materials (decision 031). Images are written beside the scene
//! unchanged, for the host to decode (decision 030).

use engine_scene::{
    SceneAlphaMode, SceneDocument, SceneFilter, SceneMap, SceneMaterial, SceneMipmapFilter,
    SceneSurface, SceneTexture, SceneTextureMap, SceneWrap,
};
use gltf::texture::{MagFilter, MinFilter, WrappingMode};
use serde_json::Value;

use crate::{ImportError, ImportedFile, unsupported};

/// The first twelve bytes of every KTX 2.0 file.
const KTX2_IDENTIFIER: [u8; 12] = [
    0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
];

pub(crate) fn import(
    document: &gltf::Document,
    json: &Value,
    binary: &[u8],
    id: &str,
    scene: &mut SceneDocument,
    files: &mut Vec<ImportedFile>,
) -> Result<(), ImportError> {
    for image in document.images() {
        let path = format!("$.images[{}]", image.index());
        let gltf::image::Source::View { view, mime_type } = image.source() else {
            return Err(unsupported(
                path,
                "only images embedded in the GLB are supported; external and data URIs are rejected",
            ));
        };
        // KTX2 images (KHR_texture_basisu) stay supercompressed; hosts
        // transcode them at load (decision 037).
        let extension = match mime_type {
            "image/png" => "png",
            "image/jpeg" => "jpg",
            "image/ktx2" => "ktx2",
            _ => {
                return Err(unsupported(
                    path,
                    "images must be image/png, image/jpeg or image/ktx2",
                ));
            }
        };
        if view.stride().is_some() {
            return Err(unsupported(path, "image buffer views must be contiguous"));
        }
        let bytes = &binary[view.offset()..view.offset() + view.length()];
        if extension == "ktx2" && !bytes.starts_with(&KTX2_IDENTIFIER) {
            return Err(ImportError::new(
                "invalid_image",
                path,
                "image/ktx2 images start with the KTX 2.0 identifier",
            ));
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
        let index = material.index().expect("enumerated materials have indices");
        let path = format!("$.materials[{index}]");
        let source = &json["materials"][index];
        let pbr = material.pbr_metallic_roughness();
        let [red, green, blue, alpha] = pbr.base_color_factor();
        // Opacity is the base color factor's alpha; opaque materials ignore it.
        // Scene opacities are positive; alpha 0 masks or blends everything.
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
                .map(|texture| texture_map(json, info, &texture, id, &format!("{path}.{field}")))
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
        // Glass (decision 038): the transmission factor, with the index of
        // refraction and the volume's thickness factor that bend it; the
        // thickness counts as world units and transmission textures are
        // not read.
        let extensions = &source["extensions"];
        let factor = |value: &Value| value.as_f64().map(|value| value as f32);
        let transmission = factor(&extensions["KHR_materials_transmission"]["transmissionFactor"])
            .filter(|share| *share > 0.0)
            .map(|share| engine_core::math::min(share, 1.0));
        let ior = factor(&extensions["KHR_materials_ior"]["ior"])
            .filter(|_| transmission.is_some())
            .map(|ior| engine_core::math::clamp(ior, 1.0, 3.0));
        let thickness = factor(&extensions["KHR_materials_volume"]["thicknessFactor"])
            .filter(|thickness| transmission.is_some() && *thickness > 0.0);
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
            transmission,
            ior,
            thickness,
            world_tiling: None,
            top: None,
        });
    }
    Ok(())
}

/// A texture reference of a material: glTF sampler state and
/// `KHR_texture_transform` placement from its texture info `info`, as a
/// plain ID when both are glTF's defaults. A `KHR_texture_basisu` image
/// replaces the texture's own source (decision 037).
fn texture_map(
    root: &Value,
    info: &Value,
    texture: &gltf::Texture<'_>,
    id: &str,
    path: &str,
) -> Result<SceneMap, ImportError> {
    let basisu = &root["textures"][texture.index()]["extensions"]["KHR_texture_basisu"]["source"];
    let images = root["images"].as_array().map_or(0, Vec::len);
    let image = match (basisu.as_u64(), texture.source()) {
        (Some(image), _) if (image as usize) < images => image as usize,
        (Some(_), _) => {
            return Err(ImportError::new(
                "invalid_texture",
                format!(
                    "$.textures[{}].extensions.KHR_texture_basisu.source",
                    texture.index()
                ),
                "the image is outside the image array",
            ));
        }
        (None, Some(image)) => image.index(),
        (None, None) => {
            return Err(unsupported(
                format!("$.textures[{}]", texture.index()),
                "textures need an image or a KHR_texture_basisu image",
            ));
        }
    };
    let transform = &info["extensions"]["KHR_texture_transform"];
    let tex_coord = transform["texCoord"]
        .as_u64()
        .or_else(|| info["texCoord"].as_u64())
        .unwrap_or(0);
    if tex_coord > 1 {
        return Err(unsupported(
            format!("{path}.texCoord"),
            "maps read TEXCOORD_0 or TEXCOORD_1",
        ));
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
