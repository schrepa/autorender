use std::io::Cursor;

use base64::{Engine, engine::general_purpose::STANDARD};
use engine_scene::{SceneDocument, SceneMaterial, SceneTexture};
use gltf::texture::{MagFilter, MinFilter, WrappingMode};

use crate::{ImportError, reserve_bytes, unsupported};

pub(crate) fn import(
    document: &gltf::Document,
    binary: &[u8],
    id: &str,
    scene: &mut SceneDocument,
    resident_bytes: &mut usize,
) -> Result<(), ImportError> {
    for sampler in document.samplers() {
        if sampler.wrap_s() != WrappingMode::ClampToEdge
            || sampler.wrap_t() != WrappingMode::ClampToEdge
            || sampler
                .mag_filter()
                .is_some_and(|filter| filter != MagFilter::Linear)
            // The renderer samples every texture trilinearly from mip levels
            // it generates, which LINEAR_MIPMAP_LINEAR states explicitly.
            || sampler.min_filter().is_some_and(|filter| {
                !matches!(filter, MinFilter::Linear | MinFilter::LinearMipmapLinear)
            })
        {
            return Err(unsupported(
                format!("$.samplers[{}]", sampler.index().unwrap_or(0)),
                "only CLAMP_TO_EDGE in both axes and LINEAR filters (or unspecified filters) are supported; the minification filter may also be LINEAR_MIPMAP_LINEAR",
            ));
        }
    }
    for texture in document.textures() {
        // A missing sampler defaults to REPEAT in glTF and cannot silently be
        // treated as clamp by our renderer.
        if texture.sampler().index().is_none() {
            return Err(unsupported(
                format!("$.textures[{}].sampler", texture.index()),
                "a sampler explicitly declaring CLAMP_TO_EDGE is required",
            ));
        }
    }
    for image in document.images() {
        let path = format!("$.images[{}]", image.index());
        let gltf::image::Source::View { view, mime_type } = image.source() else {
            return Err(unsupported(
                path,
                "only embedded PNG images are supported; external and data URIs are rejected",
            ));
        };
        if mime_type != "image/png" || view.stride().is_some() {
            return Err(unsupported(
                path,
                "images must be contiguous embedded image/png buffer views",
            ));
        }
        let bytes = &binary[view.offset()..view.offset() + view.length()];
        let (width, height, rgba) = decode_png(bytes, &path, resident_bytes)?;
        scene.textures.push(SceneTexture {
            id: format!("{id}.image{}", image.index()),
            width,
            height,
            rgba8: STANDARD.encode(rgba),
        });
    }
    for material in document.materials() {
        let index = material.index().expect("enumerated materials have indices");
        let path = format!("$.materials[{index}]");
        let pbr = material.pbr_metallic_roughness();
        if material.alpha_mode() != gltf::material::AlphaMode::Opaque
            || material.double_sided()
            || pbr.base_color_factor()[3] != 1.0
        {
            return Err(unsupported(
                &path,
                "only opaque single-sided materials with base color alpha 1 are supported",
            ));
        }
        if pbr.metallic_factor() != 0.0
            || pbr.roughness_factor() != 1.0
            || pbr.metallic_roughness_texture().is_some()
            || material.normal_texture().is_some()
            || material.occlusion_texture().is_some()
            || material.emissive_texture().is_some()
            || material.emissive_factor() != [0.0; 3]
        {
            return Err(unsupported(
                &path,
                "the diffuse profile requires explicit metallicFactor 0, roughnessFactor 1, no emissive factor and no textures beyond baseColorTexture; engine lighting is not glTF PBR",
            ));
        }
        let texture = pbr
            .base_color_texture()
            .map(|info| {
                if info.tex_coord() != 0 {
                    return Err(unsupported(
                        format!("{path}.pbrMetallicRoughness.baseColorTexture.texCoord"),
                        "only TEXCOORD_0 is supported",
                    ));
                }
                Ok(format!("{id}.image{}", info.texture().source().index()))
            })
            .transpose()?;
        scene.materials.push(SceneMaterial {
            id: format!("{id}.material{index}"),
            color: pbr.base_color_factor(),
            texture,
        });
    }
    Ok(())
}

fn decode_png(
    bytes: &[u8],
    path: &str,
    resident_bytes: &mut usize,
) -> Result<(u32, u32, Vec<u8>), ImportError> {
    let invalid = |message: String| ImportError::new("invalid_png", path, message);
    // Read the fixed header first so oversized images fail before the decoder
    // allocates image storage or interprets ancillary chunks.
    if bytes.len() < 33
        || &bytes[..8] != b"\x89PNG\r\n\x1a\n"
        || &bytes[12..16] != b"IHDR"
        || bytes[8..12] != 13u32.to_be_bytes()
    {
        return Err(invalid("expected a PNG with a standard IHDR header".into()));
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().expect("four-byte width"));
    let height = u32::from_be_bytes(bytes[20..24].try_into().expect("four-byte height"));
    if width == 0 || height == 0 || width > 2048 || height > 2048 {
        return Err(ImportError::new(
            "texture_dimensions",
            path,
            "texture dimensions must be from 1 through 2048",
        ));
    }
    if bytes[24] != 8 || !matches!(bytes[25], 2 | 6) {
        return Err(unsupported(path, "only 8-bit RGB or RGBA PNG is supported"));
    }
    let rgba_bytes = width as usize * height as usize * 4;
    reserve_bytes(resident_bytes, rgba_bytes, path)?;
    let mut decoder = png::Decoder::new_with_limits(
        Cursor::new(bytes),
        png::Limits {
            bytes: 16 * 1024 * 1024,
        },
    );
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);
    let mut reader = decoder
        .read_info()
        .map_err(|error| invalid(error.to_string()))?;
    if reader.info().animation_control.is_some() {
        return Err(unsupported(path, "animated PNG is unsupported"));
    }
    let mut decoded = vec![0; reader.output_buffer_size()];
    let output = reader
        .next_frame(&mut decoded)
        .map_err(|error| invalid(error.to_string()))?;
    reader
        .finish()
        .map_err(|error| invalid(error.to_string()))?;
    decoded.truncate(output.buffer_size());
    let rgba = match output.color_type {
        png::ColorType::Rgb => {
            if reader.info().trns.is_some() {
                return Err(unsupported(
                    path,
                    "PNG transparency is unsupported, including tRNS",
                ));
            }
            let mut rgba = Vec::with_capacity(rgba_bytes);
            for rgb in decoded.as_chunks::<3>().0 {
                rgba.extend_from_slice(rgb);
                rgba.push(255);
            }
            rgba
        }
        png::ColorType::Rgba => decoded,
        _ => {
            return Err(unsupported(
                path,
                "PNG decoded to an unsupported color format",
            ));
        }
    };
    if rgba.as_chunks::<4>().0.iter().any(|pixel| pixel[3] != 255) {
        return Err(unsupported(path, "all PNG texels must have alpha 255"));
    }
    Ok((width, height, rgba))
}

#[cfg(test)]
mod tests {
    use super::decode_png;

    fn png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(rgba).unwrap();
        }
        bytes
    }

    #[test]
    fn texture_rows_and_alpha_survive_decode_without_color_conversion() {
        let pixels = [255, 0, 0, 255, 0, 0, 255, 255];
        let mut used = 0;
        let (width, height, decoded) = decode_png(&png(1, 2, &pixels), "$", &mut used).unwrap();
        assert_eq!((width, height, used), (1, 2, 8));
        assert_eq!(decoded, pixels);
        assert_eq!(
            decode_png(&png(1, 1, &[255, 0, 0, 128]), "$", &mut 0)
                .unwrap_err()
                .code,
            "unsupported_feature"
        );
    }

    #[test]
    fn dimensions_and_shared_budget_fail_before_decoded_allocation() {
        let mut bytes = png(1, 1, &[255; 4]);
        let mut used = crate::MAX_CPU_BYTES - 3;
        assert_eq!(
            decode_png(&bytes, "$", &mut used).unwrap_err().code,
            "asset_budget"
        );
        // Deliberately leave the checksum stale: a dimensions failure must occur
        // before handing an oversized header to the decoder at all.
        bytes[16..20].copy_from_slice(&2049u32.to_be_bytes());
        assert_eq!(
            decode_png(&bytes, "$", &mut 0).unwrap_err().code,
            "texture_dimensions"
        );
    }

    #[test]
    fn damaged_png_checksums_are_reported_as_structured_errors() {
        let mut bytes = png(1, 1, &[255; 4]);
        bytes[29] ^= 1;
        assert_eq!(
            decode_png(&bytes, "$", &mut 0).unwrap_err().code,
            "invalid_png"
        );
    }
}
