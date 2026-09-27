//! Authored materials: color, surface and opacity, and from version 8 the
//! glTF 2.0 metallic-roughness texture maps (decision 031).

use engine_core::{
    AlphaMode, Filter, Material, Sampler, TextureHandle, TextureRef, UvTransform, Wrap,
};
use serde::{Deserialize, Serialize};

use crate::{SceneError, SceneSurface};

/// One named material; color channels are linear values in `[0, 1]` and
/// alpha is 1. `opacity` below 1 (version 6) draws it transparent unless
/// `alpha_mode` says otherwise. Version 8 adds glTF's texture maps, emissive
/// color, alpha modes and double-sided surfaces (decision 031); `surface`
/// roughness and metalness are glTF's roughness and metallic factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneMaterial {
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub id: String,
    pub color: [f32; 4],
    /// Base color texture: its sRGB color multiplies `color`. A texture ID,
    /// or (version 8) a map with sampler and transform settings. IDs clamp at
    /// the texture's edges before version 8 and repeat from version 8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<SceneMap>,
    /// How the surface responds to light beyond its color (decision 019).
    /// Requires version 5; without it the material is matte.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<SceneSurface>,
    /// The fraction of light behind it the surface covers, in `(0, 1]`
    /// (decision 026). Below 1 it draws transparent, blending in any order,
    /// writing no depth and casting no shadow. Requires version 6. With
    /// `alpha_mode` `mask` it is the alpha the cutoff tests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001, max = 1.0)))]
    pub opacity: Option<f32>,
    /// Version 8: linear green multiplies roughness and blue metalness.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallic_roughness_texture: Option<SceneMap>,
    /// Version 8: a tangent-space normal map (green up the image).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_texture: Option<SceneMap>,
    /// Version 8: scales the normal map's deflection; default 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_scale: Option<f32>,
    /// Version 8: linear red darkens indirect light.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occlusion_texture: Option<SceneMap>,
    /// Version 8: how much the occlusion map darkens, in `[0, 1]`; default 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub occlusion_strength: Option<f32>,
    /// Version 8: emitted linear color, not negative (glTF's emissive factor
    /// times its emissive strength).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emissive: Option<[f32; 3]>,
    /// Version 8: sRGB color multiplying `emissive`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emissive_texture: Option<SceneMap>,
    /// Version 8: how alpha (opacity times base color texture alpha) is
    /// used. Default `blend` below opacity 1, `opaque` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha_mode: Option<SceneAlphaMode>,
    /// Version 8, `mask` only: alpha below it is not drawn; default 0.5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0)))]
    pub alpha_cutoff: Option<f32>,
    /// Version 8: draw back faces too, lit from their own side.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub double_sided: Option<bool>,
}

/// A material's use of a texture: its ID, or (version 8) a map with glTF
/// sampler and `KHR_texture_transform` settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum SceneMap {
    Id(String),
    Map(SceneTextureMap),
}

impl SceneMap {
    /// The referenced texture ID.
    pub fn id(&self) -> &str {
        match self {
            Self::Id(id) => id,
            Self::Map(map) => &map.id,
        }
    }
}

impl From<&str> for SceneMap {
    fn from(id: &str) -> Self {
        Self::Id(id.to_owned())
    }
}

/// A texture map with its sampling and placement. Omitted settings take
/// glTF's defaults: repeating, trilinear, untransformed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTextureMap {
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wrap_u: Option<SceneWrap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wrap_v: Option<SceneWrap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mag_filter: Option<SceneFilter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_filter: Option<SceneFilter>,
    /// Filtering between mip levels; `none` samples the full-size texture
    /// only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mipmap_filter: Option<SceneMipmapFilter>,
    /// Texture coordinates are scaled, rotated by `rotation` radians, then
    /// offset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<[f32; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneWrap {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneFilter {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneMipmapFilter {
    None,
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneAlphaMode {
    Opaque,
    Mask,
    Blend,
}

/// Map fields in [`Material::maps`] order.
const MAP_FIELDS: [&str; 5] = [
    "texture",
    "metallic_roughness_texture",
    "normal_texture",
    "occlusion_texture",
    "emissive_texture",
];

/// A validated material without texture handles, which exist only once its
/// scene's textures are resident.
pub(crate) struct PreparedMaterial {
    material: Material,
    maps: [Option<(String, Sampler, UvTransform)>; 5],
    normal_scale: f32,
    occlusion_strength: f32,
}

impl PreparedMaterial {
    /// The material with its maps' textures, `texture` resolving IDs.
    pub(crate) fn material(&self, texture: impl Fn(&str) -> TextureHandle) -> Material {
        let [base, packed, normal, occlusion, emissive] = self.maps.each_ref().map(|map| {
            map.as_ref().map(|(id, sampler, transform)| {
                TextureRef::new(texture(id))
                    .with_sampler(*sampler)
                    .with_transform(*transform)
            })
        });
        let mut material = self.material;
        // Transforms and parameters were validated with the material.
        let valid = "prepared material maps are valid";
        if let Some(map) = base {
            material = material.with_base_color_texture(map).expect(valid);
        }
        if let Some(map) = packed {
            material = material.with_metallic_roughness_texture(map).expect(valid);
        }
        if let Some(map) = normal {
            material = material
                .with_normal_texture(map, self.normal_scale)
                .expect(valid);
        }
        if let Some(map) = occlusion {
            material = material
                .with_occlusion_texture(map, self.occlusion_strength)
                .expect(valid);
        }
        if let Some(map) = emissive {
            material = material.with_emissive_texture(map).expect(valid);
        }
        material
    }
}

impl SceneMaterial {
    /// Map fields and the texture IDs they reference.
    pub(crate) fn texture_ids(&self) -> impl Iterator<Item = (&'static str, &str)> {
        self.maps()
            .into_iter()
            .zip(MAP_FIELDS)
            .filter_map(|(map, field)| map.map(|map| (field, map.id())))
    }

    fn maps(&self) -> [Option<&SceneMap>; 5] {
        [
            self.texture.as_ref(),
            self.metallic_roughness_texture.as_ref(),
            self.normal_texture.as_ref(),
            self.occlusion_texture.as_ref(),
            self.emissive_texture.as_ref(),
        ]
    }

    /// The first field needing version 8, if any.
    pub(crate) fn version_eight_field(&self) -> Option<&'static str> {
        if matches!(self.texture, Some(SceneMap::Map(_))) {
            return Some("texture");
        }
        [
            (
                "metallic_roughness_texture",
                self.metallic_roughness_texture.is_some(),
            ),
            ("normal_texture", self.normal_texture.is_some()),
            ("normal_scale", self.normal_scale.is_some()),
            ("occlusion_texture", self.occlusion_texture.is_some()),
            ("occlusion_strength", self.occlusion_strength.is_some()),
            ("emissive", self.emissive.is_some()),
            ("emissive_texture", self.emissive_texture.is_some()),
            ("alpha_mode", self.alpha_mode.is_some()),
            ("alpha_cutoff", self.alpha_cutoff.is_some()),
            ("double_sided", self.double_sided.is_some()),
        ]
        .into_iter()
        .find_map(|(field, set)| set.then_some(field))
    }

    /// The material at authored path `path` (`$.materials[i]`) of a
    /// document of `version`.
    pub(crate) fn prepare(&self, path: &str, version: u32) -> Result<PreparedMaterial, SceneError> {
        let invalid = |field: &str, message: String| {
            SceneError::new("invalid_material", format!("{path}.{field}"), message)
        };
        let mut material =
            Material::new(self.color).map_err(|error| invalid("color", error.to_string()))?;
        if let Some(opacity) = self.opacity {
            material = material
                .with_opacity(opacity)
                .map_err(|error| invalid("opacity", error.to_string()))?;
        }
        if let Some(mode) = self.alpha_mode {
            if self.alpha_cutoff.is_some() && mode != SceneAlphaMode::Mask {
                return Err(invalid(
                    "alpha_cutoff",
                    "an alpha cutoff needs alpha_mode mask".into(),
                ));
            }
            let mode = match mode {
                SceneAlphaMode::Opaque if self.opacity.is_some_and(|opacity| opacity < 1.0) => {
                    return Err(invalid(
                        "alpha_mode",
                        "opaque materials have opacity 1; use blend or mask".into(),
                    ));
                }
                SceneAlphaMode::Opaque => AlphaMode::Opaque,
                SceneAlphaMode::Mask => AlphaMode::Mask {
                    cutoff: self.alpha_cutoff.unwrap_or(0.5),
                },
                SceneAlphaMode::Blend => AlphaMode::Blend,
            };
            material = material
                .with_alpha_mode(mode)
                .map_err(|error| invalid("alpha_cutoff", error.to_string()))?;
        } else if self.alpha_cutoff.is_some() {
            return Err(invalid(
                "alpha_cutoff",
                "an alpha cutoff needs alpha_mode mask".into(),
            ));
        }
        if let Some(surface) = self.surface {
            material = material
                .with_surface(surface.surface())
                .map_err(|error| invalid("surface", error.to_string()))?;
        }
        if let Some(emissive) = self.emissive {
            material = material
                .with_emissive(emissive)
                .map_err(|error| invalid("emissive", error.to_string()))?;
        }
        if let Some(double_sided) = self.double_sided {
            material = material.with_double_sided(double_sided);
        }
        let normal_scale = self.normal_scale.unwrap_or(1.0);
        if !normal_scale.is_finite() {
            return Err(invalid(
                "normal_scale",
                "the normal scale must be finite".into(),
            ));
        }
        let occlusion_strength = self.occlusion_strength.unwrap_or(1.0);
        if !occlusion_strength.is_finite() || !(0.0..=1.0).contains(&occlusion_strength) {
            return Err(invalid(
                "occlusion_strength",
                "the occlusion strength must be in [0, 1]".into(),
            ));
        }
        let mut maps: [Option<(String, Sampler, UvTransform)>; 5] = Default::default();
        for ((slot, map), field) in maps.iter_mut().zip(self.maps()).zip(MAP_FIELDS) {
            let Some(map) = map else {
                continue;
            };
            *slot = Some(match map {
                // Plain IDs clamp before version 8 (their original meaning)
                // and take glTF's defaults from version 8.
                SceneMap::Id(id) => (
                    id.clone(),
                    if version < 8 {
                        Sampler::CLAMP
                    } else {
                        Sampler::REPEAT
                    },
                    UvTransform::IDENTITY,
                ),
                SceneMap::Map(map) => {
                    let transform = UvTransform {
                        offset: map.offset.unwrap_or([0.0; 2]),
                        rotation: map.rotation.unwrap_or(0.0),
                        scale: map.scale.unwrap_or([1.0; 2]),
                    };
                    if !transform
                        .offset
                        .iter()
                        .chain(&transform.scale)
                        .chain([&transform.rotation])
                        .all(|value| value.is_finite())
                    {
                        return Err(invalid(
                            field,
                            "texture offset, rotation and scale must be finite".into(),
                        ));
                    }
                    (map.id.clone(), map.sampler(), transform)
                }
            });
        }
        Ok(PreparedMaterial {
            material,
            maps,
            normal_scale,
            occlusion_strength,
        })
    }
}

impl SceneTextureMap {
    fn sampler(&self) -> Sampler {
        let wrap = |wrap: Option<SceneWrap>| match wrap.unwrap_or(SceneWrap::Repeat) {
            SceneWrap::Repeat => Wrap::Repeat,
            SceneWrap::MirroredRepeat => Wrap::MirroredRepeat,
            SceneWrap::ClampToEdge => Wrap::ClampToEdge,
        };
        let filter = |filter: Option<SceneFilter>| match filter.unwrap_or(SceneFilter::Linear) {
            SceneFilter::Nearest => Filter::Nearest,
            SceneFilter::Linear => Filter::Linear,
        };
        Sampler {
            wrap_u: wrap(self.wrap_u),
            wrap_v: wrap(self.wrap_v),
            mag_filter: filter(self.mag_filter),
            min_filter: filter(self.min_filter),
            mipmap_filter: match self.mipmap_filter.unwrap_or(SceneMipmapFilter::Linear) {
                SceneMipmapFilter::None => None,
                SceneMipmapFilter::Nearest => Some(Filter::Nearest),
                SceneMipmapFilter::Linear => Some(Filter::Linear),
            },
        }
    }
}
