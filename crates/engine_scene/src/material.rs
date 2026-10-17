//! Authored materials: color, surface and opacity, and the glTF 2.0
//! metallic-roughness texture maps.

use engine_core::{
    AlphaMode, Filter, Material, Sampler, TextureHandle, TextureRef, Transmission, UvTransform,
    Wrap,
};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};

use crate::{SceneAlphaCoverage, SceneError, SceneSurface};

/// One named material; color channels are linear values in `[0, 1]` and
/// alpha is 1. `opacity` below 1 draws it transparent unless `alpha_mode`
/// says otherwise. glTF's texture maps, emissive color, alpha modes and
/// double-sided surfaces complete it; `surface`
/// roughness and metalness are glTF's roughness and metallic factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(transform = state_material_rules)
)]
#[serde(deny_unknown_fields)]
pub struct SceneMaterial {
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::rule::id()))
    )]
    pub id: String,
    pub color: [f32; 4],
    /// Base color texture: its sRGB color multiplies `color`. A texture ID,
    /// which repeats, or a map with sampler and transform settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<SceneMap>,
    /// How the surface responds to light beyond its color; without it the
    /// material is matte.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<SceneSurface>,
    /// The fraction of light behind it the surface covers, in `(0, 1]`.
    /// Below 1 it draws transparent, blending in any order,
    /// writing no depth and casting no shadow. With
    /// `alpha_mode` `mask` it is the alpha the cutoff tests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f32>,
    /// Linear green multiplies roughness and blue metalness.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallic_roughness_texture: Option<SceneMap>,
    /// A tangent-space normal map (green up the image).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_texture: Option<SceneMap>,
    /// Scales the normal map's deflection; default 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_scale: Option<f32>,
    /// Linear red darkens indirect light.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occlusion_texture: Option<SceneMap>,
    /// How much the occlusion map darkens, in `[0, 1]`; default 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occlusion_strength: Option<f32>,
    /// Emitted linear color, not negative (glTF's emissive factor
    /// times its emissive strength).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emissive: Option<[f32; 3]>,
    /// SRGB color multiplying `emissive`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emissive_texture: Option<SceneMap>,
    /// How alpha (opacity times base color texture alpha) is
    /// used. Default `blend` below opacity 1, `opaque` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha_mode: Option<SceneAlphaMode>,
    /// With `mask` only: alpha below it is not drawn; default 0.5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha_cutoff: Option<f32>,
    /// Alpha-mask mip filtering: default `auto` prepares a texture/cutoff
    /// curve for power-of-two RGBA8 UV maps with linear min/mag/mip filters,
    /// cutoff in (0, 1) and opacity 1. Other cases, non-unit combined
    /// instance/vertex opacity and uniform mip tails use the original
    /// `1 + 0.25 * mip` rule. `legacy` always selects that rule. Camera,
    /// shadows and source impostor bakes agree; raw alpha and final atlases
    /// are unchanged. Inspect AlphaCoverage for effective behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha_coverage: Option<SceneAlphaCoverage>,
    /// Draw back faces too, lit from their own side.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub double_sided: Option<bool>,
    /// The share of light passing through the surface instead of
    /// diffusing off it, in `[0, 1]`, tinted by its color (
    /// glTF `KHR_materials_transmission`). Transmissive materials draw in
    /// the transparent pass; the environment's `refraction` bends what shows
    /// through them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transmission: Option<f32>,
    /// With `transmission`: the index of refraction in `[1, 3]`;
    /// default 1.5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ior: Option<f32>,
    /// With `transmission`: the depth in world units of the
    /// volume beneath the surface that bends rays; default 0, a thin wall.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thickness: Option<f32>,
    /// Lays the maps from the world's axes, one repeat every
    /// this many meters, instead of by texture coordinates:
    /// no seams and one texel density at any size, for surfaces that stay
    /// where they are, such as rocks. In `(0, 10000]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_tiling: Option<f32>,
    /// A layer settling on faces turned up, such as moss,
    /// lichen, dust or snow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top: Option<SceneTopLayer>,
    /// The ground staining the base of objects terrains place.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grounding: Option<SceneGrounding>,
}

/// States the rules the material's builders check on the authored fields
/// they take.
#[cfg(feature = "schema")]
fn state_material_rules(schema: &mut schemars::Schema) {
    let part = |rule: Option<engine_settings::Rule>| rule.expect("the part declares the field");
    for (field, rule) in [
        ("color", Material::COLOR),
        ("opacity", Material::OPACITY),
        ("normal_scale", Material::NORMAL_SCALE),
        ("occlusion_strength", Material::OCCLUSION_STRENGTH),
        ("emissive", Material::EMISSIVE),
        ("alpha_cutoff", Material::ALPHA_CUTOFF),
        ("world_tiling", Material::WORLD_TILING),
        ("transmission", part(Transmission::rule_of("factor"))),
        ("ior", part(Transmission::rule_of("ior"))),
        ("thickness", part(Transmission::rule_of("thickness"))),
    ] {
        engine_settings::state(schema, field, rule);
    }
}

/// The ground staining the base of an object standing on terrain:
/// the linear color its color is multiplied by where it meets the
/// ground, and how many units of its mesh up the stain fades out, from the
/// ground's slope under it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(transform = engine_core::Grounding::state_rules)
)]
#[serde(deny_unknown_fields)]
pub struct SceneGrounding {
    pub color: [f32; 3],
    pub height: f32,
}

/// A layer settling on a material's faces turned up: its
/// linear color, how far from straight up it reaches (0 covers nothing, 1
/// every face turned up at all) and its roughness (default 0.85).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(transform = engine_core::TopLayer::state_rules)
)]
#[serde(deny_unknown_fields)]
pub struct SceneTopLayer {
    pub color: [f32; 3],
    pub coverage: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roughness: Option<f32>,
}

/// A material's use of a texture: its ID, or a map with glTF sampler and
/// `KHR_texture_transform` settings.
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
        schemars(transform = engine_settings::stated(engine_settings::rule::id()))
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
    /// The texture coordinates the map reads: 0, the mesh's first (the
    /// default), or 1, its second.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tex_coord: Option<u32>,
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
    maps: [Option<(String, Sampler, UvTransform, u32)>; 5],
    normal_scale: f32,
    occlusion_strength: f32,
}

impl PreparedMaterial {
    /// The material with its maps' textures, `texture` resolving IDs.
    pub(crate) fn material(&self, texture: impl Fn(&str) -> TextureHandle) -> Material {
        let [base, packed, normal, occlusion, emissive] = self.maps.each_ref().map(|map| {
            map.as_ref().map(|(id, sampler, transform, tex_coord)| {
                TextureRef::new(texture(id))
                    .with_sampler(*sampler)
                    .with_transform(*transform)
                    .with_tex_coord(*tex_coord)
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
    /// `material` as a scene authors it, under ID `id`, its textures by
    /// the IDs `texture` gives them: every setting stated, so what
    /// draws reads back whole and copies into a scene as is.
    pub fn from_runtime(
        id: impl Into<String>,
        material: &Material,
        texture: impl Fn(TextureHandle) -> String,
    ) -> Self {
        let map = |map: Option<TextureRef>| {
            map.map(|map| SceneMap::Map(SceneTextureMap::from_runtime(texture(map.texture), &map)))
        };
        let [red, green, blue, opacity] = material.color();
        let surface = material.surface();
        let transmission = material.transmission();
        let (alpha_mode, alpha_cutoff) = match material.alpha_mode() {
            AlphaMode::Opaque => (SceneAlphaMode::Opaque, None),
            AlphaMode::Mask { cutoff } => (SceneAlphaMode::Mask, Some(cutoff)),
            AlphaMode::Blend => (SceneAlphaMode::Blend, None),
        };
        Self {
            id: id.into(),
            color: [red, green, blue, 1.0],
            texture: map(material.base_color_texture()),
            surface: Some(SceneSurface {
                roughness: surface.roughness,
                metalness: surface.metalness,
                emission: surface.emission,
                ripples: surface.ripples,
            }),
            opacity: Some(opacity),
            metallic_roughness_texture: map(material.metallic_roughness_texture()),
            normal_texture: map(material.normal_texture()),
            normal_scale: Some(material.normal_scale()),
            occlusion_texture: map(material.occlusion_texture()),
            occlusion_strength: Some(material.occlusion_strength()),
            emissive: Some(material.emissive()),
            emissive_texture: map(material.emissive_texture()),
            alpha_mode: Some(alpha_mode),
            alpha_cutoff,
            alpha_coverage: Some(material.alpha_coverage().into()),
            double_sided: Some(material.is_double_sided()),
            transmission: Some(transmission.factor),
            ior: Some(transmission.ior),
            thickness: Some(transmission.thickness),
            world_tiling: material.world_tiling(),
            top: material.top_layer().map(|top| SceneTopLayer {
                color: top.color,
                coverage: top.coverage,
                roughness: Some(top.roughness),
            }),
            grounding: material.grounding().map(|grounding| SceneGrounding {
                color: grounding.color,
                height: grounding.height,
            }),
        }
    }

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

    /// The material at authored path `path` (`$.materials[i]`).
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn prepare(&self, path: &str) -> Result<PreparedMaterial, SceneError> {
        let invalid = |field: &str, message: String| {
            SceneError::new(
                ErrorCode::InvalidMaterial,
                format!("{path}.{field}"),
                message,
            )
        };
        let mut material = Material::new(self.color)
            .map_err(|error| invalid("color", error.to_string()))?
            .with_alpha_coverage(self.alpha_coverage.unwrap_or_default().into());
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
        if let Some(tiling) = self.world_tiling {
            material = material
                .with_world_mapping(tiling)
                .map_err(|error| invalid("world_tiling", error.to_string()))?;
        }
        if let Some(top) = self.top {
            material = material
                .with_top_layer(engine_core::TopLayer {
                    color: top.color,
                    coverage: top.coverage,
                    roughness: top.roughness.unwrap_or(0.85),
                })
                .map_err(|error| invalid("top", error.to_string()))?;
        }
        if let Some(grounding) = self.grounding {
            material = material
                .with_grounding(engine_core::Grounding {
                    color: grounding.color,
                    height: grounding.height,
                })
                .map_err(|error| invalid("grounding", error.to_string()))?;
        }
        match (self.transmission, self.ior, self.thickness) {
            (Some(factor), ior, thickness) => {
                material = material
                    .with_transmission(Transmission {
                        factor,
                        ior: ior.unwrap_or(Transmission::NONE.ior),
                        thickness: thickness.unwrap_or(0.0),
                    })
                    .map_err(|error| invalid("transmission", error.to_string()))?;
            }
            (None, Some(_), _) => {
                return Err(invalid(
                    "ior",
                    "an index of refraction needs transmission".into(),
                ));
            }
            (None, None, Some(_)) => {
                return Err(invalid(
                    "thickness",
                    "a thickness needs transmission".into(),
                ));
            }
            (None, None, None) => {}
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
        let mut maps: [Option<(String, Sampler, UvTransform, u32)>; 5] = Default::default();
        for ((slot, map), field) in maps.iter_mut().zip(self.maps()).zip(MAP_FIELDS) {
            let Some(map) = map else {
                continue;
            };
            *slot = Some(match map {
                // Plain IDs take glTF's defaults.
                SceneMap::Id(id) => (id.clone(), Sampler::REPEAT, UvTransform::IDENTITY, 0),
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
                    let tex_coord = map.tex_coord.unwrap_or(0);
                    if tex_coord > 1 {
                        return Err(invalid(
                            field,
                            "maps read texture coordinates 0 or 1".into(),
                        ));
                    }
                    (map.id.clone(), map.sampler(), transform, tex_coord)
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
    /// `map` of texture `id` with every sampling and placement setting
    /// stated.
    pub fn from_runtime(id: String, map: &TextureRef) -> Self {
        let wrap = |wrap: Wrap| match wrap {
            Wrap::Repeat => SceneWrap::Repeat,
            Wrap::MirroredRepeat => SceneWrap::MirroredRepeat,
            Wrap::ClampToEdge => SceneWrap::ClampToEdge,
        };
        let filter = |filter: Filter| match filter {
            Filter::Nearest => SceneFilter::Nearest,
            Filter::Linear => SceneFilter::Linear,
        };
        let sampler = map.sampler;
        Self {
            id,
            wrap_u: Some(wrap(sampler.wrap_u)),
            wrap_v: Some(wrap(sampler.wrap_v)),
            mag_filter: Some(filter(sampler.mag_filter)),
            min_filter: Some(filter(sampler.min_filter)),
            mipmap_filter: Some(match sampler.mipmap_filter {
                None => SceneMipmapFilter::None,
                Some(Filter::Nearest) => SceneMipmapFilter::Nearest,
                Some(Filter::Linear) => SceneMipmapFilter::Linear,
            }),
            offset: Some(map.transform.offset),
            rotation: Some(map.transform.rotation),
            scale: Some(map.transform.scale),
            tex_coord: Some(map.tex_coord),
        }
    }

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
