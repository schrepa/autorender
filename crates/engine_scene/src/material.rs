use std::collections::BTreeMap;

use bevy_ecs::system::{Res, ResMut, SystemParamItem};
use engine_core::{
    AlphaMode, AssetLibrary, ColorSpace, Cull, DepthBias, Error, Filter, LightingModel, MAPPED,
    MAPPED_MAPS, MAPPED_SLOTS, Material, Operation, Sampler, SceneContext, SurfaceSlot,
    TextureHandle, TextureRef, UvTransform, Wrap,
};
use engine_settings::{ErrorCode, Expected};
use serde::{Deserialize, Serialize};

use crate::SceneSurface;
use crate::validation::{identifier, moved};

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
        schemars(transform = engine_settings::stated(engine_settings::fields::id()))
    )]
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::fields::id()))
    )]
    pub provided: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<[f32; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<SceneSurface>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_scale: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occlusion_strength: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emissive: Option<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha_mode: Option<SceneAlphaMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha_cutoff: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cull: Option<SceneCull>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_test: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_write: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_bias: Option<SceneDepthBias>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transmission: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slots: Option<BTreeMap<String, SceneSlot>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<BTreeMap<String, f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lighting: Option<String>,
}

#[cfg(feature = "schema")]
fn state_material_rules(schema: &mut schemars::Schema) {
    for (field, rule) in [
        ("color", Material::COLOR),
        ("opacity", Material::OPACITY),
        ("normal_scale", Material::NORMAL_SCALE),
        ("occlusion_strength", Material::OCCLUSION_STRENGTH),
        ("emissive", Material::EMISSIVE),
        ("alpha_cutoff", Material::ALPHA_CUTOFF),
        ("transmission", Material::TRANSMISSION),
    ] {
        engine_settings::state(schema, field, rule);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneCull {
    Back,
    Front,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneDepthBias {
    pub constant: i32,
    pub slope: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum SceneSlot {
    Module(String),
    Filled(SceneSlotFill),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneSlotFill {
    pub module: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub maps: BTreeMap<String, SceneMap>,
}

impl SceneSlot {
    pub fn module(&self) -> &str {
        match self {
            Self::Module(module) => module,
            Self::Filled(fill) => &fill.module,
        }
    }

    pub fn maps(&self) -> impl Iterator<Item = (&str, &SceneMap)> {
        let maps = match self {
            Self::Module(_) => None,
            Self::Filled(fill) => Some(&fill.maps),
        };
        maps.into_iter()
            .flatten()
            .map(|(name, map)| (name.as_str(), map))
    }
}

impl From<&str> for SceneSlot {
    fn from(module: &str) -> Self {
        Self::Module(module.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum SceneMap {
    Id(String),
    Map(SceneTextureMap),
}

impl SceneMap {
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTextureMap {
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::fields::id()))
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mipmap_filter: Option<SceneMipmapFilter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<[f32; 2]>,
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
    Additive,
}

type PreparedMap = (String, String, Sampler, UvTransform, u32);

pub(crate) struct PreparedMaterial {
    material: Material,
    maps: Vec<PreparedMap>,
}

impl PreparedMaterial {
    pub(crate) fn material(&self, texture: impl Fn(&str) -> TextureHandle) -> Material {
        let mut material = self.material.clone();
        for (name, id, sampler, transform, tex_coord) in &self.maps {
            let map = TextureRef::new(texture(id))
                .with_sampler(*sampler)
                .with_transform(*transform)
                .with_tex_coord(*tex_coord);
            material = material
                .with_map(name.as_str(), map)
                .expect("prepared_maps_valid");
        }
        material
    }
}

fn home_slot(material: &Material, map: &str) -> Option<SurfaceSlot> {
    let mapped = |slot: SurfaceSlot| {
        MAPPED_SLOTS
            .iter()
            .any(|(known, maps)| *known == slot && maps.contains(&map))
    };
    let slots = material.slots();
    slots
        .iter()
        .find(|(slot, module)| module == MAPPED && mapped(*slot))
        .or_else(|| slots.iter().find(|(_, module)| module != MAPPED))
        .or_else(|| slots.first())
        .map(|(slot, _)| *slot)
}

impl SceneMaterial {
    pub fn from_runtime(
        id: impl Into<String>,
        material: &Material,
        texture: impl Fn(TextureHandle) -> String,
    ) -> Self {
        let mut fills: BTreeMap<String, SceneSlotFill> = material
            .slots()
            .iter()
            .map(|(slot, module)| {
                (
                    slot.name().to_owned(),
                    SceneSlotFill {
                        module: module.clone(),
                        maps: BTreeMap::new(),
                    },
                )
            })
            .collect();
        for (name, map) in material.maps() {
            let fill = home_slot(material, name).and_then(|slot| fills.get_mut(slot.name()));
            if let Some(fill) = fill {
                fill.maps.insert(
                    name.clone(),
                    SceneMap::Map(SceneTextureMap::from_runtime(texture(map.texture), map)),
                );
            }
        }
        let [red, green, blue, opacity] = material.color();
        let surface = material.surface();
        let state = material.render_state();
        let (alpha_mode, alpha_cutoff) = match state.alpha {
            AlphaMode::Opaque => (SceneAlphaMode::Opaque, None),
            AlphaMode::Mask { cutoff } => (SceneAlphaMode::Mask, Some(cutoff)),
            AlphaMode::Blend => (SceneAlphaMode::Blend, None),
            AlphaMode::Additive => (SceneAlphaMode::Additive, None),
        };
        Self {
            id: id.into(),
            provided: None,
            color: Some([red, green, blue, 1.0]),
            surface: Some(SceneSurface {
                roughness: surface.roughness,
                metalness: surface.metalness,
                emission: surface.emission,
            }),
            opacity: Some(opacity),
            normal_scale: Some(material.normal_scale()),
            occlusion_strength: Some(material.occlusion_strength()),
            emissive: Some(material.emissive()),
            alpha_mode: Some(alpha_mode),
            alpha_cutoff,
            cull: Some(match state.cull {
                Cull::Back => SceneCull::Back,
                Cull::Front => SceneCull::Front,
                Cull::None => SceneCull::None,
            }),
            depth_test: Some(state.depth_test),
            depth_write: Some(state.depth_write),
            depth_bias: Some(SceneDepthBias {
                constant: state.bias.constant,
                slope: state.bias.slope,
            }),
            transmission: Some(material.transmission()),
            slots: (!fills.is_empty()).then(|| {
                fills
                    .into_iter()
                    .map(|(slot, fill)| {
                        let slot_fill = if fill.maps.is_empty() {
                            SceneSlot::Module(fill.module)
                        } else {
                            SceneSlot::Filled(fill)
                        };
                        (slot, slot_fill)
                    })
                    .collect()
            }),
            values: (!material.values().is_empty())
                .then(|| material.values().iter().cloned().collect()),
            lighting: Some(match material.lighting() {
                LightingModel::Lit => "lit".to_owned(),
                LightingModel::Unlit => "unlit".to_owned(),
                LightingModel::Named(name) => name.clone(),
            }),
        }
    }

    fn maps(&self) -> impl Iterator<Item = (&str, &str, &SceneMap)> {
        self.slots.iter().flatten().flat_map(|(slot, fill)| {
            fill.maps()
                .map(move |(name, map)| (slot.as_str(), name, map))
        })
    }

    pub(crate) fn texture_ids(&self) -> impl Iterator<Item = (String, &str)> {
        self.maps()
            .map(|(slot, name, map)| (format!("slots.{slot}.maps.{name}"), map.id()))
    }

    pub(crate) fn map_spaces(&self) -> impl Iterator<Item = (&str, ColorSpace)> + '_ {
        self.maps().map(|(_, name, map)| {
            let space = MAPPED_MAPS
                .iter()
                .find(|(known, _)| *known == name)
                .map_or(ColorSpace::Linear, |(_, space)| *space);
            (map.id(), space)
        })
    }

    pub(crate) fn prepare(&self, path: &str) -> Result<PreparedMaterial, Error> {
        match (&self.provided, self.color) {
            (Some(name), _) => self.prepare_provided(name, path),
            (None, Some(color)) => self.prepare_authored(color, path),
            (None, None) => Err(Error::new(ErrorCode::MissingField, format!("{path}.color"))
                .expected(Expected::Present)),
        }
    }

    fn prepare_provided(&self, name: &str, path: &str) -> Result<PreparedMaterial, Error> {
        identifier(name, &format!("{path}.provided"))?;
        let given = [
            ("color", self.color.is_some()),
            ("surface", self.surface.is_some()),
            ("opacity", self.opacity.is_some()),
            ("normal_scale", self.normal_scale.is_some()),
            ("occlusion_strength", self.occlusion_strength.is_some()),
            ("emissive", self.emissive.is_some()),
            ("alpha_mode", self.alpha_mode.is_some()),
            ("alpha_cutoff", self.alpha_cutoff.is_some()),
            ("cull", self.cull.is_some()),
            ("depth_test", self.depth_test.is_some()),
            ("depth_write", self.depth_write.is_some()),
            ("depth_bias", self.depth_bias.is_some()),
            ("transmission", self.transmission.is_some()),
            ("slots", self.slots.is_some()),
            ("values", self.values.is_some()),
            ("lighting", self.lighting.is_some()),
        ];
        if let Some((field, _)) = given.into_iter().find(|&(_, given)| given) {
            return Err(
                Error::new(ErrorCode::InvalidMaterial, format!("{path}.{field}"))
                    .expected(Expected::Absent),
            );
        }
        Ok(PreparedMaterial {
            material: Material::new([1.0; 4]).expect("white_material_valid"),
            maps: Vec::new(),
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn prepare_authored(&self, color: [f32; 4], path: &str) -> Result<PreparedMaterial, Error> {
        let invalid =
            |field: &str| Error::new(ErrorCode::InvalidMaterial, format!("{path}.{field}"));
        let failed = |error: Error, field: &str| {
            moved(error, ErrorCode::InvalidMaterial, format!("{path}.{field}"))
        };
        let mut material = Material::new(color).map_err(|error| failed(error, "color"))?;
        if let Some(mode) = self.alpha_mode {
            if self.alpha_cutoff.is_some() && mode != SceneAlphaMode::Mask {
                return Err(invalid("alpha_cutoff").expected(Expected::Absent));
            }
            let mode = match mode {
                SceneAlphaMode::Opaque => AlphaMode::Opaque,
                SceneAlphaMode::Mask => AlphaMode::Mask {
                    cutoff: self.alpha_cutoff.unwrap_or(0.5),
                },
                SceneAlphaMode::Blend => AlphaMode::Blend,
                SceneAlphaMode::Additive => AlphaMode::Additive,
            };
            material = material
                .with_alpha_mode(mode)
                .map_err(|error| failed(error, "alpha_cutoff"))?;
        } else if self.alpha_cutoff.is_some() {
            return Err(invalid("alpha_cutoff").expected(Expected::Absent));
        }
        if let Some(opacity) = self.opacity {
            material = material
                .with_opacity(opacity)
                .map_err(|error| failed(error, "opacity"))?;
        }
        if let Some(surface) = self.surface {
            material = material
                .with_surface(surface.surface())
                .map_err(|error| failed(error, "surface"))?;
        }
        if let Some(emissive) = self.emissive {
            material = material
                .with_emissive(emissive)
                .map_err(|error| failed(error, "emissive"))?;
        }
        let mut state = material.render_state();
        if let Some(cull) = self.cull {
            state.cull = match cull {
                SceneCull::Back => Cull::Back,
                SceneCull::Front => Cull::Front,
                SceneCull::None => Cull::None,
            };
        }
        if let Some(test) = self.depth_test {
            state.depth_test = test;
        }
        if let Some(write) = self.depth_write {
            state.depth_write = write;
        }
        if let Some(bias) = self.depth_bias {
            state.bias = DepthBias {
                constant: bias.constant,
                slope: bias.slope,
            };
        }
        material = material
            .with_render_state(state)
            .map_err(|error| failed(error, "depth_bias"))?;
        if let Some(transmission) = self.transmission {
            material = material
                .with_transmission(transmission)
                .map_err(|error| failed(error, "transmission"))?;
        }
        for (name, fill) in self.slots.iter().flatten() {
            let Some(slot) = SurfaceSlot::named(name) else {
                return Err(
                    Error::new(ErrorCode::UnknownVariant, format!("{path}.slots.{name}"))
                        .variants(SurfaceSlot::ALL.map(SurfaceSlot::name))
                        .got(name.as_str()),
                );
            };
            material = material
                .with_slot(slot, fill.module())
                .map_err(|error| failed(error, &format!("slots.{name}")))?;
        }
        for (name, value) in self.values.iter().flatten() {
            material = material
                .with_value(name.as_str(), *value)
                .map_err(|error| failed(error, &format!("values.{name}")))?;
        }
        if let Some(lighting) = &self.lighting {
            material = material.with_lighting(match lighting.as_str() {
                "lit" => LightingModel::Lit,
                "unlit" => LightingModel::Unlit,
                named => LightingModel::Named(named.to_owned()),
            });
        }
        if let Some(scale) = self.normal_scale {
            material = material
                .with_normal_scale(scale)
                .map_err(|error| failed(error, "normal_scale"))?;
        }
        if let Some(strength) = self.occlusion_strength {
            material = material
                .with_occlusion_strength(strength)
                .map_err(|error| failed(error, "occlusion_strength"))?;
        }
        let mut maps: Vec<PreparedMap> = Vec::new();
        for (slot, name, map) in self.maps() {
            let field = format!("slots.{slot}.maps.{name}");
            if maps.iter().any(|(known, ..)| known == name) {
                return Err(Error::new(ErrorCode::NameTwice, format!("{path}.{field}")).got(name));
            }
            if !Material::is_name(name) {
                return Err(invalid(&field).expected(Expected::Id).got(name));
            }
            let (id, sampler, transform, tex_coord) = match map {
                SceneMap::Id(id) => (id.clone(), Sampler::REPEAT, UvTransform::IDENTITY, 0),
                SceneMap::Map(map) => {
                    let transform = UvTransform {
                        offset: map.offset.unwrap_or([0.0; 2]),
                        rotation: map.rotation.unwrap_or(0.0),
                        scale: map.scale.unwrap_or([1.0; 2]),
                    };
                    let unfinite = [
                        (
                            "offset",
                            transform.offset.iter().all(|value| value.is_finite()),
                        ),
                        ("rotation", transform.rotation.is_finite()),
                        (
                            "scale",
                            transform.scale.iter().all(|value| value.is_finite()),
                        ),
                    ]
                    .into_iter()
                    .find(|&(_, finite)| !finite);
                    if let Some((part, _)) = unfinite {
                        return Err(invalid(&format!("{field}.{part}")).expected(Expected::Finite));
                    }
                    let tex_coord = map.tex_coord.unwrap_or(0);
                    if tex_coord > 1 {
                        return Err(invalid(&format!("{field}.tex_coord"))
                            .at_most(1)
                            .got(tex_coord));
                    }
                    (map.id.clone(), map.sampler(), transform, tex_coord)
                }
            };
            maps.push((name.to_owned(), id, sampler, transform, tex_coord));
        }
        Ok(PreparedMaterial { material, maps })
    }
}

impl SceneTextureMap {
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
            ..Sampler::REPEAT
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneMaterialControl {
    pub material: SceneMaterial,
}

#[cfg(not(feature = "schema"))]
impl schemars::JsonSchema for SceneMaterialControl {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SceneMaterialControl".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "object",
            "properties": { "material": { "type": "object" } },
            "required": ["material"],
        })
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SceneMaterialReply {
    pub temporary: bool,
    pub material: String,
}

impl Operation for SceneMaterialControl {
    const NAME: &'static str = "scene_material";
    type Params = (
        Option<Res<'static, SceneContext>>,
        Option<ResMut<'static, AssetLibrary>>,
    );
    type Request = Self;
    type Reply = SceneMaterialReply;

    fn run(
        (context, assets): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<SceneMaterialReply, Error> {
        let id = request.material.id.as_str();
        let unknown = |names: Vec<&str>| {
            Error::new(ErrorCode::UnknownReference, "$.material.id")
                .got(id)
                .variants(names)
        };
        let Some(context) = context else {
            return Err(unknown(Vec::new()));
        };
        let handle = context
            .assets
            .materials
            .get(id)
            .copied()
            .ok_or_else(|| unknown(context.assets.materials.keys().map(String::as_str).collect()))?;
        let textures = &context.assets.textures;
        for (field, texture) in request.material.texture_ids() {
            if !textures.contains_key(texture) {
                return Err(
                    Error::new(ErrorCode::UnknownReference, format!("$.material.{field}"))
                        .got(texture)
                        .variants(textures.keys().map(String::as_str)),
                );
            }
        }
        let material = request
            .material
            .prepare("$.material")?
            .material(|texture| textures[texture]);
        let mut assets = assets.ok_or_else(|| Error::new(ErrorCode::MissingAssetLibrary, "$"))?;
        assets.replace_material(handle, material)?;
        Ok(SceneMaterialReply {
            temporary: true,
            material: id.to_owned(),
        })
    }
}
