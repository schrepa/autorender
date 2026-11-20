use engine_codec::{Error, ErrorCode, Expected};
use engine_settings::{InvalidSetting, Rule, Setting};

use crate::{Asset, Texture};

fn invalid(setting: InvalidSetting) -> Error {
    setting.error(ErrorCode::InvalidMaterial, "$")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Wrap {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Filter {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compare {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sampler {
    pub wrap_u: Wrap,
    pub wrap_v: Wrap,
    pub wrap_w: Wrap,
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mipmap_filter: Option<Filter>,
    pub compare: Option<Compare>,
    pub anisotropy: u16,
}

impl Sampler {
    pub const NEAREST: Self = Self {
        wrap_u: Wrap::ClampToEdge,
        wrap_v: Wrap::ClampToEdge,
        wrap_w: Wrap::ClampToEdge,
        mag_filter: Filter::Nearest,
        min_filter: Filter::Nearest,
        mipmap_filter: Some(Filter::Nearest),
        compare: None,
        anisotropy: 1,
    };
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvTransform {
    pub offset: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
}

impl UvTransform {
    pub const IDENTITY: Self = Self {
        offset: [0.0; 2],
        rotation: 0.0,
        scale: [1.0; 2],
    };

    fn is_finite(&self) -> bool {
        self.offset
            .iter()
            .chain(&self.scale)
            .chain([&self.rotation])
            .all(|value| value.is_finite())
    }
}

impl Default for UvTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureRef {
    pub texture: Asset<Texture>,
    pub sampler: Option<Sampler>,
    pub transform: UvTransform,
    pub tex_coord: u32,
}

impl TextureRef {
    pub const fn new(texture: Asset<Texture>) -> Self {
        Self {
            texture,
            sampler: None,
            transform: UvTransform::IDENTITY,
            tex_coord: 0,
        }
    }

    #[must_use]
    pub const fn with_tex_coord(mut self, tex_coord: u32) -> Self {
        self.tex_coord = tex_coord;
        self
    }

    #[must_use]
    pub const fn with_sampler(mut self, sampler: Sampler) -> Self {
        self.sampler = Some(sampler);
        self
    }

    #[must_use]
    pub const fn with_transform(mut self, transform: UvTransform) -> Self {
        self.transform = transform;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask {
        cutoff: f32,
    },
    Blend,
    Additive,
}

impl AlphaMode {
    pub const fn masked(self) -> bool {
        matches!(self, Self::Mask { .. })
    }

    pub const fn blended(self) -> bool {
        matches!(self, Self::Blend | Self::Additive)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Cull {
    #[default]
    Back,
    Front,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DepthBias {
    pub constant: i32,
    pub slope: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderState {
    pub alpha: AlphaMode,
    pub depth_test: bool,
    pub depth_write: bool,
    pub cull: Cull,
    pub bias: DepthBias,
}

impl RenderState {
    pub const OPAQUE: Self = Self {
        alpha: AlphaMode::Opaque,
        depth_test: true,
        depth_write: true,
        cull: Cull::Back,
        bias: DepthBias {
            constant: 0,
            slope: 0.0,
        },
    };

    fn check(&self) -> Result<(), Error> {
        if let AlphaMode::Mask { cutoff } = self.alpha {
            cutoff
                .check("alpha_cutoff", Material::ALPHA_CUTOFF)
                .map_err(invalid)?;
        }
        if !self.bias.slope.is_finite() {
            return Err(Error::new(ErrorCode::InvalidMaterial, "$.state.bias.slope")
                .expected(Expected::Finite)
                .got(self.bias.slope));
        }
        Ok(())
    }
}

impl Default for RenderState {
    fn default() -> Self {
        Self::OPAQUE
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum LightingModel {
    #[default]
    Unlit,
    Named(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    state: RenderState,
    color: [f32; 4],
    emission: f32,
    surface: Vec<(String, f32)>,
    normal_scale: f32,
    emissive: [f32; 3],
    maps: Vec<(String, TextureRef)>,
    slots: Vec<(String, String)>,
    values: Vec<(String, f32)>,
    lighting: LightingModel,
}

impl Material {
    pub const COLOR: Rule = Rule::range(0.0, 1.0);
    pub const OPACITY: Rule = Rule::above_to(0.0, 1.0);
    pub const ALPHA_CUTOFF: Rule = Rule::at_least(0.0);
    pub const NORMAL_SCALE: Rule = Rule::finite();
    pub const EMISSIVE: Rule = Rule::at_least(0.0);
    pub const MAX_EMISSION: f32 = 16.0;
    pub const EMISSION: Rule = Rule::range(0.0, Self::MAX_EMISSION as f64);
    pub const SURFACE: Rule = Rule::range(0.0, 1.0);
    pub const VALUE: Rule = Rule::finite();

    pub fn is_name(name: &str) -> bool {
        is_identifier(name)
    }

    pub fn new(color: [f32; 4]) -> Result<Self, Error> {
        color.check("color", Self::COLOR).map_err(invalid)?;
        if color[3] != 1.0 {
            return Err(Error::new(ErrorCode::InvalidMaterial, "$.color[3]")
                .expected(Expected::Opaque)
                .got(color[3]));
        }
        Ok(Self {
            state: RenderState::OPAQUE,
            color,
            emission: 0.0,
            surface: Vec::new(),
            normal_scale: 1.0,
            emissive: [0.0; 3],
            maps: Vec::new(),
            slots: Vec::new(),
            values: Vec::new(),
            lighting: LightingModel::Unlit,
        })
    }

    pub fn with_opacity(mut self, opacity: f32) -> Result<Self, Error> {
        opacity.check("opacity", Self::OPACITY).map_err(invalid)?;
        if opacity < 1.0 && self.state.alpha == AlphaMode::Opaque {
            return Err(Error::new(ErrorCode::InvalidMaterial, "$.opacity")
                .expected(Expected::Opaque)
                .got(opacity));
        }
        self.color[3] = opacity;
        Ok(self)
    }

    pub fn with_alpha_mode(self, mode: AlphaMode) -> Result<Self, Error> {
        let state = RenderState {
            alpha: mode,
            depth_write: !mode.blended(),
            ..self.state
        };
        self.with_render_state(state)
    }

    pub fn with_render_state(mut self, state: RenderState) -> Result<Self, Error> {
        state.check()?;
        if state.alpha == AlphaMode::Opaque {
            self.color[3] = 1.0;
        }
        self.state = state;
        Ok(self)
    }

    pub fn render_state(&self) -> RenderState {
        self.state
    }

    pub fn opacity(&self) -> f32 {
        self.color[3]
    }

    pub fn alpha_mode(&self) -> AlphaMode {
        self.state.alpha
    }

    pub fn cull(&self) -> Cull {
        self.state.cull
    }

    pub fn with_emission(mut self, emission: f32) -> Result<Self, Error> {
        emission
            .check("emission", Self::EMISSION)
            .map_err(invalid)?;
        self.emission = emission;
        Ok(self)
    }

    pub fn with_surface(mut self, name: impl Into<String>, value: f32) -> Result<Self, Error> {
        let name = name.into();
        if !is_identifier(&name) {
            return Err(Error::new(ErrorCode::InvalidMaterial, "$.surface")
                .expected(Expected::Id)
                .got(name));
        }
        value.check("surface", Self::SURFACE).map_err(invalid)?;
        match self.surface.iter_mut().find(|(known, _)| *known == name) {
            Some((_, held)) => *held = value,
            None => self.surface.push((name, value)),
        }
        Ok(self)
    }

    pub fn with_map(mut self, name: impl Into<String>, texture: TextureRef) -> Result<Self, Error> {
        let name = name.into();
        if !is_identifier(&name) {
            return Err(Error::new(ErrorCode::InvalidMaterial, "$.maps")
                .expected(Expected::Id)
                .got(name));
        }
        let texture = checked(texture)?;
        match self.maps.iter_mut().find(|(known, _)| *known == name) {
            Some((_, held)) => *held = texture,
            None => self.maps.push((name, texture)),
        }
        Ok(self)
    }

    pub fn with_normal_scale(mut self, scale: f32) -> Result<Self, Error> {
        scale
            .check("normal_scale", Self::NORMAL_SCALE)
            .map_err(invalid)?;
        self.normal_scale = scale;
        Ok(self)
    }

    pub fn with_emissive(mut self, color: [f32; 3]) -> Result<Self, Error> {
        color.check("emissive", Self::EMISSIVE).map_err(invalid)?;
        self.emissive = color;
        Ok(self)
    }

    pub fn with_slot(mut self, slot: &str, module: impl Into<String>) -> Result<Self, Error> {
        if !is_identifier(slot) {
            return Err(Error::new(ErrorCode::InvalidMaterial, "$.slots")
                .expected(Expected::Id)
                .got(slot));
        }
        let module = module.into();
        if !is_identifier(&module) {
            return Err(
                Error::new(ErrorCode::InvalidMaterial, format!("$.slots.{slot}"))
                    .expected(Expected::Id)
                    .got(module),
            );
        }
        self.slots.retain(|(known, _)| known != slot);
        self.slots.push((slot.to_owned(), module));
        self.slots.sort();
        Ok(self)
    }

    pub fn with_value(mut self, name: impl Into<String>, value: f32) -> Result<Self, Error> {
        let name = name.into();
        if !is_identifier(&name) {
            return Err(Error::new(ErrorCode::InvalidMaterial, "$.values")
                .expected(Expected::Id)
                .got(name));
        }
        value.check("values", Self::VALUE).map_err(invalid)?;
        match self.values.iter_mut().find(|(known, _)| *known == name) {
            Some((_, held)) => *held = value,
            None => self.values.push((name, value)),
        }
        Ok(self)
    }

    #[must_use]
    pub fn with_lighting(mut self, lighting: LightingModel) -> Self {
        self.lighting = lighting;
        self
    }

    pub fn color(&self) -> [f32; 4] {
        self.color
    }

    pub fn emission(&self) -> f32 {
        self.emission
    }

    pub fn surface(&self) -> &[(String, f32)] {
        &self.surface
    }

    pub fn map(&self, name: &str) -> Option<TextureRef> {
        self.maps
            .iter()
            .find(|(known, _)| known == name)
            .map(|(_, texture)| *texture)
    }

    pub fn maps(&self) -> &[(String, TextureRef)] {
        &self.maps
    }

    pub fn normal_scale(&self) -> f32 {
        self.normal_scale
    }

    pub fn emissive(&self) -> [f32; 3] {
        self.emissive
    }

    pub fn slots(&self) -> &[(String, String)] {
        &self.slots
    }

    pub fn values(&self) -> &[(String, f32)] {
        &self.values
    }

    pub fn value(&self, name: &str) -> Option<f32> {
        self.values
            .iter()
            .find(|(known, _)| known == name)
            .map(|(_, value)| *value)
    }

    pub fn lighting(&self) -> &LightingModel {
        &self.lighting
    }

    pub fn textures(&self) -> impl Iterator<Item = Asset<Texture>> + '_ {
        self.maps.iter().map(|(_, map)| map.texture)
    }
}

fn is_identifier(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_lowercase())
        && characters.all(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
        })
}

fn checked(texture: TextureRef) -> Result<TextureRef, Error> {
    if !texture.transform.is_finite() {
        return Err(
            Error::new(ErrorCode::InvalidMaterial, "$.transform").expected(Expected::Finite)
        );
    }
    if texture.tex_coord > 1 {
        return Err(Error::new(ErrorCode::InvalidMaterial, "$.tex_coord")
            .at_most(1)
            .got(texture.tex_coord));
    }
    Ok(texture)
}
