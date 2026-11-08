use engine_codec::{Error, ErrorCode, Expected};
use engine_settings::{InvalidSetting, Rule, Setting, fields};

use crate::{ColorSpace, TextureHandle};

fn invalid(setting: InvalidSetting) -> Error {
    setting.error(ErrorCode::InvalidMaterial, "$")
}

fn invalid_part(part: &'static str, setting: InvalidSetting) -> Error {
    invalid(setting.within(part))
}

fields! {
    unauthored;
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Surface {
        pub roughness: f32 = Surface::MATTE.roughness => range(0.0, 1.0),
        pub metalness: f32 = Surface::MATTE.metalness => range(0.0, 1.0),
        pub emission: f32 = Surface::MATTE.emission => range(0.0, Surface::MAX_EMISSION),
    }
}

impl Surface {
    pub const MATTE: Self = Self {
        roughness: 0.8,
        metalness: 0.0,
        emission: 0.0,
    };

    pub const MAX_EMISSION: f32 = 16.0;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Wrap {
    #[default]
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Filter {
    Nearest,
    #[default]
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
    pub const ANISOTROPY: u16 = 16;

    pub const REPEAT: Self = Self {
        wrap_u: Wrap::Repeat,
        wrap_v: Wrap::Repeat,
        wrap_w: Wrap::ClampToEdge,
        mag_filter: Filter::Linear,
        min_filter: Filter::Linear,
        mipmap_filter: Some(Filter::Linear),
        compare: None,
        anisotropy: Self::ANISOTROPY,
    };

    pub const CLAMP: Self = Self {
        wrap_u: Wrap::ClampToEdge,
        wrap_v: Wrap::ClampToEdge,
        ..Self::REPEAT
    };

    pub const LINEAR_CLAMP: Self = Self {
        anisotropy: 1,
        ..Self::CLAMP
    };

    pub const NEAREST: Self = Self {
        mag_filter: Filter::Nearest,
        min_filter: Filter::Nearest,
        mipmap_filter: Some(Filter::Nearest),
        ..Self::LINEAR_CLAMP
    };

    pub const COMPARE_LINEAR: Self = Self {
        compare: Some(Compare::LessEqual),
        ..Self::LINEAR_CLAMP
    };
}

impl Default for Sampler {
    fn default() -> Self {
        Self::REPEAT
    }
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

    pub const fn repeat(scale: [f32; 2]) -> Self {
        Self {
            scale,
            ..Self::IDENTITY
        }
    }

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
    pub texture: TextureHandle,
    pub sampler: Sampler,
    pub transform: UvTransform,
    pub tex_coord: u32,
}

impl TextureRef {
    pub const fn new(texture: TextureHandle) -> Self {
        Self {
            texture,
            sampler: Sampler::REPEAT,
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
        self.sampler = sampler;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SurfaceSlot {
    Position,
    BaseColor,
    Opacity,
    Cutoff,
    Normal,
    Roughness,
    Metalness,
    Emissive,
    Occlusion,
    Transmission,
    Kind,
    LightInputs,
    Output,
}

impl SurfaceSlot {
    pub const ALL: [Self; 13] = [
        Self::Position,
        Self::BaseColor,
        Self::Opacity,
        Self::Cutoff,
        Self::Normal,
        Self::Roughness,
        Self::Metalness,
        Self::Emissive,
        Self::Occlusion,
        Self::Transmission,
        Self::Kind,
        Self::LightInputs,
        Self::Output,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Position => "position",
            Self::BaseColor => "base_color",
            Self::Opacity => "opacity",
            Self::Cutoff => "cutoff",
            Self::Normal => "normal",
            Self::Roughness => "roughness",
            Self::Metalness => "metalness",
            Self::Emissive => "emissive",
            Self::Occlusion => "occlusion",
            Self::Transmission => "transmission",
            Self::Kind => "kind",
            Self::LightInputs => "light_inputs",
            Self::Output => "output",
        }
    }

    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|slot| slot.name() == name)
    }

    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum LightingModel {
    #[default]
    Lit,
    Unlit,
    Named(String),
}

pub const MAPPED: &str = "mapped";
pub const BASE_COLOR_MAP: &str = "base_color";
pub const METALLIC_ROUGHNESS_MAP: &str = "metallic_roughness";
pub const NORMAL_MAP: &str = "normal";
pub const OCCLUSION_MAP: &str = "occlusion";
pub const EMISSIVE_MAP: &str = "emissive";
pub const MAPPED_MAPS: [(&str, ColorSpace); 5] = [
    (BASE_COLOR_MAP, ColorSpace::Srgb),
    (METALLIC_ROUGHNESS_MAP, ColorSpace::Linear),
    (NORMAL_MAP, ColorSpace::Linear),
    (OCCLUSION_MAP, ColorSpace::Linear),
    (EMISSIVE_MAP, ColorSpace::Srgb),
];
pub const MAPPED_SLOTS: [(SurfaceSlot, &[&str]); 8] = [
    (SurfaceSlot::BaseColor, &[BASE_COLOR_MAP]),
    (SurfaceSlot::Opacity, &[BASE_COLOR_MAP]),
    (SurfaceSlot::Cutoff, &[BASE_COLOR_MAP]),
    (SurfaceSlot::Normal, &[NORMAL_MAP]),
    (SurfaceSlot::Roughness, &[METALLIC_ROUGHNESS_MAP]),
    (SurfaceSlot::Metalness, &[METALLIC_ROUGHNESS_MAP]),
    (SurfaceSlot::Emissive, &[EMISSIVE_MAP, BASE_COLOR_MAP]),
    (SurfaceSlot::Occlusion, &[OCCLUSION_MAP]),
];

#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    state: RenderState,
    color: [f32; 4],
    surface: Surface,
    normal_scale: f32,
    occlusion_strength: f32,
    emissive: [f32; 3],
    transmission: f32,
    maps: Vec<(String, TextureRef)>,
    slots: Vec<(SurfaceSlot, String)>,
    values: Vec<(String, f32)>,
    lighting: LightingModel,
}

impl Material {
    pub const COLOR: Rule = Rule::range(0.0, 1.0);
    pub const OPACITY: Rule = Rule::above_to(0.0, 1.0);
    pub const ALPHA_CUTOFF: Rule = Rule::at_least(0.0);
    pub const NORMAL_SCALE: Rule = Rule::finite();
    pub const OCCLUSION_STRENGTH: Rule = Rule::range(0.0, 1.0);
    pub const EMISSIVE: Rule = Rule::at_least(0.0);
    pub const TRANSMISSION: Rule = Rule::range(0.0, 1.0);
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
            surface: Surface::MATTE,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            emissive: [0.0; 3],
            transmission: 0.0,
            maps: Vec::new(),
            slots: Vec::new(),
            values: Vec::new(),
            lighting: LightingModel::Lit,
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

    #[must_use]
    pub fn with_cull(mut self, cull: Cull) -> Self {
        self.state.cull = cull;
        self
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

    pub fn with_surface(mut self, surface: Surface) -> Result<Self, Error> {
        surface
            .validate()
            .map_err(|setting| invalid_part("surface", setting))?;
        self.surface = surface;
        Ok(self)
    }

    pub fn textured(color: [f32; 4], texture: TextureHandle) -> Result<Self, Error> {
        Self::new(color)?
            .with_mapped(BASE_COLOR_MAP, TextureRef::new(texture).with_sampler(Sampler::CLAMP))
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

    pub fn with_mapped(self, map: &str, texture: TextureRef) -> Result<Self, Error> {
        let mut material = self.with_map(map, texture)?;
        for (slot, maps) in MAPPED_SLOTS {
            if maps.contains(&map) && material.slot(slot).is_none() {
                material = material.with_slot(slot, MAPPED)?;
            }
        }
        Ok(material)
    }

    pub fn with_normal_scale(mut self, scale: f32) -> Result<Self, Error> {
        scale
            .check("normal_scale", Self::NORMAL_SCALE)
            .map_err(invalid)?;
        self.normal_scale = scale;
        Ok(self)
    }

    pub fn with_occlusion_strength(mut self, strength: f32) -> Result<Self, Error> {
        strength
            .check("occlusion_strength", Self::OCCLUSION_STRENGTH)
            .map_err(invalid)?;
        self.occlusion_strength = strength;
        Ok(self)
    }

    pub fn with_emissive(mut self, color: [f32; 3]) -> Result<Self, Error> {
        color.check("emissive", Self::EMISSIVE).map_err(invalid)?;
        self.emissive = color;
        Ok(self)
    }

    pub fn with_transmission(mut self, transmission: f32) -> Result<Self, Error> {
        transmission
            .check("transmission", Self::TRANSMISSION)
            .map_err(invalid)?;
        self.transmission = transmission;
        Ok(self)
    }

    pub fn with_slot(mut self, slot: SurfaceSlot, module: impl Into<String>) -> Result<Self, Error> {
        let module = module.into();
        if !is_identifier(&module) {
            return Err(
                Error::new(ErrorCode::InvalidMaterial, format!("$.slots.{}", slot.name()))
                    .expected(Expected::Id)
                    .got(module),
            );
        }
        self.slots.retain(|(known, _)| *known != slot);
        self.slots.push((slot, module));
        self.slots.sort_by_key(|(known, _)| *known);
        Ok(self)
    }

    pub fn with_value(mut self, name: impl Into<String>, value: f32) -> Result<Self, Error> {
        let name = name.into();
        if !is_identifier(&name) {
            return Err(Error::new(ErrorCode::InvalidMaterial, "$.values")
                .expected(Expected::Id)
                .got(name));
        }
        value
            .check("values", Self::VALUE)
            .map_err(invalid)?;
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

    pub fn surface(&self) -> Surface {
        self.surface
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

    pub fn occlusion_strength(&self) -> f32 {
        self.occlusion_strength
    }

    pub fn emissive(&self) -> [f32; 3] {
        self.emissive
    }

    pub fn transmission(&self) -> f32 {
        self.transmission
    }

    pub fn slots(&self) -> &[(SurfaceSlot, String)] {
        &self.slots
    }

    pub fn slot(&self, slot: SurfaceSlot) -> Option<&str> {
        self.slots
            .iter()
            .find(|(known, _)| *known == slot)
            .map(|(_, module)| module.as_str())
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

    pub fn textures(&self) -> impl Iterator<Item = TextureHandle> + '_ {
        self.maps.iter().map(|(_, map)| map.texture)
    }

    pub fn binding(&self) -> Option<MaterialBinding> {
        let plain = self.maps.is_empty()
            && self.slots.is_empty()
            && self.values.is_empty()
            && self.emissive == [0.0; 3]
            && self.transmission == 0.0
            && self.state == RenderState::OPAQUE
            && self.lighting == LightingModel::Lit;
        (!plain).then(|| MaterialBinding {
            state: (
                match self.state.alpha {
                    AlphaMode::Opaque => (0, 0),
                    AlphaMode::Mask { cutoff } => (1, cutoff.to_bits()),
                    AlphaMode::Blend => (2, 0),
                    AlphaMode::Additive => (3, 0),
                },
                self.state.depth_test,
                self.state.depth_write,
                self.state.cull,
                (self.state.bias.constant, self.state.bias.slope.to_bits()),
            ),
            maps: self
                .maps
                .iter()
                .map(|(name, map)| {
                    let UvTransform {
                        offset: [u, v],
                        rotation,
                        scale: [x, y],
                    } = map.transform;
                    (
                        name.clone(),
                        map.texture,
                        map.sampler,
                        [u, v, rotation, x, y].map(f32::to_bits),
                        map.tex_coord,
                    )
                })
                .collect(),
            factors: [
                self.normal_scale,
                self.occlusion_strength,
                self.emissive[0],
                self.emissive[1],
                self.emissive[2],
                self.transmission,
            ]
            .map(f32::to_bits),
            slots: self.slots.clone(),
            values: self
                .values
                .iter()
                .map(|(name, value)| (name.clone(), value.to_bits()))
                .collect(),
            lighting: self.lighting.clone(),
        })
    }
}

type BoundMap = (String, TextureHandle, Sampler, [u32; 5], u32);
type BoundState = ((u8, u32), bool, bool, Cull, (i32, u32));

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MaterialBinding {
    state: BoundState,
    maps: Vec<BoundMap>,
    factors: [u32; 6],
    slots: Vec<(SurfaceSlot, String)>,
    values: Vec<(String, u32)>,
    lighting: LightingModel,
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
