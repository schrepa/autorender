use engine_codec::{Error, ErrorCode, Expected};
use engine_settings::{InvalidSetting, Rule, Setting, fields};

use crate::TextureHandle;

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
        pub ripples: f32 = Surface::MATTE.ripples => range(0.0, 1.0),
    }
}

impl Surface {
    pub const MATTE: Self = Self {
        roughness: 0.8,
        metalness: 0.0,
        emission: 0.0,
        ripples: 0.0,
    };

    pub const MAX_EMISSION: f32 = 16.0;
}

fields! {
    unauthored;
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct TopLayer {
        pub color: [f32; 3] = [0.5; 3] => range(0.0, 1.0),
        pub coverage: f32 = 0.5 => range(0.0, 1.0),
        pub roughness: f32 = 0.85 => range(0.0, 1.0),
    }
}

pub const MAX_WORLD_TILING: f32 = 10_000.0;

fields! {
    unauthored;
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Grounding {
        pub height: f32 = 1.0 => above_to(0.0, MAX_GROUNDING_HEIGHT),
    }
}

pub const MAX_GROUNDING_HEIGHT: f32 = 10_000.0;

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
pub struct Sampler {
    pub wrap_u: Wrap,
    pub wrap_v: Wrap,
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mipmap_filter: Option<Filter>,
}

impl Sampler {
    pub const REPEAT: Self = Self {
        wrap_u: Wrap::Repeat,
        wrap_v: Wrap::Repeat,
        mag_filter: Filter::Linear,
        min_filter: Filter::Linear,
        mipmap_filter: Some(Filter::Linear),
    };

    pub const CLAMP: Self = Self {
        wrap_u: Wrap::ClampToEdge,
        wrap_v: Wrap::ClampToEdge,
        ..Self::REPEAT
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    color: [f32; 4],
    base_color: Option<TextureRef>,
    surface: Surface,
    metallic_roughness: Option<TextureRef>,
    normal: Option<TextureRef>,
    normal_scale: f32,
    occlusion: Option<TextureRef>,
    occlusion_strength: f32,
    emissive: [f32; 3],
    emissive_texture: Option<TextureRef>,
    alpha_mode: AlphaMode,
    double_sided: bool,
    foliage: bool,
    world_tiling: Option<f32>,
    top: Option<TopLayer>,
    grounding: Option<Grounding>,
}

impl Material {
    pub const COLOR: Rule = Rule::range(0.0, 1.0);
    pub const OPACITY: Rule = Rule::above_to(0.0, 1.0);
    pub const ALPHA_CUTOFF: Rule = Rule::at_least(0.0);
    pub const NORMAL_SCALE: Rule = Rule::finite();
    pub const OCCLUSION_STRENGTH: Rule = Rule::range(0.0, 1.0);
    pub const EMISSIVE: Rule = Rule::at_least(0.0);
    pub const WORLD_TILING: Rule = Rule::above_to(0.0, MAX_WORLD_TILING as f64);

    pub fn new(color: [f32; 4]) -> Result<Self, Error> {
        color.check("color", Self::COLOR).map_err(invalid)?;
        if color[3] != 1.0 {
            return Err(Error::new(ErrorCode::InvalidMaterial, "$.color[3]")
                .expected(Expected::Opaque)
                .got(color[3]));
        }
        Ok(Self {
            color,
            base_color: None,
            surface: Surface::MATTE,
            metallic_roughness: None,
            normal: None,
            normal_scale: 1.0,
            occlusion: None,
            occlusion_strength: 1.0,
            emissive: [0.0; 3],
            emissive_texture: None,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            foliage: false,
            world_tiling: None,
            top: None,
            grounding: None,
        })
    }

    pub fn with_opacity(mut self, opacity: f32) -> Result<Self, Error> {
        opacity.check("opacity", Self::OPACITY).map_err(invalid)?;
        if opacity < 1.0 && self.alpha_mode == AlphaMode::Opaque {
            return Err(Error::new(ErrorCode::InvalidMaterial, "$.opacity")
                .expected(Expected::Opaque)
                .got(opacity));
        }
        self.color[3] = opacity;
        Ok(self)
    }

    pub fn with_alpha_mode(mut self, mode: AlphaMode) -> Result<Self, Error> {
        match mode {
            AlphaMode::Opaque => self.color[3] = 1.0,
            AlphaMode::Mask { cutoff } => {
                cutoff
                    .check("alpha_cutoff", Self::ALPHA_CUTOFF)
                    .map_err(invalid)?;
            }
        }
        self.alpha_mode = mode;
        Ok(self)
    }

    pub fn opacity(&self) -> f32 {
        self.color[3]
    }

    pub fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    pub fn with_world_mapping(mut self, tiling: f32) -> Result<Self, Error> {
        tiling
            .check("world_tiling", Self::WORLD_TILING)
            .map_err(invalid)?;
        self.world_tiling = Some(tiling);
        Ok(self)
    }

    pub fn world_tiling(&self) -> Option<f32> {
        self.world_tiling
    }

    pub fn with_top_layer(mut self, top: TopLayer) -> Result<Self, Error> {
        top.validate()
            .map_err(|setting| invalid_part("top", setting))?;
        self.top = Some(top);
        Ok(self)
    }

    pub fn top_layer(&self) -> Option<TopLayer> {
        self.top
    }

    pub fn with_grounding(mut self, grounding: Grounding) -> Result<Self, Error> {
        grounding
            .validate()
            .map_err(|setting| invalid_part("grounding", setting))?;
        self.grounding = Some(grounding);
        Ok(self)
    }

    pub fn grounding(&self) -> Option<Grounding> {
        self.grounding
    }

    pub fn with_surface(mut self, surface: Surface) -> Result<Self, Error> {
        surface
            .validate()
            .map_err(|setting| invalid_part("surface", setting))?;
        self.surface = surface;
        Ok(self)
    }

    pub fn textured(color: [f32; 4], texture: TextureHandle) -> Result<Self, Error> {
        let mut material = Self::new(color)?;
        material.base_color = Some(TextureRef::new(texture).with_sampler(Sampler::CLAMP));
        Ok(material)
    }

    pub fn with_base_color_texture(mut self, texture: TextureRef) -> Result<Self, Error> {
        self.base_color = Some(checked(texture)?);
        Ok(self)
    }

    pub fn with_metallic_roughness_texture(mut self, texture: TextureRef) -> Result<Self, Error> {
        self.metallic_roughness = Some(checked(texture)?);
        Ok(self)
    }

    pub fn with_normal_texture(mut self, texture: TextureRef, scale: f32) -> Result<Self, Error> {
        scale
            .check("normal_scale", Self::NORMAL_SCALE)
            .map_err(invalid)?;
        self.normal = Some(checked(texture)?);
        self.normal_scale = scale;
        Ok(self)
    }

    pub fn with_occlusion_texture(
        mut self,
        texture: TextureRef,
        strength: f32,
    ) -> Result<Self, Error> {
        strength
            .check("occlusion_strength", Self::OCCLUSION_STRENGTH)
            .map_err(invalid)?;
        self.occlusion = Some(checked(texture)?);
        self.occlusion_strength = strength;
        Ok(self)
    }

    pub fn with_emissive(mut self, color: [f32; 3]) -> Result<Self, Error> {
        color.check("emissive", Self::EMISSIVE).map_err(invalid)?;
        self.emissive = color;
        Ok(self)
    }

    pub fn with_emissive_texture(mut self, texture: TextureRef) -> Result<Self, Error> {
        self.emissive_texture = Some(checked(texture)?);
        Ok(self)
    }

    #[must_use]
    pub fn with_double_sided(mut self, double_sided: bool) -> Self {
        self.double_sided = double_sided;
        self
    }

    #[must_use]
    pub fn with_foliage(mut self, foliage: bool) -> Self {
        self.foliage = foliage;
        self
    }

    pub fn color(&self) -> [f32; 4] {
        self.color
    }

    pub fn texture(&self) -> Option<TextureHandle> {
        self.base_color.map(|texture| texture.texture)
    }

    pub fn base_color_texture(&self) -> Option<TextureRef> {
        self.base_color
    }

    pub fn surface(&self) -> Surface {
        self.surface
    }

    pub fn metallic_roughness_texture(&self) -> Option<TextureRef> {
        self.metallic_roughness
    }

    pub fn normal_texture(&self) -> Option<TextureRef> {
        self.normal
    }

    pub fn normal_scale(&self) -> f32 {
        self.normal_scale
    }

    pub fn occlusion_texture(&self) -> Option<TextureRef> {
        self.occlusion
    }

    pub fn occlusion_strength(&self) -> f32 {
        self.occlusion_strength
    }

    pub fn emissive(&self) -> [f32; 3] {
        self.emissive
    }

    pub fn emissive_texture(&self) -> Option<TextureRef> {
        self.emissive_texture
    }

    pub fn is_double_sided(&self) -> bool {
        self.double_sided
    }

    pub fn is_foliage(&self) -> bool {
        self.foliage
    }

    pub fn maps(&self) -> [Option<TextureRef>; 5] {
        [
            self.base_color,
            self.metallic_roughness,
            self.normal,
            self.occlusion,
            self.emissive_texture,
        ]
    }

    pub fn textures(&self) -> impl Iterator<Item = TextureHandle> + '_ {
        self.maps().into_iter().flatten().map(|map| map.texture)
    }

    pub fn has_maps(&self) -> bool {
        self.maps().iter().any(Option::is_some)
            || self.emissive != [0.0; 3]
            || self.double_sided
            || matches!(self.alpha_mode, AlphaMode::Mask { .. })
    }

    pub fn binding(&self) -> Option<MaterialBinding> {
        self.has_maps().then(|| MaterialBinding {
            maps: self.maps().map(|map| {
                map.map(|map| {
                    let UvTransform {
                        offset: [u, v],
                        rotation,
                        scale: [x, y],
                    } = map.transform;
                    (
                        map.texture,
                        map.sampler,
                        [u, v, rotation, x, y].map(f32::to_bits),
                        map.tex_coord,
                    )
                })
            }),
            parameters: [
                self.normal_scale,
                self.occlusion_strength,
                self.emissive[0],
                self.emissive[1],
                self.emissive[2],
                match self.alpha_mode {
                    AlphaMode::Mask { cutoff } => cutoff,
                    AlphaMode::Opaque => 0.0,
                },
            ]
            .map(f32::to_bits),
            masked: matches!(self.alpha_mode, AlphaMode::Mask { .. }),
            double_sided: self.double_sided,
            foliage: self.foliage,
        })
    }
}

type BoundMap = (TextureHandle, Sampler, [u32; 5], u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialBinding {
    maps: [Option<BoundMap>; 5],
    parameters: [u32; 6],
    masked: bool,
    double_sided: bool,
    foliage: bool,
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
