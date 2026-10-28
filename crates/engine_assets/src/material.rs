use engine_settings::{InvalidSetting, Rule, Setting, settings};

use crate::{AssetError, TextureHandle};

fn invalid_part(part: Option<&'static str>, invalid: InvalidSetting) -> AssetError {
    AssetError::InvalidSurface {
        field: match part {
            Some(part) => invalid.within(part).field,
            None => invalid.field,
        },
    }
}

settings! {
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

settings! {
    unauthored;
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct TopLayer {
        pub color: [f32; 3] = [0.5; 3] => range(0.0, 1.0),
        pub coverage: f32 = 0.5 => range(0.0, 1.0),
        pub roughness: f32 = 0.85 => range(0.0, 1.0),
    }
}

pub const MAX_WORLD_TILING: f32 = 10_000.0;

settings! {
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

    pub fn new(color: [f32; 4]) -> Result<Self, AssetError> {
        for (channel, value) in color.iter().enumerate() {
            if value.check("color", Self::COLOR).is_err() {
                return Err(AssetError::InvalidColor { channel });
            }
        }
        if color[3] != 1.0 {
            return Err(AssetError::UnsupportedAlpha);
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

    pub fn with_opacity(mut self, opacity: f32) -> Result<Self, AssetError> {
        if opacity.check("opacity", Self::OPACITY).is_err() {
            return Err(AssetError::InvalidOpacity);
        }
        if opacity < 1.0 && self.alpha_mode == AlphaMode::Opaque {
            return Err(AssetError::UnsupportedAlpha);
        }
        self.color[3] = opacity;
        Ok(self)
    }

    pub fn with_alpha_mode(mut self, mode: AlphaMode) -> Result<Self, AssetError> {
        match mode {
            AlphaMode::Opaque => self.color[3] = 1.0,
            AlphaMode::Mask { cutoff }
                if cutoff.check("alpha_cutoff", Self::ALPHA_CUTOFF).is_err() =>
            {
                return Err(AssetError::InvalidMaterialMap {
                    field: "alpha_cutoff",
                });
            }
            AlphaMode::Mask { .. } => {}
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

    pub fn with_world_mapping(mut self, tiling: f32) -> Result<Self, AssetError> {
        if tiling.check("world_tiling", Self::WORLD_TILING).is_err() {
            return Err(AssetError::InvalidSurface {
                field: "world_tiling".into(),
            });
        }
        self.world_tiling = Some(tiling);
        Ok(self)
    }

    pub fn world_tiling(&self) -> Option<f32> {
        self.world_tiling
    }

    pub fn with_top_layer(mut self, top: TopLayer) -> Result<Self, AssetError> {
        top.validate()
            .map_err(|invalid| invalid_part(Some("top"), invalid))?;
        self.top = Some(top);
        Ok(self)
    }

    pub fn top_layer(&self) -> Option<TopLayer> {
        self.top
    }

    pub fn with_grounding(mut self, grounding: Grounding) -> Result<Self, AssetError> {
        grounding
            .validate()
            .map_err(|invalid| invalid_part(Some("grounding"), invalid))?;
        self.grounding = Some(grounding);
        Ok(self)
    }

    pub fn grounding(&self) -> Option<Grounding> {
        self.grounding
    }

    pub fn with_surface(mut self, surface: Surface) -> Result<Self, AssetError> {
        surface
            .validate()
            .map_err(|invalid| invalid_part(None, invalid))?;
        self.surface = surface;
        Ok(self)
    }

    pub fn textured(color: [f32; 4], texture: TextureHandle) -> Result<Self, AssetError> {
        let mut material = Self::new(color)?;
        material.base_color = Some(TextureRef::new(texture).with_sampler(Sampler::CLAMP));
        Ok(material)
    }

    pub fn with_base_color_texture(mut self, texture: TextureRef) -> Result<Self, AssetError> {
        self.base_color = Some(checked(texture)?);
        Ok(self)
    }

    pub fn with_metallic_roughness_texture(
        mut self,
        texture: TextureRef,
    ) -> Result<Self, AssetError> {
        self.metallic_roughness = Some(checked(texture)?);
        Ok(self)
    }

    pub fn with_normal_texture(
        mut self,
        texture: TextureRef,
        scale: f32,
    ) -> Result<Self, AssetError> {
        if scale.check("normal_scale", Self::NORMAL_SCALE).is_err() {
            return Err(AssetError::InvalidMaterialMap {
                field: "normal_scale",
            });
        }
        self.normal = Some(checked(texture)?);
        self.normal_scale = scale;
        Ok(self)
    }

    pub fn with_occlusion_texture(
        mut self,
        texture: TextureRef,
        strength: f32,
    ) -> Result<Self, AssetError> {
        if strength
            .check("occlusion_strength", Self::OCCLUSION_STRENGTH)
            .is_err()
        {
            return Err(AssetError::InvalidMaterialMap {
                field: "occlusion_strength",
            });
        }
        self.occlusion = Some(checked(texture)?);
        self.occlusion_strength = strength;
        Ok(self)
    }

    pub fn with_emissive(mut self, color: [f32; 3]) -> Result<Self, AssetError> {
        if color.check("emissive", Self::EMISSIVE).is_err() {
            return Err(AssetError::InvalidMaterialMap { field: "emissive" });
        }
        self.emissive = color;
        Ok(self)
    }

    pub fn with_emissive_texture(mut self, texture: TextureRef) -> Result<Self, AssetError> {
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

fn checked(texture: TextureRef) -> Result<TextureRef, AssetError> {
    if !texture.transform.is_finite() {
        return Err(AssetError::InvalidMaterialMap { field: "transform" });
    }
    if texture.tex_coord > 1 {
        return Err(AssetError::InvalidMaterialMap { field: "tex_coord" });
    }
    Ok(texture)
}
