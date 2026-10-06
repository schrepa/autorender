//! Materials: glTF 2.0 metallic-roughness factors and texture maps (decision
//! 031) with the engine's surface extensions (decision 019).

use crate::{AssetError, TextureHandle};

/// How a surface responds to light beyond its color (decision 019).
///
/// `roughness` and `metalness` are glTF's `roughnessFactor` and
/// `metallicFactor`: a metallic-roughness texture multiplies them.
/// [`Material::with_surface`] validates the ranges. Renderers quantize each
/// value to 8 bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Surface {
    /// Microsurface roughness in `[0, 1]`: 0 is a mirror, 1 spreads highlights
    /// and sky reflections widest.
    pub roughness: f32,
    /// In `[0, 1]`: metals reflect in their own color and diffuse no light.
    pub metalness: f32,
    /// Emitted light as a multiple of the surface color, in
    /// `[0, Surface::MAX_EMISSION]`. Base color texture alpha masks it per
    /// texel. glTF's emissive color is [`Material::with_emissive`].
    pub emission: f32,
    /// Animated ripples of liquid surfaces in `[0, 1]`: wave normals that move
    /// with presentation time. 0 keeps the mesh normals.
    pub ripples: f32,
}

impl Surface {
    /// Broad, dim highlights and no emission: the default.
    pub const MATTE: Self = Self {
        roughness: 0.8,
        metalness: 0.0,
        emission: 0.0,
        ripples: 0.0,
    };

    /// The largest accepted emission multiple.
    pub const MAX_EMISSION: f32 = 16.0;

    fn validate(&self) -> Result<(), AssetError> {
        for (field, value, max) in [
            ("roughness", self.roughness, 1.0),
            ("metalness", self.metalness, 1.0),
            ("emission", self.emission, Self::MAX_EMISSION),
            ("ripples", self.ripples, 1.0),
        ] {
            if !value.is_finite() || !(0.0..=max).contains(&value) {
                return Err(AssetError::InvalidSurface { field });
            }
        }
        Ok(())
    }
}

impl Default for Surface {
    fn default() -> Self {
        Self::MATTE
    }
}

/// A layer settling on a material's faces turned up (decision 093): moss,
/// lichen, dust or snow. It covers faces whose normals point up by more
/// than one less its `coverage`, broken by noise and gathering in the
/// darker hollows of the material's color, in its own linear `color` and
/// `roughness`. [`Material::with_top_layer`] validates the ranges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopLayer {
    /// Linear color, each channel in `[0, 1]`.
    pub color: [f32; 3],
    /// How far from straight up it reaches, in `[0, 1]`: 0 covers nothing,
    /// 1 every face turned up at all.
    pub coverage: f32,
    /// Its roughness, in `[0, 1]`.
    pub roughness: f32,
}

impl TopLayer {
    fn validate(&self) -> Result<(), AssetError> {
        for (field, value) in [
            ("top.color", self.color[0]),
            ("top.color", self.color[1]),
            ("top.color", self.color[2]),
            ("top.coverage", self.coverage),
            ("top.roughness", self.roughness),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(AssetError::InvalidSurface { field });
            }
        }
        Ok(())
    }
}

/// The meters one repeat of a world-mapped material's maps may cover at
/// most (decision 093).
pub const MAX_WORLD_TILING: f32 = 10_000.0;

/// Light passing through a surface instead of diffusing off it (decision
/// 038): glTF's `KHR_materials_transmission`, `KHR_materials_ior` and the
/// thickness of `KHR_materials_volume`. [`Material::with_transmission`]
/// validates the ranges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transmission {
    /// The share of light passing through, in `[0, 1]`, tinted by the base
    /// color; 0 is an ordinary surface.
    pub factor: f32,
    /// Index of refraction in `[1, 3]`: how far rays bend entering the
    /// surface (1.5 for glass, 1.33 for water).
    pub ior: f32,
    /// Depth of the volume beneath the surface, in world units: how far a
    /// bent ray travels before it leaves. 0 is a thin wall, which does not
    /// bend what shows through it.
    pub thickness: f32,
}

impl Transmission {
    /// No light passes through: the default.
    pub const NONE: Self = Self {
        factor: 0.0,
        ior: 1.5,
        thickness: 0.0,
    };

    /// Thin, clear glass: all light passes through, tinted by the color.
    pub const GLASS: Self = Self {
        factor: 1.0,
        ior: 1.5,
        thickness: 0.0,
    };

    fn validate(&self) -> Result<(), AssetError> {
        for (field, value, min, max) in [
            ("transmission", self.factor, 0.0, 1.0),
            ("ior", self.ior, 1.0, 3.0),
            ("thickness", self.thickness, 0.0, f32::MAX),
        ] {
            if !value.is_finite() || !(min..=max).contains(&value) {
                return Err(AssetError::InvalidSurface { field });
            }
        }
        Ok(())
    }
}

impl Default for Transmission {
    fn default() -> Self {
        Self::NONE
    }
}

/// How a texture continues outside `[0, 1]` along one axis (glTF sampler wrap
/// modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Wrap {
    /// Tiles: glTF's default.
    #[default]
    Repeat,
    /// Tiles, mirroring every other repetition.
    MirroredRepeat,
    /// Stretches the edge texels.
    ClampToEdge,
}

/// Texel filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Filter {
    Nearest,
    #[default]
    Linear,
}

/// How a material samples a texture (a glTF sampler). Renderers keep a full
/// mip chain of every texture; `mipmap_filter` `None` samples its base level
/// only, like glTF's `NEAREST` and `LINEAR` minification filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sampler {
    pub wrap_u: Wrap,
    pub wrap_v: Wrap,
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mipmap_filter: Option<Filter>,
}

impl Sampler {
    /// Repeating and trilinear: glTF's default sampler.
    pub const REPEAT: Self = Self {
        wrap_u: Wrap::Repeat,
        wrap_v: Wrap::Repeat,
        mag_filter: Filter::Linear,
        min_filter: Filter::Linear,
        mipmap_filter: Some(Filter::Linear),
    };

    /// Clamped at the edges and trilinear: how [`Material::textured`] and
    /// scenes before version 8 sample.
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

/// Where a texture lies on a mesh's texture coordinates (glTF
/// `KHR_texture_transform`): coordinates are scaled, rotated by `rotation`
/// radians and offset, in that order.
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

    /// `scale` repetitions of the texture across texture coordinates.
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

/// A material's use of a texture: which texture, how it is sampled and where
/// it lies on the mesh's texture coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureRef {
    pub texture: TextureHandle,
    pub sampler: Sampler,
    pub transform: UvTransform,
    /// The texture coordinates it reads (glTF `texCoord`): 0, the mesh's
    /// first, or 1, its second ([`crate::Mesh::with_uv1`]; meshes without
    /// them read their first).
    pub tex_coord: u32,
}

impl TextureRef {
    /// `texture` with glTF's default sampler (repeating, trilinear), no
    /// transform, reading the first texture coordinates.
    pub const fn new(texture: TextureHandle) -> Self {
        Self {
            texture,
            sampler: Sampler::REPEAT,
            transform: UvTransform::IDENTITY,
            tex_coord: 0,
        }
    }

    pub const fn with_tex_coord(mut self, tex_coord: u32) -> Self {
        self.tex_coord = tex_coord;
        self
    }

    pub const fn with_sampler(mut self, sampler: Sampler) -> Self {
        self.sampler = sampler;
        self
    }

    pub const fn with_transform(mut self, transform: UvTransform) -> Self {
        self.transform = transform;
        self
    }
}

/// How a material uses its alpha, the color's alpha times the base color
/// texture's (glTF `alphaMode`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AlphaMode {
    /// Alpha is ignored; base color texture alpha masks the emission
    /// extension ([`Surface::emission`]).
    #[default]
    Opaque,
    /// Surfaces with alpha below `cutoff` are not drawn; the rest are opaque.
    Mask { cutoff: f32 },
    /// Drawn transparent (decision 026): alpha is the fraction of light behind
    /// the surface it covers.
    Blend,
}

/// A glTF 2.0 metallic-roughness material (decision 031): a linear base color
/// factor, [`Surface`] factors and extensions, and optional texture maps.
///
/// Materials are opaque unless given an opacity below 1
/// ([`Material::with_opacity`], decision 026) or another [`AlphaMode`]; the
/// color's alpha is that opacity. Renderers draw materials without maps
/// ([`Material::has_maps`]) from their color and surface alone.
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
    transmission: Transmission,
    world_tiling: Option<f32>,
    top: Option<TopLayer>,
}

impl Material {
    pub fn new(color: [f32; 4]) -> Result<Self, AssetError> {
        for (channel, value) in color.iter().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(value) {
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
            transmission: Transmission::NONE,
            world_tiling: None,
            top: None,
        })
    }

    /// This material with alpha `opacity` in `(0, 1]` (decision 026): the
    /// fraction of light behind it that it covers. Below 1, an opaque
    /// material becomes [`AlphaMode::Blend`]. Transparent surfaces blend in
    /// any order, write no depth and cast no shadow.
    pub fn with_opacity(mut self, opacity: f32) -> Result<Self, AssetError> {
        if !opacity.is_finite() || !(opacity > 0.0 && opacity <= 1.0) {
            return Err(AssetError::InvalidOpacity);
        }
        self.color[3] = opacity;
        if opacity < 1.0 && self.alpha_mode == AlphaMode::Opaque {
            self.alpha_mode = AlphaMode::Blend;
        }
        Ok(self)
    }

    /// This material with `mode`. [`AlphaMode::Opaque`] resets the opacity
    /// to 1; a mask cutoff must be finite and not negative.
    pub fn with_alpha_mode(mut self, mode: AlphaMode) -> Result<Self, AssetError> {
        match mode {
            AlphaMode::Opaque => self.color[3] = 1.0,
            AlphaMode::Mask { cutoff } if !cutoff.is_finite() || cutoff < 0.0 => {
                return Err(AssetError::InvalidMaterialMap {
                    field: "alpha_cutoff",
                });
            }
            _ => {}
        }
        self.alpha_mode = mode;
        Ok(self)
    }

    /// The color's alpha: 1 for opaque materials.
    pub fn opacity(&self) -> f32 {
        self.color[3]
    }

    pub fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    /// Whether this material draws in the transparent pass: it blends, or
    /// light passes through it (decision 038).
    pub fn is_transparent(&self) -> bool {
        self.alpha_mode == AlphaMode::Blend || self.transmission.factor > 0.0
    }

    /// This material letting light through (decision 038). Transmissive
    /// materials draw in the transparent pass; with the `Refraction`
    /// resource they show the opaque scene behind them, bent and blurred,
    /// and without it they blend as plain transparency.
    pub fn with_transmission(mut self, transmission: Transmission) -> Result<Self, AssetError> {
        transmission.validate()?;
        self.transmission = transmission;
        Ok(self)
    }

    pub fn transmission(&self) -> Transmission {
        self.transmission
    }

    /// This material with its maps laid from the world's axes, one repeat
    /// every `tiling` meters, rather than by texture coordinates (decision
    /// 093): each map blends its projections along the axes the surface
    /// faces most, so meshes need no coordinates, show no seams and keep
    /// one texel density at any size. Map transforms still apply. For
    /// surfaces that stay where they are, such as rocks and cliffs: maps
    /// slide over moving ones. `tiling` is finite, positive and at most
    /// [`MAX_WORLD_TILING`].
    pub fn with_world_mapping(mut self, tiling: f32) -> Result<Self, AssetError> {
        if !tiling.is_finite() || tiling <= 0.0 || tiling > MAX_WORLD_TILING {
            return Err(AssetError::InvalidSurface {
                field: "world_tiling",
            });
        }
        self.world_tiling = Some(tiling);
        Ok(self)
    }

    /// The meters one repeat of world-mapped maps covers, if laid so.
    pub fn world_tiling(&self) -> Option<f32> {
        self.world_tiling
    }

    /// This material with `top` settling on its faces turned up (decision
    /// 093).
    pub fn with_top_layer(mut self, top: TopLayer) -> Result<Self, AssetError> {
        top.validate()?;
        self.top = Some(top);
        Ok(self)
    }

    pub fn top_layer(&self) -> Option<TopLayer> {
        self.top
    }

    /// This material with `surface`, whose values must lie in their ranges.
    pub fn with_surface(mut self, surface: Surface) -> Result<Self, AssetError> {
        surface.validate()?;
        self.surface = surface;
        Ok(self)
    }

    /// Multiplies an opaque texture's decoded linear color by this color
    /// factor, clamped at its edges ([`Sampler::CLAMP`]). The library checks
    /// the texture handle when this material is inserted.
    pub fn textured(color: [f32; 4], texture: TextureHandle) -> Result<Self, AssetError> {
        let mut material = Self::new(color)?;
        material.base_color = Some(TextureRef::new(texture).with_sampler(Sampler::CLAMP));
        Ok(material)
    }

    /// This material with a base color texture: sRGB color multiplying the
    /// color factor, and alpha multiplying its alpha.
    pub fn with_base_color_texture(mut self, texture: TextureRef) -> Result<Self, AssetError> {
        self.base_color = Some(checked(texture)?);
        Ok(self)
    }

    /// This material with a metallic-roughness texture: linear green
    /// multiplies roughness and blue metalness.
    pub fn with_metallic_roughness_texture(
        mut self,
        texture: TextureRef,
    ) -> Result<Self, AssetError> {
        self.metallic_roughness = Some(checked(texture)?);
        Ok(self)
    }

    /// This material with a tangent-space normal texture whose X and Y
    /// deflections scale by the finite `scale`.
    pub fn with_normal_texture(
        mut self,
        texture: TextureRef,
        scale: f32,
    ) -> Result<Self, AssetError> {
        if !scale.is_finite() {
            return Err(AssetError::InvalidMaterialMap {
                field: "normal_scale",
            });
        }
        self.normal = Some(checked(texture)?);
        self.normal_scale = scale;
        Ok(self)
    }

    /// This material with an occlusion texture whose linear red darkens
    /// indirect light, by `strength` in `[0, 1]`.
    pub fn with_occlusion_texture(
        mut self,
        texture: TextureRef,
        strength: f32,
    ) -> Result<Self, AssetError> {
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(AssetError::InvalidMaterialMap {
                field: "occlusion_strength",
            });
        }
        self.occlusion = Some(checked(texture)?);
        self.occlusion_strength = strength;
        Ok(self)
    }

    /// This material emitting linear `color` (glTF's emissive factor times
    /// its emissive strength): finite and not negative.
    pub fn with_emissive(mut self, color: [f32; 3]) -> Result<Self, AssetError> {
        if color.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(AssetError::InvalidMaterialMap { field: "emissive" });
        }
        self.emissive = color;
        Ok(self)
    }

    /// This material with an emissive texture: sRGB color multiplying the
    /// emissive color.
    pub fn with_emissive_texture(mut self, texture: TextureRef) -> Result<Self, AssetError> {
        self.emissive_texture = Some(checked(texture)?);
        Ok(self)
    }

    /// This material drawn from both sides, back faces lit with the normal
    /// flipped.
    pub fn with_double_sided(mut self, double_sided: bool) -> Self {
        self.double_sided = double_sided;
        self
    }

    pub fn color(&self) -> [f32; 4] {
        self.color
    }

    /// The base color texture's handle.
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

    /// Every texture map, in the order base color, metallic-roughness,
    /// normal, occlusion and emissive.
    pub fn maps(&self) -> [Option<TextureRef>; 5] {
        [
            self.base_color,
            self.metallic_roughness,
            self.normal,
            self.occlusion,
            self.emissive_texture,
        ]
    }

    /// The textures this material reads.
    pub fn textures(&self) -> impl Iterator<Item = TextureHandle> + '_ {
        self.maps().into_iter().flatten().map(|map| map.texture)
    }

    /// Whether renderers bind this material's own maps and parameters: it
    /// has a texture, an emissive color, two sides, an alpha mask, blends
    /// with full alpha or transmits light. Other materials draw from color
    /// and surface alone.
    pub fn has_maps(&self) -> bool {
        self.maps().iter().any(Option::is_some)
            || self.emissive != [0.0; 3]
            || self.double_sided
            || matches!(self.alpha_mode, AlphaMode::Mask { .. })
            || (self.alpha_mode == AlphaMode::Blend && self.color[3] == 1.0)
            || self.transmission.factor > 0.0
    }

    /// What renderers bind for a material with maps: everything but its
    /// color and surface, which they read per object. Materials with equal
    /// bindings draw together whatever their colors.
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
                    _ => 0.0,
                },
                self.transmission.factor,
                self.transmission.ior,
                self.transmission.thickness,
            ]
            .map(f32::to_bits),
            alpha: match self.alpha_mode {
                AlphaMode::Opaque => 0,
                AlphaMode::Mask { .. } => 1,
                AlphaMode::Blend => 2,
            },
            double_sided: self.double_sided,
        })
    }
}

/// One map of a binding: its texture, sampler, transform bits and
/// coordinate set.
type BoundMap = (TextureHandle, Sampler, [u32; 5], u32);

/// A material's maps and map parameters ([`Material::binding`]): its
/// textures with their samplers and transforms, normal scale, occlusion
/// strength, emissive color, alpha mode, transmission and sides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialBinding {
    maps: [Option<BoundMap>; 5],
    parameters: [u32; 9],
    alpha: u8,
    double_sided: bool,
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
