//! Materials: glTF 2.0 metallic-roughness factors and texture maps
//! with the engine's surface extensions.

use engine_settings::{ExtraRules, InvalidSetting, Rule, Setting, settings};

use crate::{AssetError, TextureHandle};

/// The error of material part `part`'s field breaking its rule.
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
    /// How a surface responds to light beyond its color.
    ///
    /// `roughness` and `metalness` are glTF's `roughnessFactor` and
    /// `metallicFactor`: a metallic-roughness texture multiplies them.
    /// [`Material::with_surface`] checks the rules. Renderers quantize each
    /// value to 8 bits.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Surface {
        /// Microsurface roughness: 0 is a mirror, 1 spreads highlights and sky
        /// reflections widest.
        pub roughness: f32 = Surface::MATTE.roughness => range(0.0, 1.0),
        /// Metals reflect in their own color and diffuse no light.
        pub metalness: f32 = Surface::MATTE.metalness => range(0.0, 1.0),
        /// Emitted light as a multiple of the surface color. Base color
        /// texture alpha masks it per texel. glTF's emissive color is
        /// [`Material::with_emissive`].
        pub emission: f32 = Surface::MATTE.emission => range(0.0, Surface::MAX_EMISSION),
        /// Animated ripples of liquid surfaces: wave normals that move with
        /// presentation time. 0 keeps the mesh normals.
        pub ripples: f32 = Surface::MATTE.ripples => range(0.0, 1.0),
    }
}

impl ExtraRules for Surface {}

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
}

settings! {
    unauthored;
    /// A layer settling on a material's faces turned up: moss,
    /// lichen, dust or snow. It covers faces whose normals point up by more
    /// than one less its `coverage`, broken by noise and gathering in the
    /// darker hollows of the material's color, in its own linear `color` and
    /// `roughness`. [`Material::with_top_layer`] checks the rules.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct TopLayer {
        /// Linear color.
        pub color: [f32; 3] = [0.5; 3] => range(0.0, 1.0),
        /// How far from straight up it reaches: 0 covers nothing, 1 every
        /// face turned up at all.
        pub coverage: f32 = 0.5 => range(0.0, 1.0),
        /// Its roughness.
        pub roughness: f32 = 0.85 => range(0.0, 1.0),
    }
}

impl ExtraRules for TopLayer {}

/// The meters one repeat of a world-mapped material's maps may cover at
/// most.
pub const MAX_WORLD_TILING: f32 = 10_000.0;

settings! {
    unauthored;
    /// The ground staining the base of an object standing on terrain:
    /// soil, damp or dust where rock or wood meets the ground. Where the
    /// object meets the ground its color is multiplied by `color`, the stain
    /// fading out `height` units of its mesh higher (so it scales with the
    /// object), measured from the ground's own slope under it. Objects
    /// terrains place stand on terrain; others are never stained.
    /// [`Material::with_grounding`] checks the rules.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Grounding {
        /// Linear color the stain multiplies by.
        pub color: [f32; 3] = [0.5; 3] => range(0.0, 1.0),
        /// How far up the stain reaches, in mesh units.
        pub height: f32 = 1.0 => above_to(0.0, MAX_GROUNDING_HEIGHT),
    }
}

impl ExtraRules for Grounding {}

/// The mesh units a ground stain may reach up at most.
pub const MAX_GROUNDING_HEIGHT: f32 = 10_000.0;

settings! {
    unauthored;
    /// Light passing through a surface instead of diffusing off it:
    /// glTF's `KHR_materials_transmission`, `KHR_materials_ior` and the
    /// thickness of `KHR_materials_volume`. [`Material::with_transmission`]
    /// checks the rules.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Transmission {
        /// The share of light passing through, tinted by the base color; 0 is
        /// an ordinary surface.
        pub factor: f32 = Transmission::NONE.factor => range(0.0, 1.0),
        /// Index of refraction: how far rays bend entering the surface (1.5
        /// for glass, 1.33 for water).
        pub ior: f32 = Transmission::NONE.ior => range(1.0, 3.0),
        /// Depth of the volume beneath the surface, in world units: how far a
        /// bent ray travels before it leaves. 0 is a thin wall, which does not
        /// bend what shows through it.
        pub thickness: f32 = Transmission::NONE.thickness => at_least(0.0),
    }
}

impl ExtraRules for Transmission {}

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
    /// Drawn transparent: alpha is the fraction of light behind
    /// the surface it covers.
    Blend,
}

/// A glTF 2.0 metallic-roughness material: a linear base color
/// factor, [`Surface`] factors and extensions, and optional texture maps.
///
/// Materials are opaque unless given an opacity below 1
/// ([`Material::with_opacity`]) or another [`AlphaMode`]; the
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
    grounding: Option<Grounding>,
}

impl Material {
    /// The rules of the material's own values, which its builders check and
    /// authored materials' schemas state: each color channel,
    pub const COLOR: Rule = Rule::range(0.0, 1.0);
    /// the opacity,
    pub const OPACITY: Rule = Rule::above_to(0.0, 1.0);
    /// a mask's alpha cutoff,
    pub const ALPHA_CUTOFF: Rule = Rule::at_least(0.0);
    /// the normal map's scale,
    pub const NORMAL_SCALE: Rule = Rule::finite();
    /// the occlusion map's strength,
    pub const OCCLUSION_STRENGTH: Rule = Rule::range(0.0, 1.0);
    /// each emissive channel,
    pub const EMISSIVE: Rule = Rule::at_least(0.0);
    /// and the meters one repeat of world-mapped maps covers.
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
            transmission: Transmission::NONE,
            world_tiling: None,
            top: None,
            grounding: None,
        })
    }

    /// This material with alpha `opacity` in `(0, 1]`: the
    /// fraction of light behind it that it covers. Below 1, an opaque
    /// material becomes [`AlphaMode::Blend`]. Transparent surfaces blend in
    /// any order, write no depth and cast no shadow.
    pub fn with_opacity(mut self, opacity: f32) -> Result<Self, AssetError> {
        if opacity.check("opacity", Self::OPACITY).is_err() {
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
            AlphaMode::Mask { cutoff }
                if cutoff.check("alpha_cutoff", Self::ALPHA_CUTOFF).is_err() =>
            {
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
    /// light passes through it.
    pub fn is_transparent(&self) -> bool {
        self.alpha_mode == AlphaMode::Blend || self.transmission.factor > 0.0
    }

    /// This material letting light through. Transmissive
    /// materials draw in the transparent pass; with the `Refraction`
    /// resource they show the opaque scene behind them, bent and blurred,
    /// and without it they blend as plain transparency.
    pub fn with_transmission(mut self, transmission: Transmission) -> Result<Self, AssetError> {
        transmission
            .validate()
            .map_err(|invalid| invalid_part(None, invalid))?;
        self.transmission = transmission;
        Ok(self)
    }

    pub fn transmission(&self) -> Transmission {
        self.transmission
    }

    /// This material with its maps laid from the world's axes, one repeat
    /// every `tiling` meters, rather than by texture coordinates:
    /// each map blends its projections along the axes the surface
    /// faces most, so meshes need no coordinates, show no seams and keep
    /// one texel density at any size. Map transforms still apply. For
    /// surfaces that stay where they are, such as rocks and cliffs: maps
    /// slide over moving ones. `tiling` is finite, positive and at most
    /// [`MAX_WORLD_TILING`].
    pub fn with_world_mapping(mut self, tiling: f32) -> Result<Self, AssetError> {
        if tiling.check("world_tiling", Self::WORLD_TILING).is_err() {
            return Err(AssetError::InvalidSurface {
                field: "world_tiling".into(),
            });
        }
        self.world_tiling = Some(tiling);
        Ok(self)
    }

    /// The meters one repeat of world-mapped maps covers, if laid so.
    pub fn world_tiling(&self) -> Option<f32> {
        self.world_tiling
    }

    /// This material with `top` settling on its faces turned up.
    pub fn with_top_layer(mut self, top: TopLayer) -> Result<Self, AssetError> {
        top.validate()
            .map_err(|invalid| invalid_part(Some("top"), invalid))?;
        self.top = Some(top);
        Ok(self)
    }

    pub fn top_layer(&self) -> Option<TopLayer> {
        self.top
    }

    /// This material with the ground staining its base where it stands on
    /// terrain.
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

    /// This material with `surface`, whose values must lie in their ranges.
    pub fn with_surface(mut self, surface: Surface) -> Result<Self, AssetError> {
        surface
            .validate()
            .map_err(|invalid| invalid_part(None, invalid))?;
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
        if scale.check("normal_scale", Self::NORMAL_SCALE).is_err() {
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

    /// This material emitting linear `color` (glTF's emissive factor times
    /// its emissive strength): finite and not negative.
    pub fn with_emissive(mut self, color: [f32; 3]) -> Result<Self, AssetError> {
        if color.check("emissive", Self::EMISSIVE).is_err() {
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
    #[must_use]
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
