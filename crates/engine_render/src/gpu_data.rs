//! Explicit schemas for the adjacent shader, separate from gameplay components.

pub(crate) use crate::gpu_layouts::scene::{CASTER_PLANES, MAX_CASCADES, MAX_LEVELS};
use engine_core::{
    AlphaMode, Atmosphere, Bloom, Material, MeshVertex, Reflections, RenderInstance, Surface,
    UvTransform, glam::Mat4,
};

use crate::gpu_layouts::constants;

pub(crate) use crate::gpu_layouts::constants::COMPOSED_BYTES;
pub(crate) use crate::gpu_layouts::constants::DRAW_ARGS_BYTES;
pub(crate) use crate::gpu_layouts::constants::FLAG_PHASE1;
pub(crate) use crate::gpu_layouts::constants::FLAG_PHASE2;
pub(crate) use crate::gpu_layouts::constants::FLAG_PROBE;
pub(crate) use crate::gpu_layouts::constants::FLAG_VISIBLE;
pub(crate) use crate::gpu_layouts::constants::LOD_LEVELS;
pub(crate) use crate::gpu_layouts::constants::LOD_STATE;
pub(crate) use crate::gpu_layouts::constants::MOVED_SLOT;
pub(crate) use crate::gpu_layouts::constants::STILL_FRAMES;
pub(crate) use crate::gpu_layouts::scene::CameraUniform;

impl CameraUniform {
    /// An unjittered camera at `position` without temporal history, at
    /// presentation time `seconds`.
    pub(crate) fn exact(view_projection: Mat4, position: glam::Vec3, seconds: f32) -> Self {
        let columns = view_projection.to_cols_array_2d();
        Self {
            view_projection: columns,
            previous_view_projection: columns,
            inverse_view_projection: view_projection.inverse().to_cols_array_2d(),
            position: position.extend(seconds).to_array(),
            jitter: [0.0; 4],
            options: [0.0; 4],
            wind: [0.0, 0.0, 0.0, seconds],
        }
    }

    /// A reflection probe's cube face `size` texels a side seen from
    /// `position` through `view_projection`, at presentation
    /// time `seconds`.
    pub(crate) fn probe(
        view_projection: Mat4,
        position: glam::Vec3,
        seconds: f32,
        size: u32,
    ) -> Self {
        Self {
            options: [1.0, pixel_angle(view_projection, size), 0.0, 0.0],
            ..Self::exact(view_projection, position, seconds)
        }
    }

    /// A temporally antialiased frame of a `width` x `height` viewport:
    /// `view_projection` jittered by `jitter` (see [`jitter_clip`]), with the
    /// previous frame's unjittered `previous` for motion vectors.
    pub(crate) fn temporal(
        view_projection: Mat4,
        position: glam::Vec3,
        seconds: f32,
        previous: Mat4,
        jitter: [f32; 2],
        (width, height): (u32, u32),
        frame: u64,
    ) -> Self {
        // The golden ratio's fraction turns each frame's dither far from the
        // last few.
        let turn = (frame % 1024) as f32 * 0.618_034;
        Self {
            view_projection: jittered(view_projection, jitter).to_cols_array_2d(),
            previous_view_projection: previous.to_cols_array_2d(),
            jitter: [
                jitter[0],
                jitter[1],
                2.0 / width.max(1) as f32,
                2.0 / height.max(1) as f32,
            ],
            options: [0.0, 0.0, turn - turn.floor(), 0.0],
            ..Self::exact(view_projection, position, seconds)
        }
    }
}

/// The angle (radians) a pixel spans at the center of a view `height`
/// pixels tall seen through `view_projection`, which shading measures
/// pixels' footprints by.
pub(crate) fn pixel_angle(view_projection: Mat4, height: u32) -> f32 {
    let row = view_projection.row(1).truncate().length() * height.max(1) as f32;
    2.0 / engine_core::math::max(row, 1.0e-12)
}

pub(crate) use crate::gpu_layouts::scene::LightingUniform;

/// Weight of the grade's S-curve for `contrast`: the shader mixes a
/// perceptual value `s` toward `smoothstep(0, 1, s)` by it, a curve that keeps
/// black and white, is monotonic for weights in `[-2, 1]` and has slope
/// `1 + weight / 2` at mid grey, which is the contrast from 0 up to 1.5.
pub(crate) fn contrast_weight(contrast: f32) -> f32 {
    engine_core::math::clamp(2.0 * (contrast - 1.0), -2.0, 1.0)
}

/// `tint` scaled to unit luminance, so it shifts hue without darkening; white
/// for tints without luminance.
fn unit_luminance(tint: glam::Vec3) -> [f32; 4] {
    let luminance = tint.dot(glam::Vec3::new(0.2126, 0.7152, 0.0722));
    if luminance > 1.0e-6 {
        (tint / luminance).extend(0.0).to_array()
    } else {
        [1.0, 1.0, 1.0, 0.0]
    }
}

impl LightingUniform {
    pub(crate) fn new(
        atmosphere: &Atmosphere,
        bloom: &Bloom,
        reflections: &Reflections,
        refraction: &engine_core::Refraction,
        sky: &engine_core::Sky,
    ) -> Self {
        let atmosphere = atmosphere.sanitized();
        let reflections = reflections.sanitized();
        let refraction = refraction.sanitized();
        let sky = sky.sanitized();
        // With the physically based sky the air hazes what
        // lies far away by its own extinction, thinning with height as its
        // scattering does; no fog. Its colors come from the sky's tables.
        let (fog, aerial, physical) = if sky.enabled {
            let altitude = sky.altitude * 0.001;
            let rayleigh = (sky.rayleigh[0] + sky.rayleigh[1] + sky.rayleigh[2]) / 3.0
                * (-altitude / sky.rayleigh_height).exp();
            let mie = (sky.mie + sky.mie_absorption) * (-altitude / sky.mie_height).exp();
            let extinction = rayleigh + mie;
            let height = if extinction > 0.0 {
                (rayleigh * sky.rayleigh_height + mie * sky.mie_height) / extinction
            } else {
                sky.rayleigh_height
            };
            (
                [0.0; 4],
                [0.0, extinction * 0.001 * sky.aerial, 0.001 / height, 0.0],
                1.0,
            )
        } else {
            (
                [
                    atmosphere.fog_start,
                    atmosphere.fog_density,
                    atmosphere.fog_falloff,
                    atmosphere.fog_max,
                ],
                [
                    atmosphere.aerial_start,
                    atmosphere.aerial_density,
                    atmosphere.aerial_falloff,
                    atmosphere.aerial_glow,
                ],
                0.0,
            )
        };
        Self {
            sun: atmosphere
                .sun_direction
                .normalize()
                .extend(atmosphere.rim)
                .to_array(),
            sun_color: atmosphere.sun_color.extend(atmosphere.specular).to_array(),
            zenith: atmosphere.zenith.extend(atmosphere.ambient).to_array(),
            horizon: atmosphere.horizon.extend(atmosphere.exposure).to_array(),
            ground: atmosphere.ground.extend(atmosphere.saturation).to_array(),
            fog,
            effects: [
                atmosphere.emission,
                bloom.sanitized().threshold,
                reflections.max_roughness,
                if reflections.enabled() { 1.0 } else { 0.0 },
            ],
            grade: [
                contrast_weight(atmosphere.contrast),
                atmosphere.vibrance,
                physical,
                0.0,
            ],
            shadow_tint: unit_luminance(atmosphere.shadow_tint),
            highlight_tint: unit_luminance(atmosphere.highlight_tint),
            refraction: [refraction.strength, refraction.blur, 0.0, 0.0],
            aerial,
            aerial_color: atmosphere.aerial_color.extend(0.0).to_array(),
            occupancy: [0.0; 4],
            occupancy_heights: [0.0; 4],
            occupancy_far: [0.0; 4],
            underwater: [0.0; 4],
            sea: [0.0; 4],
            cloud_shadow: [0.0; 4],
            cloud_layer: [0.0; 4],
        }
    }
}

/// A surface as four unsigned 8-bit fractions: roughness,
/// metalness, emission over [`Surface::MAX_EMISSION`], and ripples. Any
/// values pack: out-of-range ones clamp and NaN reads as matte.
pub(crate) fn pack_surface(surface: Surface) -> u32 {
    let unit = |value: f32, fallback: f32, max: f32| {
        let value = if value.is_nan() { fallback } else { value };
        (engine_core::math::clamp(value / max, 0.0, 1.0) * 255.0).round() as u32
    };
    let matte = Surface::MATTE;
    unit(surface.roughness, matte.roughness, 1.0)
        | unit(surface.metalness, matte.metalness, 1.0) << 8
        | unit(surface.emission, matte.emission, Surface::MAX_EMISSION) << 16
        | unit(surface.ripples, matte.ripples, 1.0) << 24
}

/// Linear color and opacity as four 16-bit fractions in two
/// words: red and green, then blue and opacity (WGSL `unpack2x16unorm`, or a
/// `Unorm16x4` vertex attribute). Out-of-range values clamp; NaN color reads
/// as 0 and NaN opacity as opaque.
pub(crate) fn pack_color(color: [f32; 4]) -> [u32; 2] {
    let unit = |value: f32, fallback: f32| {
        let value = if value.is_nan() { fallback } else { value };
        (engine_core::math::clamp(value, 0.0, 1.0) * 65535.0).round() as u32
    };
    [
        unit(color[0], 0.0) | unit(color[1], 0.0) << 16,
        unit(color[2], 0.0) | unit(color[3], 1.0) << 16,
    ]
}

/// `view_projection` followed by a clip-space translation of `jitter` units
/// per w: the projection a jittered frame rasterizes, and culls with.
pub(crate) fn jittered(view_projection: Mat4, jitter: [f32; 2]) -> Mat4 {
    Mat4::from_cols(
        glam::Vec4::X,
        glam::Vec4::Y,
        glam::Vec4::Z,
        glam::Vec4::new(jitter[0], jitter[1], 0.0, 1.0),
    ) * view_projection
}

pub(crate) use crate::gpu_layouts::scene::PoseData;

pub(crate) use crate::gpu_layouts::scene::ColorData;

pub(crate) use crate::gpu_layouts::scene::GroundData;

/// The ground of an object at `scale` whose base sinks `sink` meters below
/// ground of normal `normal` (x and z), stained as its material's
/// `grounding` says.
pub(crate) fn ground_data(
    grounding: Option<engine_core::Grounding>,
    normal: [f32; 2],
    sink: f32,
    scale: f32,
) -> GroundData {
    let Some(grounding) = grounding else {
        return [0; 4];
    };
    let snorm = |value: f32| {
        u32::from((engine_core::math::clamp(value, -1.0, 1.0) * 32767.0).round() as i16 as u16)
    };
    let byte = |value: f32| (engine_core::math::clamp(value, 0.0, 1.0) * 255.0).round() as u32;
    let [red, green, blue] = grounding.color;
    [
        snorm(normal[0]) | snorm(normal[1]) << 16,
        sink.to_bits(),
        (grounding.height * scale).to_bits(),
        byte(red) | byte(green) << 8 | byte(blue) << 16 | 255 << 24,
    ]
}

impl From<&RenderInstance> for ColorData {
    fn from(instance: &RenderInstance) -> Self {
        Self {
            color: pack_color(instance.color),
            surface: pack_surface(instance.surface),
            // One more than the offset of a skinned slot's joint palette;
            // 0 draws rigidly.
            flags: instance.skin.map_or(0, |offset| offset + 1),
        }
    }
}

pub(crate) use crate::gpu_layouts::scene::Vertex;

impl From<&MeshVertex> for Vertex {
    fn from(vertex: &MeshVertex) -> Self {
        Self {
            position: vertex.position,
            normal: octahedral(vertex.normal),
            uv: vertex.uv,
        }
    }
}

pub(crate) use crate::gpu_layouts::scene::ExtraVertex;

impl ExtraVertex {
    /// The extra stream of `mesh`, or `None` without colors, second
    /// coordinates or joints: skinned meshes' entry points
    /// read it too.
    pub(crate) fn stream(mesh: &engine_core::Mesh) -> Option<Vec<Self>> {
        if mesh.colors().is_none() && mesh.uv1().is_none() && mesh.joints().is_none() {
            return None;
        }
        Some(
            mesh.vertices()
                .iter()
                .enumerate()
                .map(|(index, vertex)| Self {
                    color: mesh.colors().map_or([u16::MAX; 4], |colors| {
                        colors[index].map(|value| {
                            (engine_core::math::clamp(value, 0.0, 1.0) * 65535.0).round() as u16
                        })
                    }),
                    uv1: mesh.uv1().map_or(vertex.uv, |uv1| uv1[index]),
                })
                .collect(),
        )
    }
}

pub(crate) use crate::gpu_layouts::scene::SkinVertex;

impl SkinVertex {
    /// The skin stream of `mesh`, or `None` without joints.
    pub(crate) fn stream(mesh: &engine_core::Mesh) -> Option<Vec<Self>> {
        let (joints, weights) = (mesh.joints()?, mesh.weights()?);
        Some(
            joints
                .iter()
                .zip(weights)
                .map(|(joints, weights)| Self {
                    joints: *joints,
                    weights: weights.map(|weight| {
                        (engine_core::math::clamp(weight, 0.0, 1.0) * 65535.0).round() as u16
                    }),
                })
                .collect(),
        )
    }
}

/// A nonzero normal as two 16-bit signed fractions: its direction on the
/// octahedron |x| + |y| + |z| = 1, the lower half folded over the upper
/// (Cigolle et al. 2014). `meshes/vertex.wgsl` decodes it with `octahedral_normal`;
/// directions keep about 0.003 degrees.
pub(crate) fn octahedral(normal: [f32; 3]) -> [i16; 2] {
    let [x, y, z] = normal;
    let length = x.abs() + y.abs() + z.abs();
    let (u, v) = (x / length, y / length);
    let sign = |value: f32| if value >= 0.0 { 1.0 } else { -1.0 };
    let (u, v) = if z < 0.0 {
        ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v))
    } else {
        (u, v)
    };
    [u, v].map(|value| (engine_core::math::clamp(value, -1.0, 1.0) * 32767.0).round() as i16)
}

pub(crate) use crate::shaders::{
    CLOUDS_SHADER, COVER_GROW_SHADER, DEFORMATION_SHADER, DEPTH_PYRAMID_SHADER, ENVIRONMENT_SHADER,
    LIGHT_SHADER, MIPMAP_SHADER, OCCUPANCY_SHADER, PARTICLE_SHADER, POST_SHADER,
    PRESENTATION_SHADER, SCATTER_CULL_SHADER, SCATTER_SNAP_SHADER, SHADER, SHADOW_SHADER,
    SHAFTS_SHADER, SKY_SHADER, TEMPORAL_SHADER, TERRAIN_CONTROL_SHADER, TERRAIN_HORIZON_SHADER,
    TERRAIN_NORMALS_SHADER, TERRAIN_SHADOW_SHADER, TERRAIN_SKY_SHADER, TRANSPARENT_SHADER,
    VIRTUAL_SHADER, VISIBILITY_SHADER, WATER_WAVES_SHADER,
};

pub(crate) use crate::gpu_layouts::scene::MaterialUniform;

impl MaterialUniform {
    /// Maps of materials without them: identity transforms and no emission.
    pub(crate) const DEFAULT: Self = Self {
        transforms: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        emissive: [0.0; 4],
        params: [1.0, 1.0, 0.5, 0.0],
        volume: [0.0, 1.5, 0.0, 0.0],
        top: [0.0; 4],
        layer: [0.0; 4],
    };

    pub(crate) fn new(material: &Material) -> Self {
        let mut uniform = Self::DEFAULT;
        for (index, map) in material.maps().iter().enumerate() {
            if let Some(map) = map {
                let [first, second] = transform_rows(&map.transform);
                uniform.transforms[index * 2] = first;
                uniform.transforms[index * 2 + 1] = second;
            }
        }
        let [r, g, b] = material.emissive();
        uniform.emissive = [r, g, b, material.world_tiling().unwrap_or(0.0)];
        if let Some(top) = material.top_layer() {
            let [r, g, b] = top.color;
            uniform.top = [r, g, b, top.coverage];
            uniform.layer = [top.roughness, 0.0, 0.0, 0.0];
        }
        let cutoff = match material.alpha_mode() {
            AlphaMode::Mask { cutoff } => cutoff,
            AlphaMode::Opaque | AlphaMode::Blend => 0.5,
        };
        // Maps reading the second texture coordinates, one bit each.
        let second = material
            .maps()
            .iter()
            .enumerate()
            .filter(|(_, map)| map.is_some_and(|map| map.tex_coord == 1))
            .fold(0_u32, |bits, (index, _)| bits | 1 << index);
        uniform.params = [
            material.normal_scale(),
            material.occlusion_strength(),
            cutoff,
            second as f32,
        ];
        let transmission = material.transmission();
        uniform.volume = [
            transmission.factor,
            transmission.ior,
            transmission.thickness,
            f32::from_bits(Variant::of(material).bits()),
        ];
        uniform
    }
}

/// The two rows of `transform`'s 2x3 matrix: coordinates scaled, rotated
/// and offset (`KHR_texture_transform`, as Khronos' sample renderer reads it).
pub(crate) fn transform_rows(transform: &UvTransform) -> [[f32; 4]; 2] {
    let (sin, cos) = engine_core::math::sin_cos(transform.rotation);
    let [sx, sy] = transform.scale;
    let [ox, oy] = transform.offset;
    [
        [cos * sx, sin * sy, ox, 0.0],
        [-sin * sx, cos * sy, oy, 0.0],
    ]
}

/// A material's pipeline variant: the `meshes/declarations.wgsl` override
/// constants it sets and whether back faces draw. Materials without maps and
/// plain opaque base color textures use [`Variant::BASIC`], the shading of
/// every material before glTF maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub(crate) struct Variant(u16);

impl Variant {
    pub(crate) const BASIC: Self = Self(0);
    const MAPS: u16 = constants::VARIANT_MAPS;
    const NORMAL_MAP: u16 = constants::VARIANT_NORMAL_MAP;
    const ALPHA_MASK: u16 = 4;
    const DOUBLE_SIDED: u16 = constants::VARIANT_DOUBLE_SIDED;
    // Meshes with a second vertex stream: extended entry
    // points rather than a constant.
    const EXTENDED: u16 = 16;
    // Skinned slots' meshes: skinned entry points, which
    // read the second stream and a skin stream.
    const SKINNED: u16 = 32;
    // Transmissive materials: glass shading in the
    // transparent pass.
    const TRANSMISSION: u16 = 64;
    // Maps a material with maps has: the others are not sampled, as the
    // white texel they would read changes nothing. Occlusion packed in the
    // metallic-roughness texture's red (the same texture, sampler, transform
    // and coordinates) is read from its sample.
    const METALLIC_ROUGHNESS_MAP: u16 = constants::VARIANT_METALLIC_ROUGHNESS_MAP;
    const OCCLUSION_MAP: u16 = constants::VARIANT_OCCLUSION_MAP;
    const EMISSIVE_MAP: u16 = constants::VARIANT_EMISSIVE_MAP;
    const PACKED_OCCLUSION: u16 = constants::VARIANT_PACKED_OCCLUSION;
    // Passes drawn rarely read the map bits from the
    // material uniform: their pipelines key on pipeline state alone.
    const UBER: u16 = 2048;
    // Maps laid from the world's axes, and a layer on faces turned up.
    const WORLD_MAPPING: u16 = constants::VARIANT_WORLD_MAPPING;
    const TOP_LAYER: u16 = constants::VARIANT_TOP_LAYER;
    // Bases the ground stains: slot vertex stages read
    // placed objects' grounds.
    const GROUNDING: u16 = 16384;
    // Placed objects fading between how the camera shows them:
    // their lists' entries carry their fades, which their pixels
    // dither by. A draw's, not a material's.
    const FADING: u16 = 32768;

    pub(crate) fn of(material: &Material) -> Self {
        let mode = material.alpha_mode();
        let base = material.base_color_texture();
        let maps = material.metallic_roughness_texture().is_some()
            || material.normal_texture().is_some()
            || material.occlusion_texture().is_some()
            || material.emissive_texture().is_some()
            || material.emissive() != [0.0; 3]
            || base.is_some_and(|map| map.transform != UvTransform::IDENTITY)
            || matches!(mode, AlphaMode::Mask { .. })
            || (mode == AlphaMode::Blend && base.is_some())
            || material.world_tiling().is_some();
        let mut bits = 0;
        if maps {
            bits |= Self::MAPS;
            let metallic_roughness = material.metallic_roughness_texture();
            let occlusion = material.occlusion_texture();
            if metallic_roughness.is_some() {
                bits |= Self::METALLIC_ROUGHNESS_MAP;
            }
            if occlusion.is_some() && occlusion == metallic_roughness {
                bits |= Self::PACKED_OCCLUSION;
            } else if occlusion.is_some() {
                bits |= Self::OCCLUSION_MAP;
            }
            if material.emissive_texture().is_some() {
                bits |= Self::EMISSIVE_MAP;
            }
        }
        if material.normal_texture().is_some() {
            bits |= Self::NORMAL_MAP;
        }
        if matches!(mode, AlphaMode::Mask { .. }) {
            bits |= Self::ALPHA_MASK;
        }
        if material.is_double_sided() {
            bits |= Self::DOUBLE_SIDED;
        }
        if material.transmission().factor > 0.0 {
            bits |= Self::TRANSMISSION;
        }
        if material.world_tiling().is_some() {
            bits |= Self::WORLD_MAPPING;
        }
        if material.top_layer().is_some() {
            bits |= Self::TOP_LAYER;
        }
        if material.grounding().is_some() {
            bits |= Self::GROUNDING;
        }
        Self(bits)
    }

    /// This variant for a pass that reads map presence from the material
    /// uniform: only what its pipeline state needs, vertex
    /// streams, culling, alpha masking and transmission.
    pub(crate) fn uber(self) -> Self {
        let state = Self::EXTENDED
            | Self::SKINNED
            | Self::ALPHA_MASK
            | Self::DOUBLE_SIDED
            | Self::TRANSMISSION;
        Self(self.0 & state | Self::UBER)
    }

    /// The bits the material uniform carries (`volume.w`).
    pub(crate) fn bits(self) -> u32 {
        u32::from(self.0)
    }

    /// Override constants: fragment stages read them all, slot vertex
    /// stages GROUNDING and FADE_LIST.
    pub(crate) fn constants(self) -> [(&'static str, f64); 14] {
        let set = |bit: u16| if self.0 & bit != 0 { 1.0 } else { 0.0 };
        [
            ("WORLD_MAPPING", set(Self::WORLD_MAPPING)),
            ("TOP_LAYER", set(Self::TOP_LAYER)),
            ("GROUNDING", set(Self::GROUNDING)),
            ("FADE_LIST", set(Self::FADING)),
            ("MATERIAL_MAPS", set(Self::MAPS)),
            ("NORMAL_MAP", set(Self::NORMAL_MAP)),
            ("ALPHA_MASK", set(Self::ALPHA_MASK)),
            ("DOUBLE_SIDED", set(Self::DOUBLE_SIDED)),
            ("TRANSMISSION", set(Self::TRANSMISSION)),
            ("METALLIC_ROUGHNESS_MAP", set(Self::METALLIC_ROUGHNESS_MAP)),
            ("OCCLUSION_MAP", set(Self::OCCLUSION_MAP)),
            ("EMISSIVE_MAP", set(Self::EMISSIVE_MAP)),
            ("PACKED_OCCLUSION", set(Self::PACKED_OCCLUSION)),
            ("UBER_MATERIALS", set(Self::UBER)),
        ]
    }

    pub(crate) fn transmission(self) -> bool {
        self.0 & Self::TRANSMISSION != 0
    }

    pub(crate) fn double_sided(self) -> bool {
        self.0 & Self::DOUBLE_SIDED != 0
    }

    pub(crate) fn alpha_mask(self) -> bool {
        self.0 & Self::ALPHA_MASK != 0
    }

    /// This variant drawing a mesh with a second vertex stream, or not.
    pub(crate) fn with_extended(self, extended: bool) -> Self {
        Self(if extended {
            self.0 | Self::EXTENDED
        } else {
            self.0 & !Self::EXTENDED
        })
    }

    pub(crate) fn extended(self) -> bool {
        self.0 & Self::EXTENDED != 0
    }

    /// This variant drawing a mesh with joints through retained slots, or
    /// not; skinned variants are extended.
    pub(crate) fn with_skinned(self, skinned: bool) -> Self {
        Self(if skinned {
            self.0 | Self::SKINNED | Self::EXTENDED
        } else {
            self.0 & !Self::SKINNED
        })
    }

    pub(crate) fn skinned(self) -> bool {
        self.0 & Self::SKINNED != 0
    }

    /// This variant drawing placed objects' fading lists, or
    /// not.
    pub(crate) fn with_fading(self, fading: bool) -> Self {
        Self(if fading {
            self.0 | Self::FADING
        } else {
            self.0 & !Self::FADING
        })
    }
}

pub(crate) use crate::gpu_layouts::scene::ShadowUniform;

pub(crate) use crate::gpu_layouts::scene::ShadowCullUniform;

pub(crate) use crate::gpu_layouts::scene::PostUniform;

/// Levels of a full mip chain down to 1x1 for a `width` x `height` texture.
pub(crate) fn mip_levels(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).max(1).leading_zeros()
}

/// RGBA8 bytes of every level of that chain after the first.
pub(crate) fn mip_chain_bytes(width: u32, height: u32) -> u64 {
    (1..mip_levels(width, height))
        .map(|level| u64::from((width >> level).max(1)) * u64::from((height >> level).max(1)) * 4)
        .sum()
}

pub(crate) use crate::gpu_layouts::scene::ResolveUniform;

/// Samples in the jitter sequence before it repeats.
pub(crate) const JITTER_SAMPLES: u32 = 8;

/// Radical inverse of `index` in `base`: the Halton sequence.
fn halton(mut index: u32, base: u32) -> f32 {
    let (mut scale, mut value) = (1.0, 0.0);
    while index > 0 {
        scale /= base as f32;
        value += scale * (index % base) as f32;
        index /= base;
    }
    value
}

/// Sub-pixel offset of display frame `frame` in pixels, in `[-0.5, 0.5)`:
/// the Halton (2, 3) sequence of [`JITTER_SAMPLES`] samples, from its
/// second point (the first is the pixel corner).
pub(crate) fn jitter_offset(frame: u32) -> [f32; 2] {
    let index = frame % JITTER_SAMPLES + 1;
    [halton(index, 2) - 0.5, halton(index, 3) - 0.5]
}

/// A pixel offset as the clip-space jitter a vertex shader adds to
/// `xy / w`: two units span the viewport, and clip y points up.
pub(crate) fn jitter_clip(offset: [f32; 2], width: u32, height: u32) -> [f32; 2] {
    [
        2.0 * offset[0] / width.max(1) as f32,
        -2.0 * offset[1] / height.max(1) as f32,
    ]
}

pub(crate) use crate::gpu_layouts::scene::PresentationFrame;

pub(crate) use crate::gpu_layouts::scene::CandidateData;

pub(crate) use crate::gpu_layouts::scene::BatchData;

pub(crate) use crate::gpu_layouts::scene::SkinnedData;

/// Clip-space depth of the near and far planes: depth is reversed,
/// so nearer is larger and the depth buffer clears to the far plane.
pub(crate) const NEAR_DEPTH: f32 = 1.0;
pub(crate) const FAR_DEPTH: f32 = 0.0;

/// Inward clip planes (WebGPU depth range) of a view-projection matrix and its
/// absolute value, as the CPU culling test derives them.
pub(crate) fn frustum_uniform(view_projection: glam::Mat4) -> ([[f32; 4]; 6], [[f32; 4]; 4]) {
    let rows = view_projection.transpose();
    let planes = [
        rows.w_axis + rows.x_axis,
        rows.w_axis - rows.x_axis,
        rows.w_axis + rows.y_axis,
        rows.w_axis - rows.y_axis,
        rows.z_axis,
        rows.w_axis - rows.z_axis,
    ]
    .map(|plane| plane.to_array());
    let absolute = glam::Mat4::from_cols(
        view_projection.x_axis.abs(),
        view_projection.y_axis.abs(),
        view_projection.z_axis.abs(),
        view_projection.w_axis.abs(),
    );
    (planes, absolute.to_cols_array_2d())
}

pub(crate) fn clamped_extent(width: u32, height: u32, limit: u32) -> Option<(u32, u32)> {
    if width == 0 || height == 0 || limit == 0 {
        return None;
    }
    let largest = width.max(height);
    if largest <= limit {
        return Some((width, height));
    }
    let fit = |dimension: u32| {
        ((u64::from(dimension) * u64::from(limit)) / u64::from(largest)).max(1) as u32
    };
    Some((fit(width), fit(height)))
}
