pub(crate) use crate::gpu_layouts::scene::MAX_LEVELS;
use engine_core::{
    AlphaMode, Atmosphere, Bloom, Material, MeshVertex, Reflections, RenderInstance, Surface,
    UvTransform, glam::Mat4,
};

use crate::gpu_layouts::constants;
use crate::targets::FrameRegion;

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
            region: [0.0; 4],
            detail: position.extend(0.0).to_array(),
        }
    }

    pub(crate) fn set_region(&mut self, view_projection: Mat4, region: FrameRegion) {
        self.options[1] = pixel_angle(view_projection, region.display.height);
        self.options[3] = region.mip_bias();
        self.region = region.uniform();
    }

    pub(crate) fn set_detail(&mut self, eye: glam::Vec3) {
        self.detail = eye.extend(0.0).to_array();
    }

    pub(crate) fn probe(
        view_projection: Mat4,
        position: glam::Vec3,
        seconds: f32,
        size: u32,
    ) -> Self {
        Self {
            options: [1.0, pixel_angle(view_projection, size), 0.0, 0.0],
            region: [size as f32, size as f32, 1.0, 1.0],
            ..Self::exact(view_projection, position, seconds)
        }
    }

    pub(crate) fn temporal(
        view_projection: Mat4,
        position: glam::Vec3,
        seconds: f32,
        previous: Mat4,
        jitter: [f32; 2],
        (width, height): (u32, u32),
        frame: u64,
    ) -> Self {
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

pub(crate) fn pixel_angle(view_projection: Mat4, height: u32) -> f32 {
    let row = view_projection.row(1).truncate().length() * height.max(1) as f32;
    2.0 / engine_core::math::max(row, 1.0e-12)
}

pub(crate) use crate::gpu_layouts::scene::LightingUniform;

pub(crate) fn contrast_weight(contrast: f32) -> f32 {
    engine_core::math::clamp(2.0 * (contrast - 1.0), -2.0, 1.0)
}

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
        (sky, highlights, windows): (
            &engine_core::Sky,
            &engine_core::Highlights,
            &engine_core::Windows,
        ),
        (foliage, temporal): (&engine_core::Foliage, bool),
    ) -> Self {
        let atmosphere = atmosphere.sanitized();
        let reflections = reflections.sanitized();
        let refraction = refraction.sanitized();
        let sky = sky.sanitized();
        let foliage = foliage.sanitized();
        let highlights = highlights.sanitized();
        let windows = windows.sanitized();
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
            occupancy_maps: [0.0; 4],
            occupancy_centers: [[0.0; 4]; crate::gpu_layouts::scene::MAX_OCCUPANCY_LEVELS / 2],
            underwater: [0.0; 4],
            sea: [0.0; 4],
            cloud_shadow: [0.0; 4],
            cloud_layer: [0.0; 4],
            night: [0.0; 4],
            star_turn: [0.0, 0.0, 0.0, 1.0],
            moon: [0.0; 4],
            night_shape: [0.0; 4],
            foliage: [
                foliage.transmission,
                foliage.variation,
                if foliage.hashed_cutoff && temporal {
                    1.0
                } else {
                    0.0
                },
                foliage.ground_blend,
            ],
            highlights: sized_highlights(&highlights, &sky),
            windows: if windows.enabled {
                let rise = atmosphere.sun_direction.normalize().y;
                let elevation = engine_core::math::clamp(rise, -1.0, 1.0)
                    .asin()
                    .to_degrees();
                [elevation, windows.dusk, windows.spread, windows.dark]
            } else {
                [0.0; 4]
            },
            haze_march: [0.0; 4],
        }
    }
}

fn sized_highlights(highlights: &engine_core::Highlights, sky: &engine_core::Sky) -> [f32; 4] {
    let cap = if highlights.capped {
        highlights.cap
    } else {
        0.0
    };
    if !highlights.enabled {
        return if cap > 0.0 {
            [1.0, 0.0, cap, 0.0]
        } else {
            [0.0; 4]
        };
    }
    let radius = if sky.enabled {
        sky.sun_size
    } else {
        highlights.sun_size
    };
    let (sine, cosine) = engine_core::math::sin_cos(radius.to_radians());
    [cosine, sine, cap, highlights.light_radius]
}

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
    [
        snorm(normal[0]) | snorm(normal[1]) << 16,
        sink.to_bits(),
        (grounding.height * scale).to_bits(),
        0,
    ]
}

impl From<&RenderInstance> for ColorData {
    fn from(instance: &RenderInstance) -> Self {
        Self {
            color: pack_color(instance.color),
            surface: pack_surface(instance.surface),
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
    BOUNCE_SHADER, CLOUDS_SHADER, COVER_GROW_SHADER, DEBUG_VIEW_SHADER, DEFORMATION_SHADER,
    DEPTH_PYRAMID_SHADER, ENVIRONMENT_SHADER, EXPOSURE_SHADER, HAZE_SHADER, LIGHT_SHADER,
    MIPMAP_SHADER, OCCUPANCY_SHADER, PARTICLE_SHADER, POST_SHADER, PRESENTATION_SHADER,
    REFLECTIONS_SHADER, SCATTER_CULL_SHADER, SCATTER_SNAP_SHADER, SHADER, SHADOW_MARKS_SHADER,
    SHADOW_SHADER, SKY_SHADER, TEMPORAL_SHADER, TERRAIN_CONTROL_SHADER, TERRAIN_NORMALS_SHADER,
    TERRAIN_SHADOW_SHADER, TRANSPARENT_SHADER, VIRTUAL_SHADER, VISIBILITY_SHADER, WATER_SIM_SHADER,
};

pub(crate) use crate::gpu_layouts::scene::MaterialUniform;

impl MaterialUniform {
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
        uniform.layer[1] = material.windows().map_or(0.0, |rooms| rooms as f32);
        let cutoff = match material.alpha_mode() {
            AlphaMode::Mask { cutoff } => cutoff,
            AlphaMode::Opaque | AlphaMode::Blend => 0.5,
        };
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

pub(crate) fn transform_rows(transform: &UvTransform) -> [[f32; 4]; 2] {
    let (sin, cos) = engine_core::math::sin_cos(transform.rotation);
    let [sx, sy] = transform.scale;
    let [ox, oy] = transform.offset;
    [
        [cos * sx, sin * sy, ox, 0.0],
        [-sin * sx, cos * sy, oy, 0.0],
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub(crate) struct Variant(u32);

impl Variant {
    pub(crate) const BASIC: Self = Self(0);
    const MAPS: u32 = constants::VARIANT_MAPS;
    const NORMAL_MAP: u32 = constants::VARIANT_NORMAL_MAP;
    const ALPHA_MASK: u32 = 4;
    const DOUBLE_SIDED: u32 = constants::VARIANT_DOUBLE_SIDED;
    const EXTENDED: u32 = 16;
    const SKINNED: u32 = 32;
    const TRANSMISSION: u32 = 64;
    const METALLIC_ROUGHNESS_MAP: u32 = constants::VARIANT_METALLIC_ROUGHNESS_MAP;
    const OCCLUSION_MAP: u32 = constants::VARIANT_OCCLUSION_MAP;
    const EMISSIVE_MAP: u32 = constants::VARIANT_EMISSIVE_MAP;
    const PACKED_OCCLUSION: u32 = constants::VARIANT_PACKED_OCCLUSION;
    const UBER: u32 = 2048;
    const WORLD_MAPPING: u32 = constants::VARIANT_WORLD_MAPPING;
    const TOP_LAYER: u32 = constants::VARIANT_TOP_LAYER;
    const GROUNDING: u32 = 16384;
    const FADING: u32 = 32768;
    const FOLIAGE: u32 = constants::VARIANT_FOLIAGE;
    const PREPASS: u32 = 131072;
    const PRETESTED: u32 = 262144;

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
        if material.is_foliage() {
            bits |= Self::FOLIAGE;
        }
        Self(bits)
    }

    pub(crate) fn uber(self) -> Self {
        let state = Self::EXTENDED
            | Self::SKINNED
            | Self::ALPHA_MASK
            | Self::DOUBLE_SIDED
            | Self::TRANSMISSION;
        Self(self.0 & state | Self::UBER)
    }

    pub(crate) fn bits(self) -> u32 {
        self.0
    }

    pub(crate) fn constants(self) -> [(&'static str, f64); 16] {
        let set = |bit: u32| if self.0 & bit != 0 { 1.0 } else { 0.0 };
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
            ("FOLIAGE", set(Self::FOLIAGE)),
            ("PRETESTED", set(Self::PRETESTED)),
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

    pub(crate) fn opaque(self) -> Self {
        if self.alpha_mask() {
            Self(self.0 | Self::PRETESTED)
        } else {
            self
        }
    }

    pub(crate) fn prepass(self) -> Self {
        Self(self.0 & !Self::PRETESTED | Self::PREPASS)
    }

    pub(crate) fn depth_only(self) -> bool {
        self.0 & Self::PREPASS != 0
    }

    pub(crate) fn pretested(self) -> bool {
        self.0 & Self::PRETESTED != 0
    }

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

    pub(crate) fn with_fading(self, fading: bool) -> Self {
        Self(if fading {
            self.0 | Self::FADING
        } else {
            self.0 & !Self::FADING
        })
    }

    pub(crate) fn with_grounding(self, grounding: bool) -> Self {
        Self(if grounding {
            self.0 | Self::GROUNDING
        } else {
            self.0 & !Self::GROUNDING
        })
    }
}

pub(crate) use crate::gpu_layouts::scene::ShadowUniform;

pub(crate) use crate::gpu_layouts::scene::ShadowCullUniform;

pub(crate) use crate::gpu_layouts::scene::{ComposeData, RouteData};

pub(crate) use crate::gpu_layouts::scene::PostUniform;

pub(crate) use crate::gpu_layouts::scene::SkyUniform;

pub(crate) fn mip_levels(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).max(1).leading_zeros()
}

pub(crate) fn mip_chain_bytes(width: u32, height: u32) -> u64 {
    (1..mip_levels(width, height))
        .map(|level| u64::from((width >> level).max(1)) * u64::from((height >> level).max(1)) * 4)
        .sum()
}

pub(crate) use crate::gpu_layouts::scene::ResolveUniform;

pub(crate) const JITTER_SAMPLES: u32 = 8;

fn halton(mut index: u32, base: u32) -> f32 {
    let (mut scale, mut value) = (1.0, 0.0);
    while index > 0 {
        scale /= base as f32;
        value += scale * (index % base) as f32;
        index /= base;
    }
    value
}

pub(crate) fn jitter_offset(frame: u32) -> [f32; 2] {
    let index = frame % JITTER_SAMPLES + 1;
    [halton(index, 2) - 0.5, halton(index, 3) - 0.5]
}

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

pub(crate) const FAR_DEPTH: f32 = 0.0;

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
