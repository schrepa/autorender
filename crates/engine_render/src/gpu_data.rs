//! Explicit schemas for the adjacent shader, separate from gameplay components.

use bytemuck::{Pod, Zeroable};
use engine_core::{
    AlphaMode, Atmosphere, Bloom, Material, MeshVertex, Reflections, RenderInstance, Surface,
    UvTransform, glam::Mat4,
};

/// WGSL `Camera` uniform: three column-major mat4x4<f32> and three
/// vec4<f32>, alignment 16, size 240. `view_projection` is the rasterized
/// projection.
/// Temporally antialiased frames (decision 017) jitter it by `jitter.xy` clip
/// units per w, and their motion vectors use the previous frame's unjittered
/// `previous_view_projection` and `jitter.zw`, two over the viewport size in
/// pixels. Other frames repeat the view-projection and keep the jitter zero.
/// The unjittered `inverse_view_projection` turns pixels into sky rays and
/// `position` (xyz) is the eye, for view-dependent lighting (decision 018);
/// its w is presentation time in seconds, for animated surfaces (decision
/// 019). `options.x` is 1 while a reflection probe captures a cube face
/// (decision 043): its surfaces read shadows from the far maps alone, which
/// hold every caster whatever the main camera sees. The GPU buffer starts at
/// offset zero; only these 240 initialized bytes are copied into it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CameraUniform {
    pub view_projection: [[f32; 4]; 4],
    pub previous_view_projection: [[f32; 4]; 4],
    pub inverse_view_projection: [[f32; 4]; 4],
    pub position: [f32; 4],
    pub jitter: [f32; 4],
    pub options: [f32; 4],
    /// The wind's velocity (x and z), gust strength and the previous
    /// frame's presentation time (decision 058).
    pub wind: [f32; 4],
}

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

    /// A reflection probe's cube face seen from `position` through
    /// `view_projection` (decision 043), at presentation time `seconds`.
    pub(crate) fn probe(view_projection: Mat4, position: glam::Vec3, seconds: f32) -> Self {
        Self {
            options: [1.0, 0.0, 0.0, 0.0],
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
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            view_projection: jittered(view_projection, jitter).to_cols_array_2d(),
            previous_view_projection: previous.to_cols_array_2d(),
            jitter: [
                jitter[0],
                jitter[1],
                2.0 / width.max(1) as f32,
                2.0 / height.max(1) as f32,
            ],
            ..Self::exact(view_projection, position, seconds)
        }
    }
}

/// WGSL `Lighting` uniform of the scene's atmosphere (decision 018) and
/// grade (decision 027): ten vec4<f32>, 160 bytes. Scalars ride in the fourth
/// components.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct LightingUniform {
    /// Normalized direction toward the sun; w: rim strength.
    pub sun: [f32; 4],
    /// w: specular strength.
    pub sun_color: [f32; 4],
    /// w: ambient scale.
    pub zenith: [f32; 4],
    /// w: exposure.
    pub horizon: [f32; 4],
    /// w: saturation.
    pub ground: [f32; 4],
    /// Start, density, height falloff and maximum.
    pub fog: [f32; 4],
    /// Emission scale, the exposed brightness above which light glows
    /// (decision 021), the roughness beyond which surfaces reflect only the
    /// sky, and 1 when screen-space reflections run (decision 025).
    pub effects: [f32; 4],
    /// The contrast curve's weight (see [`contrast_weight`]) and vibrance
    /// (decision 027); zw unused.
    pub grade: [f32; 4],
    /// Luminance-normalized split-toning gains of shadows and highlights; w
    /// unused.
    pub shadow_tint: [f32; 4],
    pub highlight_tint: [f32; 4],
    /// Refraction through glass (decision 038): the bend's strength (0
    /// without refraction) and the blur radius in pixels of fully rough
    /// glass; zw unused.
    pub refraction: [f32; 4],
    /// Aerial perspective (decision 053): start, density at height zero,
    /// height falloff and the sun's glow; the air's color (w unused).
    pub aerial: [f32; 4],
    pub aerial_color: [f32; 4],
    /// Occupancy (decision 059): the map's center (x and z), the inverse of
    /// its side in meters and the strength covers darken by (0 without a
    /// map); then the reference height its heights are above and its texels
    /// a side.
    pub occupancy: [f32; 4],
    pub occupancy_heights: [f32; 4],
}

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
    ) -> Self {
        let atmosphere = atmosphere.sanitized();
        let reflections = reflections.sanitized();
        let refraction = refraction.sanitized();
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
            fog: [
                atmosphere.fog_start,
                atmosphere.fog_density,
                atmosphere.fog_falloff,
                atmosphere.fog_max,
            ],
            effects: [
                atmosphere.emission,
                bloom.sanitized().threshold,
                reflections.max_roughness,
                if reflections.enabled() { 1.0 } else { 0.0 },
            ],
            grade: [
                contrast_weight(atmosphere.contrast),
                atmosphere.vibrance,
                0.0,
                0.0,
            ],
            shadow_tint: unit_luminance(atmosphere.shadow_tint),
            highlight_tint: unit_luminance(atmosphere.highlight_tint),
            refraction: [refraction.strength, refraction.blur, 0.0, 0.0],
            aerial: [
                atmosphere.aerial_start,
                atmosphere.aerial_density,
                atmosphere.aerial_falloff,
                atmosphere.aerial_glow,
            ],
            aerial_color: atmosphere.aerial_color.extend(0.0).to_array(),
            occupancy: [0.0; 4],
            occupancy_heights: [0.0; 4],
        }
    }
}

/// A surface (decision 019) as four unsigned 8-bit fractions: roughness,
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

/// Linear color and opacity (decision 026) as four 16-bit fractions in two
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

/// A slot pose as WGSL `Rows` reads it: core's three aligned affine rows at
/// storage offsets 0/16/32, with a 48-byte stride and the implicit final row
/// [0, 0, 0, 1]. Frames upload `RenderFrame::pose_rows` as they are.
pub(crate) type PoseData = engine_core::PoseRows;

/// Separate per-instance color, opacity and packed surface; unchanged colors
/// need no pose upload. WGSL `SlotColor`: vec2<u32> ([`pack_color`]) and two
/// u32, 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ColorData {
    pub color: [u32; 2],
    pub surface: u32,
    pub flags: u32,
}

impl From<&RenderInstance> for ColorData {
    fn from(instance: &RenderInstance) -> Self {
        Self {
            color: pack_color(instance.color),
            surface: pack_surface(instance.surface),
            // One more than the offset of a skinned slot's joint palette
            // (decision 036); 0 draws rigidly.
            flags: instance.skin.map_or(0, |offset| offset + 1),
        }
    }
}

/// Vertex attributes at offsets 0/12/16 with an explicit 24-byte vertex
/// stride: the position, the [`octahedral`] normal and texture coordinates.
/// The CPU asset representation has no GPU byte-layout contract.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Vertex {
    pub position: [f32; 3],
    pub normal: [i16; 2],
    pub uv: [f32; 2],
}

impl From<&MeshVertex> for Vertex {
    fn from(vertex: &MeshVertex) -> Self {
        Self {
            position: vertex.position,
            normal: octahedral(vertex.normal),
            uv: vertex.uv,
        }
    }
}

/// The second vertex stream of meshes with vertex colors or second texture
/// coordinates (decision 034), 16 bytes: linear RGBA as four 16-bit
/// fractions (white without colors) and the second coordinates (the first
/// without them).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct ExtraVertex {
    pub color: [u16; 4],
    pub uv1: [f32; 2],
}

impl ExtraVertex {
    /// The extra stream of `mesh`, or `None` without colors, second
    /// coordinates or joints: skinned meshes' entry points (decision 036)
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

/// The skin stream of meshes with joints (decision 036), 16 bytes: four
/// joint indices and their weights as 16-bit fractions, which the shaders
/// renormalize.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct SkinVertex {
    pub joints: [u16; 4],
    pub weights: [u16; 4],
}

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

/// Meshes after the shared scene declarations.
pub(crate) const SHADER: &str = concat!(
    include_str!("scene.wgsl"),
    include_str!("meshes/declarations.wgsl"),
    include_str!("meshes/vertex.wgsl"),
    include_str!("meshes/shading.wgsl"),
    include_str!("meshes/forward.wgsl"),
    include_str!("meshes/resolve.wgsl"),
    include_str!("meshes/sky.wgsl"),
    include_str!("meshes/decals.wgsl"),
    include_str!("impostor_frames.wgsl"),
    include_str!("meshes/impostors.wgsl"),
    include_str!("meshes/terrain_vertex.wgsl"),
    include_str!("meshes/terrain.wgsl"),
    include_str!("meshes/cover.wgsl"),
);
/// Terrain normal maps (decision 051).
pub(crate) const TERRAIN_NORMALS_SHADER: &str = include_str!("terrain_normals.wgsl");
/// Terrain horizon map builds (decision 052).
pub(crate) const TERRAIN_HORIZON_SHADER: &str = include_str!("terrain_horizon.wgsl");
/// Terrain sky visibility map builds (decision 055).
pub(crate) const TERRAIN_SKY_SHADER: &str = include_str!("terrain_sky.wgsl");
/// Ground cover's growth (decision 056).
pub(crate) const COVER_GROW_SHADER: &str = include_str!("cover_grow.wgsl");
/// Placed objects' culling (decision 057).
pub(crate) const SCATTER_CULL_SHADER: &str = include_str!("scatter_cull.wgsl");

pub(crate) const SCATTER_SNAP_SHADER: &str = include_str!("scatter_snap.wgsl");

pub(crate) const OCCUPANCY_SHADER: &str = include_str!("occupancy.wgsl");
/// Terrain's depth drawing into shadow maps (decision 051).
pub(crate) const TERRAIN_SHADOW_SHADER: &str = concat!(
    include_str!("meshes/terrain_vertex.wgsl"),
    include_str!("terrain_shadow.wgsl")
);
/// Compute passes of GPU presentation (decision 014).
pub(crate) const PRESENTATION_SHADER: &str = include_str!("presentation.wgsl");
/// The depth pyramid of occlusion culling (decision 040).
pub(crate) const DEPTH_PYRAMID_SHADER: &str = include_str!("depth_pyramid.wgsl");
/// Cluster selection of virtual geometry (decision 044).
pub(crate) const VIRTUAL_SHADER: &str = include_str!("virtual_geometry.wgsl");
/// Tile classification of the visibility buffer (decision 045).
pub(crate) const VISIBILITY_SHADER: &str = include_str!("visibility.wgsl");
/// Temporal antialiasing resolve (decision 017), after the shared effects.
pub(crate) const TEMPORAL_SHADER: &str =
    concat!(include_str!("effects.wgsl"), include_str!("temporal.wgsl"));
/// Mip generation of base color textures.
pub(crate) const MIPMAP_SHADER: &str = include_str!("mipmaps.wgsl");
/// Local light binning into view clusters (decision 023).
pub(crate) const LIGHT_SHADER: &str = include_str!("lights.wgsl");
/// Stateless particles after the shared scene declarations (decision 026).
pub(crate) const PARTICLE_SHADER: &str =
    concat!(include_str!("scene.wgsl"), include_str!("particles.wgsl"));
/// The composite of weighted blended transparency (decision 026).
pub(crate) const TRANSPARENT_SHADER: &str = include_str!("transparent.wgsl");
/// Sun shafts (decision 024) after the scene declarations and effects.
pub(crate) const SHAFTS_SHADER: &str = concat!(
    include_str!("scene.wgsl"),
    include_str!("effects.wgsl"),
    include_str!("shafts.wgsl")
);
/// Shadow caster selection and depth drawing (decision 020).
pub(crate) const SHADOW_SHADER: &str = concat!(
    include_str!("shadows.wgsl"),
    include_str!("impostor_frames.wgsl")
);
/// Ambient occlusion, bloom and their composite (decision 021).
pub(crate) const POST_SHADER: &str =
    concat!(include_str!("effects.wgsl"), include_str!("post.wgsl"));
/// Environment maps (decision 043) after the scene declarations.
pub(crate) const ENVIRONMENT_SHADER: &str =
    concat!(include_str!("scene.wgsl"), include_str!("environment.wgsl"));

/// WGSL `MaterialMaps` uniform of `meshes/declarations.wgsl` (decision 031), 192 bytes:
/// each map's texture coordinate transform, the emissive color and the map
/// parameters of one material.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct MaterialUniform {
    /// Rows of each map's 2x3 transform (base color, metallic-roughness,
    /// normal, occlusion and emissive), w unused.
    pub transforms: [[f32; 4]; 10],
    /// Linear emissive color, w unused.
    pub emissive: [f32; 4],
    /// Normal scale, occlusion strength, alpha cutoff, and the maps reading
    /// the second texture coordinates as bits in map order.
    pub params: [f32; 4],
    /// Transmission (decision 038): its share, index of refraction and
    /// thickness; w the material's variant bits as `f32` bits, which uber
    /// pipelines read (decision 049).
    pub volume: [f32; 4],
}

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
        uniform.emissive = [r, g, b, 0.0];
        let cutoff = match material.alpha_mode() {
            AlphaMode::Mask { cutoff } => cutoff,
            _ => 0.5,
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

/// A material's pipeline variant (decision 031): the `meshes/declarations.wgsl` override
/// constants it sets and whether back faces draw. Materials without maps and
/// plain opaque base color textures use [`Variant::BASIC`], the shading of
/// every material before glTF maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub(crate) struct Variant(u16);

impl Variant {
    pub(crate) const BASIC: Self = Self(0);
    const MAPS: u16 = 1;
    const NORMAL_MAP: u16 = 2;
    const ALPHA_MASK: u16 = 4;
    const DOUBLE_SIDED: u16 = 8;
    // Meshes with a second vertex stream (decision 034): extended entry
    // points rather than a constant.
    const EXTENDED: u16 = 16;
    // Skinned slots' meshes (decision 036): skinned entry points, which
    // read the second stream and a skin stream.
    const SKINNED: u16 = 32;
    // Transmissive materials (decision 038): glass shading in the
    // transparent pass.
    const TRANSMISSION: u16 = 64;
    // Maps a material with maps has: the others are not sampled, as the
    // white texel they would read changes nothing. Occlusion packed in the
    // metallic-roughness texture's red (the same texture, sampler, transform
    // and coordinates) is read from its sample.
    const METALLIC_ROUGHNESS_MAP: u16 = 128;
    const OCCLUSION_MAP: u16 = 256;
    const EMISSIVE_MAP: u16 = 512;
    const PACKED_OCCLUSION: u16 = 1024;
    // Passes drawn rarely (decision 049) read the map bits from the
    // material uniform: their pipelines key on pipeline state alone.
    const UBER: u16 = 2048;

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
            || (mode == AlphaMode::Blend && base.is_some());
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
        Self(bits)
    }

    /// This variant for a pass that reads map presence from the material
    /// uniform (decision 049): only what its pipeline state needs, vertex
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

    /// Fragment override constants.
    pub(crate) fn constants(self) -> [(&'static str, f64); 10] {
        let set = |bit: u16| if self.0 & bit != 0 { 1.0 } else { 0.0 };
        [
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
}

/// Most near shadow cascades (decision 028).
pub(crate) const MAX_CASCADES: usize = engine_core::MAX_SHADOW_CASCADES as usize;

/// WGSL `Shadow` uniform of `scene.wgsl` (decisions 020 and 028): world to
/// each near cascade and to the far shadow maps, and their settings, 400
/// bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct ShadowUniform {
    /// World to each near cascade: x and y in [-1, 1] across it, z in
    /// [0, 1].
    pub near: [[[f32; 4]; 4]; MAX_CASCADES],
    /// World to both far maps.
    pub far: [[f32; 4]; 4],
    /// Shadows on, the kept far map holds casters, the near cascades in use,
    /// and the distance where shadows begin to fade.
    pub state: [f32; 4],
    /// View distance each near cascade reaches; unreachable beyond those in
    /// use.
    pub reach: [f32; 4],
    /// Normal offsets (world units) of lookups in each near cascade.
    pub near_offsets: [f32; 4],
    /// Steepest receiver plane (depth per texel) of each near cascade.
    pub near_limits: [f32; 4],
    /// Distance where shadows are gone, the normal offset (world units) of
    /// far lookups and the far maps' steepest receiver plane; w unused.
    pub offsets: [f32; 4],
    /// Texels a side of the near cascades and the far maps; zw unused.
    pub sizes: [f32; 4],
}

/// WGSL `Cull` uniform of `shadows.wgsl`: 240 bytes. Each draw group's ID
/// region start is in a storage buffer beside it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ShadowCullUniform {
    /// World to light space: the sun's right, up and away-from-sun axes.
    pub light: [[f32; 4]; 3],
    /// Light-space rectangles (minimum x and y, maximum x and y) of each
    /// near cascade, the far maps and the region of the kept map redrawn
    /// this frame.
    pub near: [[f32; 4]; MAX_CASCADES],
    pub far: [f32; 4],
    pub refresh: [f32; 4],
    /// The part of the far maps the moving map selects casters from: where
    /// visible receivers can be.
    pub moving: [f32; 4],
    /// Light-space depth beyond which casters cannot darken receivers of
    /// each near cascade; then (y and z) of the far maps and the refreshed
    /// region, and (w) of the moving map's selection.
    pub near_depths: [f32; 4],
    pub depths: [f32; 4],
    pub candidates: u32,
    pub groups: u32,
    /// Instance IDs per map.
    pub capacity: u32,
    /// `STILLNESS | REFRESH` of the shader.
    pub mode: u32,
    /// Kept-map tiles: light-space x and y of the first tile's corner, and
    /// the tile width and height.
    pub tiles: [f32; 4],
    /// Near cascades drawn this frame.
    pub cascades: u32,
    /// Skinned slots `select_skinned` visits on GPU-presented frames.
    pub skinned: u32,
    pub pad: [u32; 2],
    /// Inward world planes (`xyz . p + w >= 0` inside) bounding where
    /// casters can darken each near cascade's receivers, then the moving
    /// map's: their part of the view swept toward the sun; unused planes
    /// pass everything.
    pub planes: [[[f32; 4]; CASTER_PLANES]; MAX_CASCADES + 1],
}

/// Planes of a map's caster volume ([`ShadowCullUniform::planes`]): the
/// view frustum's sides, near and far planes, and the receivers' first and
/// last view depths, those facing the sun.
pub(crate) const CASTER_PLANES: usize = 8;

/// WGSL `Post` uniform of `effects.wgsl` (decisions 021, 024 and 025): 240
/// bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct PostUniform {
    /// Unjittered clip to world space.
    pub inverse_view_projection: [[f32; 4]; 4],
    /// Unjittered world to clip space, for reflection rays.
    pub view_projection: [[f32; 4]; 4],
    /// The eye; w: depth-buffer pixels per world unit at distance 1.
    pub eye: [f32; 4],
    /// Scene target size in pixels, and its inverse.
    pub size: [f32; 4],
    /// Radius, strength, unused and 1 when ambient occlusion runs.
    pub occlusion: [f32; 4],
    /// Strength and 1 when bloom runs; half-resolution texel size.
    pub bloom: [f32; 4],
    /// Ray distance, strength, 1 when reflections run, and which pixel of
    /// each 2x2 block the half-resolution reflections trace this frame.
    pub reflections: [f32; 4],
    /// Strength, forward scattering, distance and 1 when sun shafts run.
    pub shafts: [f32; 4],
    /// The sun's exposed color; w unused.
    pub sun: [f32; 4],
}

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

/// `Resolve` uniform of `temporal.wgsl`: 16 bytes, WGSL uniform layout.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ResolveUniform {
    /// Size of one target pixel in texture coordinates.
    pub texel: [f32; 2],
    /// Weight of this frame's color against the reprojected history.
    pub current_weight: f32,
    /// Zero when the history must not be read (reset).
    pub history_valid: u32,
}

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

/// `Frame` uniform of `presentation.wgsl`: 320 bytes, WGSL uniform layout.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct PresentationFrame {
    pub alpha: f32,
    pub candidates: u32,
    pub budget: u32,
    pub culling: u32,
    pub groups: u32,
    pub batches: u32,
    /// Node slots per half of the pose buffer; previous poses follow.
    pub node_capacity: u32,
    /// Zero while previous poses are inconsistent (no blending).
    pub blend: u32,
    pub planes: [[f32; 4]; 6],
    pub absolute_view_projection: [[f32; 4]; 4],
    /// Nonzero on temporally antialiased frames with valid previous poses:
    /// visible IDs of moving slots then carry [`MOVED_SLOT`].
    pub motion: u32,
    /// Joints of skinned slots' palettes (decision 036).
    pub joints: u32,
    /// Candidate flags compaction gathers: [`FLAG_VISIBLE`] without
    /// occlusion culling, [`FLAG_PHASE1`] or [`FLAG_PHASE2`] with it
    /// (decision 040).
    pub mask: u32,
    /// Nonzero when occlusion culling splits drawing into two phases.
    pub occlusion: u32,
    /// The projection the frame rasterizes.
    pub view_projection: [[f32; 4]; 4],
    /// Viewport width and height in pixels, and the depth pyramid's levels.
    pub viewport: [f32; 4],
    /// Skinned slots (decision 036) whose palette bounds `skin_bounds_pass`
    /// computes and culls.
    pub skinned: u32,
    /// Nonzero when candidates covering no pixel center are left undrawn
    /// (decision 041).
    pub subpixel: u32,
    /// Candidates per visible-ID plane: each level of detail's IDs start
    /// this many entries after the last's (decision 047).
    pub plane: u32,
    /// The screen error in pixels levels of detail keep within; 0 draws
    /// every instance at full detail (decision 047).
    pub lod_error: f32,
    /// The eye the frame rasterizes from (w unused): levels of detail
    /// follow the distance to it (decision 047).
    pub eye: [f32; 4],
}

/// Candidate flags of `presentation.wgsl` the compaction passes gather.
pub(crate) const FLAG_VISIBLE: u32 = 1;
pub(crate) const FLAG_PHASE1: u32 = 8;
pub(crate) const FLAG_PHASE2: u32 = 16;
/// Reflection probe captures (decision 043): candidates of the cube face
/// being captured.
pub(crate) const FLAG_PROBE: u32 = 128;
/// Frames in a row a slot must display an unchanged pose before reflection
/// probes capture it (decision 043).
pub(crate) const STILL_FRAMES: u32 = 16;

/// Bit of a visible slot ID marking a slot whose displayed pose can differ
/// from the previous frame's (decision 017); slot indices stay below it.
pub(crate) const MOVED_SLOT: u32 = 1 << 31;

/// One candidate slot, its hierarchy node, its draw group and its index in
/// draw-group order (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CandidateData {
    pub slot: u32,
    pub node: u32,
    pub batch: u32,
    pub position: u32,
}

/// One draw group: its candidate range (also its compacted visible-ID region),
/// mesh index count, shadow draw group (decision 020), local bounds and
/// levels of detail (96 bytes). `bounds_min[3]` is 1 for meshes with joints
/// (decision 036), whose joints place them: GPU-presented frames bound their
/// slots by their palettes ([`SkinnedData`]), and they never cast kept
/// shadows.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct BatchData {
    pub start: u32,
    pub count: u32,
    pub index_count: u32,
    pub pad: u32,
    pub bounds_min: [f32; 4],
    pub bounds_max: [f32; 4],
    // Where the batch's mesh lies in the mesh arena: its first index and
    // base vertex, which its draw arguments carry, with its first instance:
    // `start` where devices draw indirectly from nonzero first instances,
    // otherwise 0.
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
    /// What each coarser level's first instance adds per level (decision
    /// 047): the visible-ID plane stride where first instances name
    /// regions, otherwise 0.
    pub first_plane: u32,
    /// Levels of detail 1 and 2 (decision 047): each one's first index and
    /// index count in the mesh arena.
    pub lod_ranges: [u32; 4],
    /// Levels 1 and 2's geometric errors in mesh units, then the level
    /// count (1 to [`LOD_LEVELS`]); w unused.
    pub lod_errors: [f32; 4],
}

/// Levels of detail a draw group has at most (decision 047): the mesh and two
/// simplifications, each with its own plane of visible IDs and draws.
pub(crate) const LOD_LEVELS: usize = 3;

/// A skinned slot of GPU-presented frames (decision 036): its slot, its
/// candidate's draw-group position, its draw group and its mesh's joint count
/// (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SkinnedData {
    pub slot: u32,
    pub position: u32,
    pub batch: u32,
    pub joints: u32,
}

/// Indexed indirect draw arguments written by the GPU (20 bytes).
pub(crate) const DRAW_ARGS_BYTES: u64 = 20;
/// Composed active-parent record of `presentation.wgsl` (112 bytes).
pub(crate) const COMPOSED_BYTES: u64 = 112;

/// Clip-space depth of the near and far planes: depth is reversed (decision
/// 042), so nearer is larger and the depth buffer clears to the far plane.
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
