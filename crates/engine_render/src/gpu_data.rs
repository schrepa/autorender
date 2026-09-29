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
        }
    }
}

/// A surface (decision 019) as four unsigned 8-bit fractions: roughness,
/// metalness, emission over [`Surface::MAX_EMISSION`], and ripples. Public
/// snapshots may carry any values: out-of-range ones clamp and NaN reads as
/// matte.
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

/// Vertex attributes, NOT a uniform/storage struct: columns at offsets
/// 0/16/32/48, packed color and opacity at 64 ([`pack_color`]) and the
/// packed surface at 72, with explicit 80-byte instance stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct InstanceData {
    pub model: [[f32; 4]; 4],
    pub color: [u32; 2],
    pub surface: u32,
    pub flags: u32,
}

impl From<&RenderInstance> for InstanceData {
    fn from(instance: &RenderInstance) -> Self {
        Self {
            model: instance.model.to_cols_array_2d(),
            color: pack_color(instance.color),
            surface: pack_surface(instance.surface),
            flags: 0,
        }
    }
}

/// Three aligned affine rows at storage offsets 0/16/32, with a 48-byte stride.
/// Only engine-owned frames establish the implicit final row [0, 0, 0, 1].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct PoseData {
    pub rows: [[f32; 4]; 3],
}

// Retained frames upload core's `PoseRows` bytes directly as these records.
const _: () = assert!(size_of::<PoseData>() == size_of::<engine_core::PoseRows>());

impl From<Mat4> for PoseData {
    fn from(m: Mat4) -> Self {
        Self {
            rows: [
                [m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x],
                [m.x_axis.y, m.y_axis.y, m.z_axis.y, m.w_axis.y],
                [m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z],
            ],
        }
    }
}

impl From<&RenderInstance> for PoseData {
    fn from(instance: &RenderInstance) -> Self {
        Self::from(instance.model)
    }
}

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
/// (Cigolle et al. 2014). `meshes.wgsl` decodes it with `octahedral_normal`;
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
pub(crate) const SHADER: &str = concat!(include_str!("scene.wgsl"), include_str!("meshes.wgsl"));
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
pub(crate) const SHADOW_SHADER: &str = include_str!("shadows.wgsl");
/// Ambient occlusion, bloom and their composite (decision 021).
pub(crate) const POST_SHADER: &str =
    concat!(include_str!("effects.wgsl"), include_str!("post.wgsl"));
/// Environment maps (decision 043) after the scene declarations.
pub(crate) const ENVIRONMENT_SHADER: &str =
    concat!(include_str!("scene.wgsl"), include_str!("environment.wgsl"));

/// WGSL `MaterialMaps` uniform of `meshes.wgsl` (decision 031), 192 bytes:
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
    /// thickness; w unused.
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
            0.0,
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

/// A material's pipeline variant (decision 031): the `meshes.wgsl` override
/// constants it sets and whether back faces draw. Materials without maps and
/// plain opaque base color textures use [`Variant::BASIC`], the shading of
/// every material before glTF maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub(crate) struct Variant(u8);

impl Variant {
    pub(crate) const BASIC: Self = Self(0);
    const MAPS: u8 = 1;
    const NORMAL_MAP: u8 = 2;
    const ALPHA_MASK: u8 = 4;
    const DOUBLE_SIDED: u8 = 8;
    // Meshes with a second vertex stream (decision 034): extended entry
    // points rather than a constant.
    const EXTENDED: u8 = 16;
    // Skinned slots' meshes (decision 036): skinned entry points, which
    // read the second stream and a skin stream.
    const SKINNED: u8 = 32;
    // Transmissive materials (decision 038): glass shading in the
    // transparent pass.
    const TRANSMISSION: u8 = 64;

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

    /// Fragment override constants.
    pub(crate) fn constants(self) -> [(&'static str, f64); 5] {
        let set = |bit: u8| if self.0 & bit != 0 { 1.0 } else { 0.0 };
        [
            ("MATERIAL_MAPS", set(Self::MAPS)),
            ("NORMAL_MAP", set(Self::NORMAL_MAP)),
            ("ALPHA_MASK", set(Self::ALPHA_MASK)),
            ("DOUBLE_SIDED", set(Self::DOUBLE_SIDED)),
            ("TRANSMISSION", set(Self::TRANSMISSION)),
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
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{AssetLibrary, Mesh, valid_global_matrix};
    use glam::{DMat3, Mat3, Mat4, Quat, Vec3};

    #[test]
    fn resize_suspends_zero_and_preserves_aspect_at_device_limit() {
        assert_eq!(clamped_extent(0, 1080, 8192), None);
        assert_eq!(clamped_extent(1920, 0, 8192), None);
        assert_eq!(clamped_extent(1920, 1080, 8192), Some((1920, 1080)));
        assert_eq!(clamped_extent(16_384, 8192, 8192), Some((8192, 4096)));
        assert_eq!(clamped_extent(u32::MAX, 1, 8192), Some((8192, 1)));
    }

    #[test]
    fn gpu_records_have_documented_offsets_and_column_order() {
        assert_eq!(size_of::<CameraUniform>(), 240);
        assert_eq!(
            std::mem::offset_of!(CameraUniform, previous_view_projection),
            64
        );
        assert_eq!(
            std::mem::offset_of!(CameraUniform, inverse_view_projection),
            128
        );
        assert_eq!(std::mem::offset_of!(CameraUniform, position), 192);
        assert_eq!(std::mem::offset_of!(CameraUniform, jitter), 208);
        assert_eq!(size_of::<LightingUniform>(), 176);
        assert_eq!(size_of::<InstanceData>(), 80);
        assert_eq!(std::mem::offset_of!(InstanceData, color), 64);
        assert_eq!(std::mem::offset_of!(InstanceData, surface), 72);
        assert_eq!(std::mem::offset_of!(ColorData, surface), 8);
        assert_eq!(size_of::<PoseData>(), 48);
        assert_eq!(size_of::<ColorData>(), 16);
        assert_eq!(size_of::<Vertex>(), 24);
        assert_eq!(std::mem::offset_of!(Vertex, normal), 12);
        assert_eq!(std::mem::offset_of!(Vertex, uv), 16);
        let mut assets = AssetLibrary::default();
        let instance = RenderInstance {
            model: Mat4::from_translation(Vec3::new(2.0, 3.0, 4.0)),
            color: [0.1, 0.2, 0.3, 1.0],
            surface: engine_core::Surface {
                roughness: 0.2,
                metalness: 1.0,
                emission: 4.0,
                ripples: 0.0,
            },
            mesh: assets.insert_mesh(Mesh::cube()).unwrap(),
            material: None,
            skin: None,
        };
        let upload = InstanceData::from(&instance);
        assert_eq!(upload.model[3], [2.0, 3.0, 4.0, 1.0]);
        assert_eq!(upload.color, pack_color([0.1, 0.2, 0.3, 1.0]));
        // Four 16-bit fractions: red and green, then blue and opacity.
        assert_eq!(
            pack_color([0.0, 1.0, 0.5, 0.25]),
            [0xffff << 16, 32768 | 16384 << 16]
        );
        assert_eq!(
            pack_color([f32::NAN, 2.0, -1.0, f32::NAN]),
            [0xffff << 16, 0xffff << 16]
        );
        assert_eq!(upload.surface, 51 | 255 << 8 | 64 << 16);
        assert_eq!(ColorData::from(&instance).surface, upload.surface);
        let affine = Mat4::from_scale_rotation_translation(
            Vec3::new(-2.0, 0.5, 3.0),
            Quat::from_rotation_y(0.7),
            Vec3::new(2.0, 3.0, 4.0),
        ) * Mat4::from_quat(Quat::from_rotation_z(0.4));
        let pose = PoseData::from(&RenderInstance {
            model: affine,
            ..instance
        });
        // Reconstruct exactly as vs_affine does. This retains shear/reflection.
        let reconstructed = Mat4::from_cols(
            pose.rows[0].into(),
            pose.rows[1].into(),
            pose.rows[2].into(),
            glam::Vec4::W,
        )
        .transpose();
        assert_eq!(reconstructed, affine);
        // Public snapshots retain all four rows even for a projective matrix.
        let projective = Mat4::perspective_rh(0.8, 1.5, 0.1, 100.0);
        let full = InstanceData::from(&RenderInstance {
            model: projective,
            ..instance
        });
        assert_eq!(full.model, projective.to_cols_array_2d());
        let vertex = Vertex::from(&MeshVertex {
            position: [1.0, 2.0, 3.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.25, 0.75],
        });
        assert_eq!(vertex.position, [1.0, 2.0, 3.0]);
        assert_eq!(vertex.normal, [0, 32767]);
        assert_eq!(vertex.uv, [0.25, 0.75]);
    }

    /// `octahedral_normal` of `meshes.wgsl`, on Snorm16x2 inputs.
    fn octahedral_normal(encoded: [i16; 2]) -> Vec3 {
        let [x, y] = encoded.map(|value| engine_core::math::max(f32::from(value) / 32767.0, -1.0));
        let z = 1.0 - x.abs() - y.abs();
        let fold = engine_core::math::max(-z, 0.0);
        let shift = |value: f32| if value >= 0.0 { fold } else { -fold };
        Vec3::new(x - shift(x), y - shift(y), z)
    }

    #[test]
    fn octahedral_normals_decode_every_direction_within_a_hundredth_of_a_degree() {
        let mut state = 0x9e37_79b9_u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            f32::from((state >> 8) as u16) / 32767.5 - 1.0
        };
        let axes = [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z];
        let directions = axes
            .into_iter()
            .chain([Vec3::new(1.0, -1.0, -1.0), Vec3::new(-0.3, 0.0, -0.95)])
            .chain((0..20_000).map(|_| Vec3::new(next(), next(), next())))
            .filter(|direction| direction.length() > 1.0e-3);
        for direction in directions {
            let direction = direction.normalize();
            let decoded = octahedral_normal(octahedral(direction.to_array())).normalize();
            // The sine of the angle between them, precise for small angles.
            assert!(decoded.dot(direction) > 0.0, "{direction} -> {decoded}");
            assert!(
                decoded.cross(direction).length() < 0.01_f32.to_radians(),
                "{direction} -> {decoded}"
            );
        }
    }

    #[test]
    fn shader_cofactors_preserve_normals_under_nonuniform_positive_scale() {
        let model = Mat3::from_quat(Quat::from_rotation_y(0.7))
            * Mat3::from_diagonal(Vec3::new(0.25, 2.0, 5.0));
        let normal = Vec3::new(1.0, 1.0, 1.0).normalize();
        let shader_normal = shader_normal(model, normal);
        let expected = (model.inverse().transpose() * normal).normalize();
        assert!(shader_normal.abs_diff_eq(expected, 1e-6));
        assert!((model * normal).normalize().distance(expected) > 0.5);
    }

    fn shader_normal(model: Mat3, normal: Vec3) -> Vec3 {
        let scale = model
            .x_axis
            .abs()
            .max(model.y_axis.abs())
            .max(model.z_axis.abs())
            .max_element();
        let model = Mat3::from_cols(
            model.x_axis / scale,
            model.y_axis / scale,
            model.z_axis / scale,
        );
        let cofactors = Mat3::from_cols(
            model.y_axis.cross(model.z_axis),
            model.z_axis.cross(model.x_axis),
            model.x_axis.cross(model.y_axis),
        );
        let sign = model.x_axis.dot(cofactors.x_axis).signum();
        // Cofactors of the rescaled basis stay within [-2, 2]; normalizing
        // by the largest component handles their magnitude.
        let transformed = cofactors * normal;
        (transformed / transformed.abs().max_element()).normalize() * sign
    }

    fn reference_normal(model: Mat3, normal: Vec3) -> Vec3 {
        let matrix = DMat3::from_cols_array(&model.to_cols_array().map(f64::from));
        (matrix.inverse().transpose() * normal.as_dvec3())
            .normalize()
            .as_vec3()
    }

    #[test]
    fn admitted_extreme_world_scales_keep_finite_unit_shader_normals() {
        for scale in [1.0e-12, 1.0e12] {
            let model = Mat3::from_quat(Quat::from_rotation_y(0.7))
                * Mat3::from_diagonal(Vec3::splat(scale));
            assert!(valid_global_matrix(Mat4::from_mat3(model)));
            for normal in [
                Vec3::X,
                Vec3::Y,
                Vec3::Z,
                Vec3::new(1.0, -1.0, 1.0).normalize(),
            ] {
                let result = shader_normal(model, normal);
                assert!(result.is_finite());
                assert!((result.length() - 1.0).abs() < 1.0e-6);
                assert!(result.abs_diff_eq(reference_normal(model, normal), 1.0e-6));
            }
            let unscaled = model.y_axis.cross(model.z_axis).normalize();
            assert!(!unscaled.is_finite() || unscaled.length() < 0.1);
        }
    }

    #[test]
    fn conditioned_shear_and_reflection_match_double_precision_inverse_transpose() {
        let model = Mat3::from_quat(Quat::from_rotation_y(0.7))
            * Mat3::from_diagonal(Vec3::new(-2.0, 0.003, 3.0))
            * Mat3::from_quat(Quat::from_rotation_z(0.4));
        assert!(valid_global_matrix(Mat4::from_mat3(model)));
        for normal in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(0.5, -1.0, 0.7).normalize(),
        ] {
            let result = shader_normal(model, normal);
            assert!(result.is_finite());
            assert!((result.length() - 1.0).abs() < 1.0e-6);
            assert!(result.abs_diff_eq(reference_normal(model, normal), 1.0e-4));
        }
    }

    #[test]
    fn unsafe_normal_conditioning_is_rejected_before_gpu_submission() {
        for model in [
            Mat3::from_diagonal(Vec3::new(0.0001, 10_000.0, 1.0)),
            Mat3::from_cols(Vec3::X, Vec3::new(1.0, 1.0e-8, 0.0), Vec3::Z),
        ] {
            assert!(model.inverse().is_finite());
            assert!(!valid_global_matrix(Mat4::from_mat3(model)));
        }
    }

    #[test]
    fn lighting_carries_the_sanitized_atmosphere_and_bloom_threshold() {
        let night = engine_core::Atmosphere::night();
        let bloom = engine_core::Bloom {
            intensity: 1.0,
            threshold: 3.0,
        };
        let off = engine_core::Reflections {
            intensity: 0.0,
            ..engine_core::Reflections::default()
        };
        let glassless = engine_core::Refraction {
            strength: 0.0,
            ..engine_core::Refraction::default()
        };
        let lighting = LightingUniform::new(&night, &bloom, &off, &glassless);
        assert_eq!(lighting.refraction, [0.0, 12.0, 0.0, 0.0]);
        assert_eq!(
            LightingUniform::new(&night, &bloom, &off, &engine_core::Refraction::default())
                .refraction,
            [1.0, 12.0, 0.0, 0.0]
        );
        assert_eq!(lighting.sun_color[3], night.specular);
        assert_eq!(lighting.horizon[3], night.exposure);
        assert_eq!(
            lighting.effects,
            [night.emission, 3.0, off.max_roughness, 0.0]
        );
        let on = engine_core::Reflections {
            max_roughness: 0.3,
            ..engine_core::Reflections::default()
        };
        assert_eq!(
            LightingUniform::new(&night, &bloom, &on, &glassless).effects,
            [night.emission, 3.0, 0.3, 1.0]
        );
        let sun = glam::Vec3::from_slice(&lighting.sun[..3]);
        assert!((sun.length() - 1.0).abs() < 1.0e-6);
        assert_eq!(
            lighting.grade,
            [contrast_weight(night.contrast), night.vibrance, 0.0, 0.0]
        );
        // The contrast curve's slope at mid grey is the contrast up to 1.5;
        // it keeps black and white and never falls.
        let curve = |weight: f32, s: f32| s + weight * (s * s * (3.0 - 2.0 * s) - s);
        for contrast in [0.0, 0.5, 1.0, 1.2, 1.5, 9.0] {
            let weight = contrast_weight(contrast);
            let slope = (curve(weight, 0.501) - curve(weight, 0.499)) / 0.002;
            assert!(
                (slope - engine_core::math::min(contrast, 1.5)).abs() < 1.0e-3,
                "{contrast}: {slope}"
            );
            assert_eq!((curve(weight, 0.0), curve(weight, 1.0)), (0.0, 1.0));
            let mut previous = 0.0;
            for step in 1..=100 {
                let value = curve(weight, step as f32 / 100.0);
                assert!(value >= previous - 1.0e-6, "{contrast}");
                previous = value;
            }
        }
        // Split-toning gains keep the tint's hue at unit luminance, so they
        // shift color without darkening; a black tint changes nothing.
        let weights = glam::Vec3::new(0.2126, 0.7152, 0.0722);
        for (tint, packed) in [
            (night.shadow_tint, lighting.shadow_tint),
            (night.highlight_tint, lighting.highlight_tint),
        ] {
            let gain = glam::Vec3::from_slice(&packed[..3]);
            assert!((gain.dot(weights) - 1.0).abs() < 1.0e-6);
            assert!((gain * tint.x - tint * gain.x).abs().max_element() < 1.0e-6);
        }
        let black = engine_core::Atmosphere {
            shadow_tint: glam::Vec3::ZERO,
            ..night
        };
        assert_eq!(
            LightingUniform::new(&black, &bloom, &off, &glassless).shadow_tint,
            [1.0, 1.0, 1.0, 0.0]
        );
        let broken = engine_core::Atmosphere {
            exposure: f32::NAN,
            ..night
        };
        assert_eq!(
            LightingUniform::new(&broken, &bloom, &off, &glassless),
            LightingUniform::new(
                &engine_core::Atmosphere::default(),
                &bloom,
                &off,
                &glassless
            )
        );
    }

    #[test]
    fn surfaces_pack_as_clamped_fractions_and_nan_reads_matte() {
        use engine_core::Surface;
        assert_eq!(pack_surface(Surface::MATTE), 204);
        let wild = Surface {
            roughness: f32::NAN,
            metalness: -3.0,
            emission: 1000.0,
            ripples: 0.5,
        };
        // NaN roughness reads as matte (0.8), metalness clamps to 0,
        // emission to its maximum, and ripples round to the nearest step.
        assert_eq!(pack_surface(wild), 204 | 255 << 16 | 128 << 24);
    }

    #[test]
    fn material_maps_follow_gltf_texture_transforms() {
        let mut library = AssetLibrary::default();
        let texture = library
            .insert_texture(engine_core::Texture::new(1, 1, vec![255; 4]).unwrap())
            .unwrap();
        let apply = |rows: [[f32; 4]; 2], uv: [f32; 2]| {
            rows.map(|row| row[0] * uv[0] + row[1] * uv[1] + row[2])
        };
        // Scaled, rotated, then offset, as Khronos' TextureTransformTest
        // expects: a quarter turn takes u to -v.
        let transform = UvTransform {
            offset: [0.5, 0.25],
            rotation: std::f32::consts::FRAC_PI_2,
            scale: [2.0, 3.0],
        };
        let [u, v] = apply(transform_rows(&transform), [1.0, 0.0]);
        assert!(
            (u - 0.5).abs() < 1.0e-6 && (v - -1.75).abs() < 1.0e-6,
            "{u} {v}"
        );
        let [u, v] = apply(transform_rows(&transform), [0.0, 1.0]);
        assert!(
            (u - 3.5).abs() < 1.0e-6 && (v - 0.25).abs() < 1.0e-6,
            "{u} {v}"
        );
        // Materials without maps keep identity transforms.
        let plain = Material::new([1.0; 4]).unwrap();
        assert_eq!(MaterialUniform::new(&plain), MaterialUniform::DEFAULT);
        let map = engine_core::TextureRef::new(texture).with_transform(transform);
        let material = plain
            .with_normal_texture(map, 0.5)
            .unwrap()
            .with_emissive([2.0, 1.0, 0.0])
            .unwrap()
            .with_alpha_mode(AlphaMode::Mask { cutoff: 0.3 })
            .unwrap();
        let uniform = MaterialUniform::new(&material);
        assert_eq!(uniform.transforms[4..6], transform_rows(&transform));
        assert_eq!(
            uniform.transforms[..4],
            MaterialUniform::DEFAULT.transforms[..4]
        );
        assert_eq!(uniform.emissive, [2.0, 1.0, 0.0, 0.0]);
        assert_eq!(uniform.params, [0.5, 1.0, 0.3, 0.0]);
    }

    #[test]
    fn extended_meshes_stream_colors_and_second_coordinates() {
        let vertices: Vec<MeshVertex> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
            .map(|uv| MeshVertex {
                position: [uv[0], uv[1], 0.0],
                normal: [0.0, 0.0, 1.0],
                uv,
            })
            .to_vec();
        let mesh = Mesh::new(vertices, vec![0, 1, 2]).unwrap();
        assert!(ExtraVertex::stream(&mesh).is_none());
        // Without second coordinates the stream repeats the first.
        let colored = mesh.with_colors(vec![[1.0, 0.5, 0.0, 1.0]; 3]).unwrap();
        let stream = ExtraVertex::stream(&colored).unwrap();
        assert_eq!(stream[1].color, [65535, 32768, 0, 65535]);
        assert_eq!(stream[1].uv1, [1.0, 0.0]);
        let second = Mesh::new(colored.vertices().to_vec(), vec![0, 1, 2])
            .unwrap()
            .with_uv1(vec![[0.25, 0.75]; 3])
            .unwrap();
        let stream = ExtraVertex::stream(&second).unwrap();
        // Without colors, white.
        assert_eq!(stream[2].color, [u16::MAX; 4]);
        assert_eq!(stream[2].uv1, [0.25, 0.75]);
        assert_eq!(size_of::<ExtraVertex>(), 16);
        // Extended meshes pick extended entry points on any variant.
        let variant = Variant::BASIC.with_extended(true);
        assert!(variant.extended() && variant != Variant::BASIC);
        assert_eq!(variant.constants(), Variant::BASIC.constants());
        assert_eq!(variant.with_extended(false), Variant::BASIC);
    }

    #[test]
    fn skinned_meshes_stream_joints_and_weights_and_slots_carry_palette_offsets() {
        let vertices: Vec<MeshVertex> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
            .map(|uv| MeshVertex {
                position: [uv[0], uv[1], 0.0],
                normal: [0.0, 0.0, 1.0],
                uv,
            })
            .to_vec();
        let rigid = Mesh::new(vertices.clone(), vec![0, 1, 2]).unwrap();
        assert!(SkinVertex::stream(&rigid).is_none());
        let skinned = Mesh::new(vertices, vec![0, 1, 2])
            .unwrap()
            .with_skin(
                vec![[0, 1, 0, 0], [2, 0, 0, 0], [1, 2, 3, 0]],
                vec![[3.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.5; 4]],
            )
            .unwrap();
        // Weights arrive normalized; zero-weight joints read as joint 0.
        let stream = SkinVertex::stream(&skinned).unwrap();
        assert_eq!(stream[0].joints, [0, 1, 0, 0]);
        assert_eq!(stream[0].weights, [49151, 16384, 0, 0]);
        assert_eq!(stream[1].joints, [2, 0, 0, 0]);
        assert_eq!(stream[2].weights, [16384; 4]);
        assert_eq!(size_of::<SkinVertex>(), 16);
        // Skinned meshes' entry points read the second stream too.
        assert!(ExtraVertex::stream(&skinned).is_some());
        let variant = Variant::BASIC.with_skinned(true);
        assert!(variant.skinned() && variant.extended());
        assert_eq!(variant.constants(), Variant::BASIC.constants());
        assert_eq!(
            variant.with_skinned(false),
            Variant::BASIC.with_extended(true)
        );
        // Slot flags are one more than the palette offset; 0 is rigid.
        let mut library = AssetLibrary::default();
        let instance = RenderInstance {
            mesh: library.insert_mesh(skinned).unwrap(),
            material: None,
            model: Mat4::IDENTITY,
            color: [1.0; 4],
            surface: engine_core::Surface::MATTE,
            skin: Some(4),
        };
        assert_eq!(ColorData::from(&instance).flags, 5);
        let rigid = RenderInstance {
            skin: None,
            ..instance
        };
        assert_eq!(ColorData::from(&rigid).flags, 0);
    }

    #[test]
    fn material_variants_keep_plain_and_base_color_materials_basic() {
        let mut library = AssetLibrary::default();
        let texture = library
            .insert_texture(engine_core::Texture::new(1, 1, vec![255; 4]).unwrap())
            .unwrap();
        let map = engine_core::TextureRef::new(texture);
        let plain = Material::new([1.0; 4]).unwrap();
        let constants = |variant: Variant| variant.constants().map(|(_, value)| value);
        assert_eq!(Variant::of(&plain), Variant::BASIC);
        // Base color textures, even repeating, shade as before maps.
        assert_eq!(
            Variant::of(&Material::textured([1.0; 4], texture).unwrap()),
            Variant::BASIC
        );
        assert_eq!(
            Variant::of(&plain.with_base_color_texture(map).unwrap()),
            Variant::BASIC
        );
        let transformed = map.with_transform(UvTransform::repeat([4.0, 4.0]));
        assert_eq!(
            constants(Variant::of(
                &plain.with_base_color_texture(transformed).unwrap()
            )),
            [1.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            constants(Variant::of(&plain.with_normal_texture(map, 1.0).unwrap())),
            [1.0, 1.0, 0.0, 0.0, 0.0]
        );
        let masked = plain
            .with_alpha_mode(AlphaMode::Mask { cutoff: 0.5 })
            .unwrap()
            .with_double_sided(true);
        assert_eq!(constants(Variant::of(&masked)), [1.0, 0.0, 1.0, 1.0, 0.0]);
        assert!(Variant::of(&masked).double_sided());
        // The visibility buffer (decision 045) leaves masked materials to
        // forward drawing.
        assert!(Variant::of(&masked).alpha_mask() && !Variant::of(&plain).alpha_mask());
        // Blending reads texture alpha as coverage.
        let glass = plain
            .with_opacity(0.5)
            .unwrap()
            .with_base_color_texture(map)
            .unwrap();
        assert_eq!(constants(Variant::of(&glass)), [1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            Variant::of(&plain.with_opacity(0.5).unwrap()),
            Variant::BASIC
        );
        // Transmissive materials (decision 038) bind their own parameters
        // and shade as glass in the transparent pass.
        let clear = plain
            .with_transmission(engine_core::Transmission {
                factor: 0.9,
                ior: 1.33,
                thickness: 0.25,
            })
            .unwrap();
        assert!(clear.is_transparent() && clear.has_maps());
        assert_eq!(constants(Variant::of(&clear)), [0.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(Variant::of(&clear).transmission());
        assert_eq!(MaterialUniform::new(&clear).volume, [0.9, 1.33, 0.25, 0.0]);
        assert!(
            plain
                .with_transmission(engine_core::Transmission {
                    ior: 0.9,
                    ..engine_core::Transmission::GLASS
                })
                .is_err()
        );
    }

    #[test]
    fn effect_uniforms_match_wgsl() {
        let size = |source: &str, name: &str| {
            let module = naga::front::wgsl::parse_str(source).unwrap();
            let (_, ty) = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(name))
                .unwrap();
            ty.inner.size(module.to_ctx()) as usize
        };
        assert_eq!(size(SHADER, "Shadow"), size_of::<ShadowUniform>());
        assert_eq!(size_of::<ShadowUniform>(), 400);
        assert_eq!(size(SHADOW_SHADER, "Cull"), size_of::<ShadowCullUniform>());
        assert_eq!(size_of::<ShadowCullUniform>(), 240);
        // Members sit where Rust puts them (decision 028's arrays).
        let offset = |source: &str, name: &str, member: &str| {
            let module = naga::front::wgsl::parse_str(source).unwrap();
            let (_, ty) = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(name))
                .unwrap();
            let naga::TypeInner::Struct { members, .. } = &ty.inner else {
                panic!("{name} is a struct");
            };
            members
                .iter()
                .find(|field| field.name.as_deref() == Some(member))
                .unwrap()
                .offset as usize
        };
        assert_eq!(
            offset(SHADER, "Shadow", "offsets"),
            std::mem::offset_of!(ShadowUniform, offsets)
        );
        assert_eq!(
            offset(SHADOW_SHADER, "Cull", "cascades"),
            std::mem::offset_of!(ShadowCullUniform, cascades)
        );

        assert_eq!(size(SHADOW_SHADER, "Batch"), size_of::<BatchData>());
        assert_eq!(size(SHADOW_SHADER, "Candidate"), size_of::<CandidateData>());
        assert_eq!(size(SHADOW_SHADER, "DrawArgs") as u64, DRAW_ARGS_BYTES);
        assert_eq!(size(POST_SHADER, "Post"), size_of::<PostUniform>());
        assert_eq!(size(TEMPORAL_SHADER, "Post"), size_of::<PostUniform>());
        assert_eq!(size(SHAFTS_SHADER, "Post"), size_of::<PostUniform>());
        assert_eq!(size_of::<PostUniform>(), 240);
        assert_eq!(size(SHADER, "SlotColor"), size_of::<ColorData>());
        // Material maps (decision 031).
        assert_eq!(size(SHADER, "MaterialMaps"), size_of::<MaterialUniform>());
        assert_eq!(size_of::<MaterialUniform>(), 208);
        assert_eq!(size(SHADER, "Lighting"), size_of::<LightingUniform>());
        // Local lights (decision 023): binning and shading agree on layouts.
        let clusters = size_of::<crate::lights::ClusterUniform>();
        assert_eq!(size(LIGHT_SHADER, "Clusters"), clusters);
        assert_eq!(size(SHADER, "Clusters"), clusters);
        assert_eq!(
            size(LIGHT_SHADER, "Input"),
            size_of::<crate::lights::LightInput>()
        );
        assert_eq!(
            size(LIGHT_SHADER, "Light") as u64,
            crate::lights::LIGHT_BYTES
        );
        assert_eq!(size(SHADER, "Light") as u64, crate::lights::LIGHT_BYTES);
        // Decals (decision 039).
        assert_eq!(
            size(SHADER, "DecalRecord"),
            size_of::<crate::decals::DecalData>()
        );
        assert_eq!(size_of::<crate::decals::DecalData>(), 96);
        // Impostors (decision 048).
        assert_eq!(
            size(SHADER, "ImpostorRecord"),
            size_of::<crate::impostors::ImpostorRecord>()
        );
        assert_eq!(size_of::<crate::impostors::ImpostorRecord>(), 48);
        assert_eq!(
            size(SHADER, "ImpostorBake"),
            size_of::<crate::impostors::ImpostorBakeUniform>()
        );
        // Particles (decision 026).
        assert_eq!(
            size(PARTICLE_SHADER, "Emitter"),
            size_of::<crate::particles::EmitterData>()
        );
        assert_eq!(
            size(PARTICLE_SHADER, "Particles"),
            size_of::<crate::particles::ParticleUniform>()
        );
        // Selection reads the presentation flag culling writes for valid poses.
        let bits = |source: &str, name: &str| {
            let module = naga::front::wgsl::parse_str(source).unwrap();
            let (_, constant) = module
                .constants
                .iter()
                .find(|(_, constant)| constant.name.as_deref() == Some(name))
                .unwrap();
            match module.global_expressions[constant.init] {
                naga::Expression::Literal(naga::Literal::U32(value)) => value,
                ref other => panic!("{name} is {other:?}"),
            }
        };
        assert_eq!(
            bits(SHADOW_SHADER, "VALID"),
            bits(PRESENTATION_SHADER, "VALID")
        );
        // Compaction masks the renderer writes into the frame uniform.
        assert_eq!(bits(PRESENTATION_SHADER, "VISIBLE"), FLAG_VISIBLE);
        assert_eq!(bits(PRESENTATION_SHADER, "PHASE1"), FLAG_PHASE1);
        assert_eq!(bits(PRESENTATION_SHADER, "PHASE2"), FLAG_PHASE2);
        assert_eq!(bits(PRESENTATION_SHADER, "PROBE"), FLAG_PROBE);
        assert_eq!(bits(PRESENTATION_SHADER, "STILL_FRAMES"), STILL_FRAMES);
        assert_eq!(bits(LIGHT_SHADER, "STILL_FRAMES"), STILL_FRAMES);
        // The kept map's dirty tiles the selection marks and the pass reads.
        assert_eq!(bits(SHADOW_SHADER, "TILES"), crate::shadow_fit::TILES);
        assert_eq!(
            bits(SHADOW_SHADER, "TILE_WORDS") as usize,
            crate::shadow_fit::TILE_WORDS
        );
    }

    #[test]
    fn shader_is_valid_wgsl_with_webgpu_core_capabilities() {
        for source in [
            SHADER,
            PRESENTATION_SHADER,
            TEMPORAL_SHADER,
            MIPMAP_SHADER,
            SHADOW_SHADER,
            POST_SHADER,
            LIGHT_SHADER,
            PARTICLE_SHADER,
            TRANSPARENT_SHADER,
            SHAFTS_SHADER,
            DEPTH_PYRAMID_SHADER,
            ENVIRONMENT_SHADER,
            VIRTUAL_SHADER,
            VISIBILITY_SHADER,
        ] {
            let module = naga::front::wgsl::parse_str(source).expect("WGSL syntax");
            // Cube arrays (the environment maps, decision 043) are core
            // WebGPU; only its compatibility mode lacks them.
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::CUBE_ARRAY_TEXTURES,
            )
            .validate(&module)
            .expect("WGSL semantics");
        }
    }

    #[test]
    fn shader_float_literals_are_finite_f32_values() {
        // Browsers reject decimal literals beyond f32::MAX that naga rounds.
        for source in [
            SHADER,
            PRESENTATION_SHADER,
            TEMPORAL_SHADER,
            MIPMAP_SHADER,
            SHADOW_SHADER,
            POST_SHADER,
            LIGHT_SHADER,
            PARTICLE_SHADER,
            TRANSPARENT_SHADER,
            SHAFTS_SHADER,
            ENVIRONMENT_SHADER,
        ] {
            let bytes = source.as_bytes();
            let mut index = 0;
            while index < bytes.len() {
                let boundary = index == 0
                    || !(bytes[index - 1].is_ascii_alphanumeric() || bytes[index - 1] == b'_');
                if !(bytes[index].is_ascii_digit() && boundary) {
                    index += 1;
                    continue;
                }
                let start = index;
                while index < bytes.len()
                    && (bytes[index].is_ascii_alphanumeric()
                        || bytes[index] == b'.'
                        || (matches!(bytes[index], b'+' | b'-')
                            && matches!(bytes[index - 1], b'e' | b'E')))
                {
                    index += 1;
                }
                let token = &source[start..index];
                if token.starts_with("0x") || token.starts_with("0X") {
                    continue;
                }
                if let Ok(value) = token.trim_end_matches(['f', 'h', 'i', 'u']).parse::<f64>() {
                    assert!(value.abs() <= f64::from(f32::MAX), "{token}");
                }
            }
        }
    }

    #[test]
    fn presentation_bounds_match_cpu_culling() {
        let module = naga::front::wgsl::parse_str(PRESENTATION_SHADER).unwrap();
        let constant = |name: &str| {
            let (_, constant) = module
                .constants
                .iter()
                .find(|(_, constant)| constant.name.as_deref() == Some(name))
                .unwrap();
            match module.global_expressions[constant.init] {
                naga::Expression::Literal(naga::Literal::F32(value)) => value,
                ref other => panic!("{name} is {other:?}"),
            }
        };
        // engine_core::culling's support-test bounds, bit for bit.
        assert_eq!(constant("ROUNDING_MARGIN"), 32.0 * f32::EPSILON);
        assert_eq!(constant("MAX_COEFFICIENT"), 1.0e15);
        assert_eq!(constant("MAX_PRODUCT_BOUND"), f32::MAX / 8.0);
        assert_eq!(constant("F32_MAX"), f32::MAX);
        // hierarchy/validation.rs's common admission path. Its proof, and so
        // this GPU gate, requires a normal-conditioning limit of at least 6200
        // (checked at compile time below).
        assert_eq!(constant("MIN_LINEAR_SCALE"), 1.0 / 1024.0);
        assert_eq!(constant("COEFFICIENT_LIMIT"), 1_048_576.0);
        assert_eq!(constant("COMMON_DETERMINANT_MARGIN"), 1.0 / 512.0);
    }

    const _: () = assert!(engine_core::MAX_NORMAL_INVERSE_NORM >= 6200.0);

    #[test]
    fn levels_of_detail_match_wgsl() {
        let constant = |source: &str, name: &str| {
            let module = naga::front::wgsl::parse_str(source).unwrap();
            let (_, constant) = module
                .constants
                .iter()
                .find(|(_, constant)| constant.name.as_deref() == Some(name))
                .unwrap();
            match module.global_expressions[constant.init] {
                naga::Expression::Literal(naga::Literal::U32(value)) => value,
                ref other => panic!("{name} is {other:?}"),
            }
        };
        assert_eq!(
            constant(PRESENTATION_SHADER, "LOD_LEVELS") as usize,
            LOD_LEVELS
        );
        // Impostors (decision 048) are the level after the mesh's levels.
        assert_eq!(
            constant(PRESENTATION_SHADER, "IMPOSTOR") as usize,
            LOD_LEVELS
        );
        assert_eq!(constant(PRESENTATION_SHADER, "IMPOSTOR_VERTICES"), 6);
        use crate::impostors::{IMPOSTOR_FRAMES, IMPOSTOR_LEVELS, IMPOSTOR_TEXELS};
        assert_eq!(constant(SHADER, "IMPOSTOR_FRAMES"), IMPOSTOR_FRAMES);
        assert_eq!(constant(SHADER, "IMPOSTOR_TEXELS"), IMPOSTOR_TEXELS);
        assert_eq!(constant(SHADER, "IMPOSTOR_LEVELS"), IMPOSTOR_LEVELS);
        // Frames keep two texels across at the coarsest level, and fit the
        // six bits a card packs each frame in.
        const {
            assert!(IMPOSTOR_TEXELS >> (IMPOSTOR_LEVELS - 1) >= 2);
            assert!(IMPOSTOR_FRAMES * IMPOSTOR_FRAMES <= 64);
        }
    }

    #[test]
    fn presentation_layouts_match_wgsl() {
        assert_eq!(size_of::<PresentationFrame>(), 320);
        assert_eq!(std::mem::offset_of!(PresentationFrame, planes), 32);
        assert_eq!(
            std::mem::offset_of!(PresentationFrame, absolute_view_projection),
            128
        );
        assert_eq!(std::mem::offset_of!(PresentationFrame, motion), 192);
        assert_eq!(std::mem::offset_of!(PresentationFrame, mask), 200);
        assert_eq!(
            std::mem::offset_of!(PresentationFrame, view_projection),
            208
        );
        assert_eq!(std::mem::offset_of!(PresentationFrame, viewport), 272);
        assert_eq!(std::mem::offset_of!(PresentationFrame, skinned), 288);
        assert_eq!(size_of::<CandidateData>(), 16);
        assert_eq!(size_of::<SkinnedData>(), 16);
        assert_eq!(size_of::<BatchData>(), 96);
        assert_eq!(std::mem::offset_of!(PresentationFrame, plane), 296);
        // Ten floats per pose, matching POSE_FLOATS in the shader.
        assert_eq!(size_of::<engine_core::NodePose>(), 40);
        assert_eq!(size_of::<engine_core::NodeInfo>(), 8);
        let module = naga::front::wgsl::parse_str(PRESENTATION_SHADER).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .expect("presentation shader validates");
        let size = |name: &str| {
            let (_, ty) = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(name))
                .unwrap();
            ty.inner.size(module.to_ctx())
        };
        assert_eq!(size("Frame"), 320);
        assert_eq!(std::mem::offset_of!(PresentationFrame, eye), 304);
        assert_eq!(size("Skinned"), size_of::<SkinnedData>() as u32);
        let (_, pose_floats) = module
            .constants
            .iter()
            .find(|(_, constant)| constant.name.as_deref() == Some("POSE_FLOATS"))
            .unwrap();
        assert!(matches!(
            module.global_expressions[pose_floats.init],
            naga::Expression::Literal(naga::Literal::U32(floats))
                if floats as usize * 4 == size_of::<engine_core::NodePose>()
        ));
        assert_eq!(size("Rows"), 48);
        assert_eq!(size("Composed") as u64, COMPOSED_BYTES);
        assert_eq!(size("Candidate"), 16);
        assert_eq!(size("Batch") as usize, size_of::<BatchData>());
        assert_eq!(size("DrawArgs") as u64, DRAW_ARGS_BYTES);
    }

    #[test]
    fn temporal_uniforms_match_wgsl() {
        let size = |source: &str, name: &str| {
            let module = naga::front::wgsl::parse_str(source).unwrap();
            let (_, ty) = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(name))
                .unwrap();
            ty.inner.size(module.to_ctx()) as usize
        };
        assert_eq!(size(SHADER, "Camera"), size_of::<CameraUniform>());
        assert_eq!(size(SHADER, "Lighting"), size_of::<LightingUniform>());
        assert_eq!(size(SHADOW_SHADER, "Rows"), size_of::<PoseData>());
        assert_eq!(
            size(TEMPORAL_SHADER, "Resolve"),
            size_of::<ResolveUniform>()
        );
        assert_eq!(size_of::<ResolveUniform>(), 16);
        // Culling marks moving slots with the bit the temporal vertex stage
        // tests and masks.
        let bits = |source: &str, name: &str| {
            let module = naga::front::wgsl::parse_str(source).unwrap();
            let (_, constant) = module
                .constants
                .iter()
                .find(|(_, constant)| constant.name.as_deref() == Some(name))
                .unwrap();
            match module.global_expressions[constant.init] {
                naga::Expression::Literal(naga::Literal::U32(value)) => value,
                ref other => panic!("{name} is {other:?}"),
            }
        };
        assert_eq!(bits(PRESENTATION_SHADER, "MOVED_SLOT"), MOVED_SLOT);
        assert_eq!(bits(SHADER, "MOVED_SLOT"), MOVED_SLOT);
        let exact =
            CameraUniform::exact(Mat4::from_scale(glam::Vec3::splat(2.0)), glam::Vec3::X, 7.5);
        assert_eq!(exact.previous_view_projection, exact.view_projection);
        assert_eq!(exact.jitter, [0.0; 4]);
        // Presentation time rides with the eye.
        assert_eq!(exact.position, [1.0, 0.0, 0.0, 7.5]);
        assert_eq!(
            Mat4::from_cols_array_2d(&exact.inverse_view_projection),
            Mat4::from_scale(glam::Vec3::splat(0.5))
        );
        assert_eq!(exact.options, [0.0; 4]);
        // Reflection probe faces (decision 043) are exact cameras that shade
        // with the far shadow maps only.
        let probe =
            CameraUniform::probe(Mat4::from_scale(glam::Vec3::splat(2.0)), glam::Vec3::X, 7.5);
        assert_eq!(probe.view_projection, exact.view_projection);
        assert_eq!(probe.previous_view_projection, exact.view_projection);
        assert_eq!(probe.position, exact.position);
        assert_eq!(probe.jitter, [0.0; 4]);
        assert_eq!(probe.options, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn jitter_translates_clip_positions_by_whole_units_of_w() {
        let camera = engine_core::Camera::default();
        let view_projection = camera.view_projection(1.5);
        let previous = view_projection * Mat4::from_translation(Vec3::X);
        let jitter = jitter_clip([0.25, -0.5], 200, 100);
        let temporal = CameraUniform::temporal(
            view_projection,
            camera.position,
            0.0,
            previous,
            jitter,
            200,
            100,
        );
        let rasterized = Mat4::from_cols_array_2d(&temporal.view_projection);
        for point in [Vec3::ZERO, Vec3::new(1.0, -2.0, -8.0)] {
            let clip = view_projection * point.extend(1.0);
            let shifted = rasterized * point.extend(1.0);
            let expected = clip + glam::Vec4::new(jitter[0], jitter[1], 0.0, 0.0) * clip.w;
            assert!(
                shifted.abs_diff_eq(expected, 1.0e-5),
                "{shifted} {expected}"
            );
        }
        // A quarter pixel right and half a pixel up, in clip units.
        assert_eq!(jitter, [0.0025, 0.01]);
        assert_eq!(
            temporal.previous_view_projection,
            previous.to_cols_array_2d()
        );
        assert_eq!(temporal.jitter, [0.0025, 0.01, 0.01, 0.02]);
    }

    #[test]
    fn jitter_covers_the_pixel_without_repeating_within_a_cycle() {
        let offsets: Vec<[f32; 2]> = (0..JITTER_SAMPLES).map(jitter_offset).collect();
        for (index, offset) in offsets.iter().enumerate() {
            assert!(
                offset.iter().all(|value| (-0.5..0.5).contains(value)),
                "{offset:?}"
            );
            assert!(!offsets[..index].contains(offset), "repeats {offset:?}");
        }
        // The cycle is balanced about the pixel center, and repeats.
        for axis in 0..2 {
            let mean: f32 =
                offsets.iter().map(|offset| offset[axis]).sum::<f32>() / JITTER_SAMPLES as f32;
            assert!(mean.abs() < 0.07, "axis {axis} mean {mean}");
        }
        assert_eq!(jitter_offset(JITTER_SAMPLES + 3), jitter_offset(3));
        // Half a pixel right and down is one clip unit per viewport pixel.
        assert_eq!(jitter_clip([0.5, 0.5], 100, 50), [0.01, -0.02]);
    }

    #[test]
    fn mip_chains_reach_one_texel_and_add_at_most_a_third() {
        assert_eq!(mip_levels(1, 1), 1);
        assert_eq!(mip_levels(64, 64), 7);
        assert_eq!(mip_levels(2048, 1), 12);
        assert_eq!(mip_levels(3, 5), 3);
        assert_eq!(mip_chain_bytes(1, 1), 0);
        // 64x64: 32x32 + 16x16 + 8x8 + 4x4 + 2x2 + 1x1 texels.
        assert_eq!(mip_chain_bytes(64, 64), 1365 * 4);
        // Odd and one-texel-high levels round down, never below one texel.
        assert_eq!(mip_chain_bytes(3, 5), (2 + 1) * 4);
        assert_eq!(mip_chain_bytes(2048, 1), 2047 * 4);
        // The CPU asset contract's dimension limit.
        let limit = 2048_u32;
        assert!(mip_chain_bytes(limit, limit) * 3 < u64::from(limit * limit) * 4);
    }

    #[test]
    fn frustum_uniform_matches_core_planes() {
        let camera = engine_core::Camera::default();
        let view_projection = camera.view_projection(1.5);
        let (planes, absolute) = frustum_uniform(view_projection);
        // A point in front of the camera is inside every plane.
        let inside = view_projection.inverse() * glam::Vec4::new(0.0, 0.0, 0.5, 1.0);
        let inside = inside / inside.w;
        for plane in planes {
            assert!(glam::Vec4::from(plane).dot(inside) > 0.0);
        }
        assert!(absolute.iter().flatten().all(|value| *value >= 0.0));
    }
}
