use bevy_ecs::prelude::{Component, Resource};
use engine_assets::{Material, MaterialHandle, Mesh, MeshHandle, Surface, Texture, TextureHandle};
use glam::{BVec4A, Mat4, Quat, Vec3, Vec4, Vec4Swizzles};

use crate::NodePose;
use std::sync::Arc;

/// Local translation, normalized rotation, and scale relative to [`crate::Parent`].
/// A root's local transform is also its world transform. The derived
/// [`crate::GlobalTransform`] preserves composed matrices, including shear.
/// Edit through ordinary mutable Bevy queries or [`crate::edit_transform`].
/// Normal change detection tracks writes; do not bypass it for transform data.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
#[component(
    on_add = crate::hierarchy::tracking::topology_changed,
    on_remove = crate::hierarchy::tracking::topology_changed)]
pub struct Transform {
    pub translation: Vec3,
    /// Keep this quaternion normalized.
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// The pose in upload layout, `[t.x, t.y, t.z, s.x, r.x, r.y, r.z, r.w,
    /// s.y, s.z]`: the hierarchy's cached copy doubles as GPU input.
    #[inline]
    pub(crate) fn to_pose(self) -> NodePose {
        let (t, r, s) = (self.translation, self.rotation, self.scale);
        [t.x, t.y, t.z, s.x, r.x, r.y, r.z, r.w, s.y, s.z]
    }

    #[inline]
    pub(crate) fn from_pose(pose: &NodePose) -> Self {
        let [tx, ty, tz, sx, rx, ry, rz, rw, sy, sz] = *pose;
        Self {
            translation: Vec3::new(tx, ty, tz),
            rotation: Quat::from_xyzw(rx, ry, rz, rw),
            scale: Vec3::new(sx, sy, sz),
        }
    }

    /// `Mat4::from_scale_rotation_translation(scale, rotation, translation)`,
    /// bit for bit except NaN payloads.
    ///
    /// Every lane performs the same IEEE operation as glam's SIMD
    /// `quat_to_axes` followed by each axis times its scale component, without
    /// extracting and reinserting scalar lanes (costly in browser Wasm). Pinned
    /// glam 0.30.9; revisit on math-library upgrades.
    #[cfg_attr(not(feature = "attribution"), inline)]
    #[cfg_attr(feature = "attribution", inline(never))]
    pub fn matrix(&self) -> Mat4 {
        const X: BVec4A = BVec4A::new(true, false, false, false);
        const Y: BVec4A = BVec4A::new(false, true, false, false);
        const Z: BVec4A = BVec4A::new(false, false, true, false);
        let q = Vec4::from(self.rotation);
        let q2 = q + q;
        // (xx, yy, zz), (xy, xz, yz) and (wx, wy, wz), with xx = x * (x + x).
        let squares = q * q2;
        let mixed = q.xxyw() * q2.yzzw();
        let w = (q.wwww() * q2).zyxw();
        // 1 - (yy + zz), 1 - (xx + zz), 1 - (xx + yy).
        let diagonal = Vec4::ONE - (squares.yxxw() + squares.zzyw());
        // (xy + wz, xz + wy, yz + wx) and (xy - wz, xz - wy, yz - wx).
        let sum = mixed + w;
        let difference = mixed - w;
        let axis = |x: Vec4, y: Vec4, z: Vec4| {
            Vec4::select(X, x, Vec4::select(Y, y, Vec4::select(Z, z, Vec4::ZERO)))
        };
        Mat4::from_cols(
            axis(diagonal, sum.xxzw(), difference.xxyw()) * self.scale.x,
            axis(difference, diagonal, sum) * self.scale.y,
            axis(sum.yyyy(), difference.zzzz(), diagonal) * self.scale.z,
            self.translation.extend(1.0),
        )
    }
}

/// A centered unit cube; its transform supplies position and dimensions.
#[derive(Component, Debug, Clone, Copy)]
pub struct Cube {
    /// Linear RGB color; cubes are always opaque and ignore alpha. Draw
    /// transparent geometry with a [`Mesh3d`] whose material has an opacity.
    pub color: [f32; 4],
}

/// An instance of immutable CPU geometry and an opaque material.
///
/// Handles resolve in the world's `AssetLibrary`. A missing asset skips this
/// entity during extraction. This component takes precedence over `Cube`.
#[derive(Component, Debug, Clone, Copy)]
pub struct Mesh3d {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}

/// The scene's single perspective camera, in a right-handed Y-up world.
#[derive(Resource, Debug, Clone, Copy)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(24.0, 19.0, 28.0),
            target: Vec3::ZERO,
            vertical_fov_radians: 55.0_f32.to_radians(),
            near: 0.1,
            far: 500.0,
        }
    }
}

impl Camera {
    /// The position [`Self::view_projection`] looks from: `position`, or
    /// the default position when it is not finite.
    pub fn eye(&self) -> Vec3 {
        if self.position.is_finite() {
            self.position
        } else {
            Self::default().position
        }
    }

    /// Builds a WebGPU-compatible projection with reversed depth (decision
    /// 042): the near plane maps to depth 1 and the far plane to 0, which a
    /// floating-point depth buffer resolves about evenly at every distance.
    /// Invalid aspect/lens values fall back to sensible defaults; a coincident
    /// target looks down -Z and a vertical direction uses Z as the up axis.
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        let (projection, position, direction, up) = self.lens(aspect);
        projection * Mat4::look_to_rh(position, direction, up)
    }

    /// [`Self::view_projection`] seen from the origin: the view and
    /// projection of positions relative to the eye (camera-relative
    /// rendering), exact however far the eye is from the world's origin.
    pub fn relative_view_projection(&self, aspect: f32) -> Mat4 {
        let (projection, _, direction, up) = self.lens(aspect);
        projection * Mat4::look_to_rh(Vec3::ZERO, direction, up)
    }

    /// The projection, eye, view direction and up axis of
    /// [`Self::view_projection`], invalid values replaced.
    fn lens(&self, aspect: f32) -> (Mat4, Vec3, Vec3, Vec3) {
        let defaults = Self::default();
        let aspect = if aspect.is_finite() && aspect > 0.0 {
            aspect
        } else {
            1.0
        };
        let fov = if self.vertical_fov_radians.is_finite()
            && self.vertical_fov_radians > 0.0
            && self.vertical_fov_radians < std::f32::consts::PI
        {
            self.vertical_fov_radians
        } else {
            defaults.vertical_fov_radians
        };
        let near = if self.near.is_finite() && self.near > 0.0 {
            self.near
        } else {
            defaults.near
        };
        let far = if self.far.is_finite() && self.far > near {
            self.far
        } else {
            near + defaults.far
        };
        let position = if self.position.is_finite() {
            self.position
        } else {
            defaults.position
        };
        let direction = (self.target - position)
            .try_normalize()
            .unwrap_or(Vec3::NEG_Z);
        // `eye` repeats this fallback.
        let up = if direction.dot(Vec3::Y).abs() > 0.999 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        // Swapping the planes reverses depth.
        (
            Mat4::perspective_rh(fov, aspect, far, near),
            position,
            direction,
            up,
        )
    }
}

/// One coarse host input batch. Movement axes are held in [-1, 1]; look values
/// are pointer deltas in CSS pixels, consumed in `Presentation`, not once per tick.
/// The host overwrites this resource before every `Engine::advance` call.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct InputState {
    pub move_forward: f32,
    pub move_right: f32,
    pub move_up: f32,
    pub look_yaw: f32,
    pub look_pitch: f32,
}

/// Multiplies one object's color channel by channel (decision 033): its
/// material's base color and opacity, or its cube's color. Factors are in
/// `[0, 1]`; others read as the nearest bound, and non-finite ones as 1.
/// Objects keep their material's draw groups, and changing a tint uploads
/// only that object's color: use it for per-object and per-frame colors
/// instead of a material per color.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Tint(pub [f32; 4]);

impl Tint {
    /// No change.
    pub const WHITE: Self = Self([1.0; 4]);

    /// `color` multiplied by this tint's sanitized factors.
    pub fn apply(&self, color: [f32; 4]) -> [f32; 4] {
        let mut tinted = color;
        for (channel, factor) in tinted.iter_mut().zip(self.0) {
            let factor = if factor.is_finite() {
                crate::math::clamp(factor, 0.0, 1.0)
            } else {
                1.0
            };
            *channel *= factor;
        }
        tinted
    }
}

impl Default for Tint {
    fn default() -> Self {
        Self::WHITE
    }
}

/// CPU data for one instance; this is deliberately not a GPU buffer layout.
#[derive(Debug, Clone, Copy)]
pub struct RenderInstance {
    pub mesh: MeshHandle,
    /// For materials with maps ([`Material::has_maps`], decision 031), the
    /// material whose maps the renderer binds: the first extracted material
    /// with an equal [`Material::binding`], listed in
    /// [`RenderSnapshot::materials`], so materials differing only in color
    /// and surface draw together. `None` draws from color and surface alone.
    pub material: Option<MaterialHandle>,
    pub model: Mat4,
    /// Linear color; alpha is the opacity (decision 026), 1 for opaque.
    /// Without a material, an opacity below 1 draws transparent; a
    /// material's alpha mode decides otherwise.
    pub color: [f32; 4],
    /// The material's surface response; cubes are [`Surface::MATTE`].
    pub surface: Surface,
    /// Skinned instances (decision 036): the offset in
    /// [`RenderSnapshot::joints`] of the first joint of their palette, which
    /// then positions their vertices instead of `model`. `None` draws rigidly.
    pub skin: Option<u32>,
}

/// One referenced mesh, retained independently of its library's lifetime.
#[derive(Debug, Clone)]
pub struct RenderMesh {
    pub handle: MeshHandle,
    pub mesh: Arc<Mesh>,
}

/// One referenced texture, retained independently of its library's lifetime.
#[derive(Debug, Clone)]
pub struct RenderTexture {
    pub handle: TextureHandle,
    pub texture: Arc<Texture>,
}

/// One referenced material with maps (decision 031), copied so it outlives
/// its library entry; its textures are in [`RenderSnapshot::textures`].
/// `version` is its [`engine_assets::AssetLibrary::material_version`]: renderers
/// keep what they built from a handle while its version stands.
#[derive(Debug, Clone, Copy)]
pub struct RenderMaterial {
    pub handle: MaterialHandle,
    pub material: Material,
    pub version: u64,
}

impl RenderInstance {
    /// Whether this instance draws in the transparent pass: its material's
    /// alpha mode decides, `material` being the instance's listed material;
    /// without one, an opacity below 1 (decision 026).
    pub fn is_transparent(&self, material: Option<&Material>) -> bool {
        match material {
            Some(material) => material.is_transparent(),
            None => self.color[3] < 1.0,
        }
    }
}

/// Owned render data, reusable across frames without borrowing the ECS world.
#[derive(Debug, Default)]
pub struct RenderSnapshot {
    pub view_projection: Mat4,
    /// The camera's position ([`Camera::eye`]), for view-dependent lighting.
    pub camera_position: Vec3,
    /// Presentation time ([`crate::FrameTime::elapsed_seconds`]) for animated
    /// surfaces such as ripples.
    pub elapsed_seconds: f32,
    /// The simulation time the frame presents ([`crate::PresentedTime`]),
    /// for what must line up with simulated objects, such as the sea's
    /// waves (decision 077).
    pub presented_seconds: f64,
    /// The scene's atmosphere and every effect's settings (decision 022),
    /// sanitized ([`crate::Environment::current`]): a missing atmosphere
    /// reads as the default one and a missing effect resource as off.
    pub environment: crate::Environment,
    /// This frame's local lights (decision 023), within the settings' budget.
    pub lights: Vec<crate::RenderLight>,
    /// Usable lights beyond the budget, or without a valid pose.
    pub dropped_lights: usize,
    /// This frame's particle emitters (decision 026), within the particle
    /// budget.
    pub emitters: Vec<crate::RenderEmitter>,
    /// Particles of emitters skipped beyond the budget.
    pub dropped_particles: usize,
    /// This frame's decals (decision 039), within the settings' budget.
    pub decals: Vec<crate::RenderDecal>,
    /// Materials with maps that decals draw with, each once, and their
    /// textures; they may repeat [`Self::materials`] and [`Self::textures`].
    pub decal_materials: Vec<RenderMaterial>,
    pub decal_textures: Vec<RenderTexture>,
    /// Usable decals beyond the budget.
    pub dropped_decals: usize,
    /// This frame's usable reflection probes (decision 043), nearest the
    /// camera first, within the settings' budget.
    pub reflection_probes: Vec<crate::RenderReflectionProbe>,
    /// Usable probes beyond the budget.
    pub dropped_reflection_probes: usize,
    /// This frame's terrains (decision 051): their resident tiles and the
    /// patches the camera draws.
    pub terrain: crate::RenderTerrains,
    /// This frame's deformation map and the stamps it presses (decision
    /// 083).
    pub deformation: crate::RenderDeformation,
    /// World streaming's state at this frame (decision 086).
    pub cells: crate::CellStats,
    /// Every instance's `color[3]` is its opacity (decision 026); see
    /// [`RenderInstance::is_transparent`].
    pub instances: Vec<RenderInstance>,
    /// Joint palettes of skinned instances (decision 036), each a skin's
    /// joints in order from its instance's [`RenderInstance::skin`] offset,
    /// followed by decals' poses (decision 039).
    pub joints: Vec<crate::RenderJoint>,
    /// Unique sources used by emitted or frustum-culled valid instances.
    /// Culled sources allow the renderer to retain already resident resources;
    /// they do not request uploads by themselves.
    pub meshes: Vec<RenderMesh>,
    /// Unique materials with maps used by emitted or frustum-culled valid
    /// instances (decision 031).
    pub materials: Vec<RenderMaterial>,
    /// Unique textures of those materials.
    pub textures: Vec<RenderTexture>,
    /// Every entity carrying Cube or Mesh3d, before transform/asset checks.
    pub render_candidates: usize,
    /// Valid instances surviving culling, before the output-instance budget.
    pub visible_instances: usize,
    /// Valid instances whose mesh bounds are proven outside the frustum.
    pub culled_instances: usize,
    pub dropped_instances: usize,
    /// Renderable entities skipped because a mesh, material, or texture is absent.
    pub missing_assets: usize,
    /// Renderable entities skipped because their derived transform is invalid.
    pub invalid_transforms: usize,
}
